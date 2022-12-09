import torch
from torch._C import Value
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
import numpy as np

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
        net = net.cuda()
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net

# Average edge features to face features 
def edgetoface(fe, meshes):
    # NOTE: M x E x C => M x F x C
    fe_f = []
    target_f = max([len(mesh.topology.faces.keys()) for mesh in meshes])
    for i in range(len(meshes)):
        mesh = meshes[i]
        # NOTE: Only necessary if mesh is not fully unpooled 
        edge_ids = torch.tensor(list(sorted(mesh.topology.edges.keys())))
        topo_edge_map = torch.zeros(torch.max(edge_ids)+1).long()
        topo_edge_map[edge_ids] = torch.arange(len(edge_ids))
        # Collect edge indices for each face 
        etof_inds = torch.tensor([[e.index for e in f.adjacentEdges()] for key, f in sorted(mesh.topology.faces.items())])
        etof = topo_edge_map[etof_inds]
    
        fe_f_e = torch.mean(fe[i, etof, :], dim=1)
        fe_f_e = torch.nn.functional.pad(fe_f_e, (0, 0, 0, target_f - len(fe_f_e)))
        assert fe_f_e.shape == (target_f, fe.shape[2]), f"Edge to face features error: target shape ({target_f, fe.shape[2]}), actual shape: {fe_f_e.shape}"
        fe_f.append(fe_f_e)
    fe_f = torch.stack(fe_f)
    return fe_f 

def define_classifier(opt, input_nc, ncf, gpu_ids, init_type, init_gain):
    net = None

    # MeshCNN architecture: U-Net with conditional segmentation module
    down_convs = [input_nc] + ncf
    up_convs = ncf[::-1]
        
    pool_res = []
    transfer = opt.transfer_data_off # False if turned off
    net = MeshSegmentation(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                            softmax=opt.softmax, transfer_data=transfer, drop_relu=opt.drop_relu,
                            time=opt.time, clamp=opt.clamp,
                            extrinsics=opt.extrinsic_features, selectdepth=opt.selectdepth, 
                            selectwidth=opt.selectwidth, binary_conv=opt.binary_conv,
                            extrinsic_cond=opt.extrinsic_condition_placement, selection_prediction=opt.selection_module,
                            resconv=opt.resconv, leakyrelu=opt.leakyrelu, ln=opt.layernorm, dropout=opt.dropout)
    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif opt.dataset_mode == "deepfeature":
        loss = torch.nn.TripletMarginLoss()
    else: 
        raise ValueError(f"No known loss associated with dataset mode: {opt.dataset_mode}!")
    return loss

class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, fcs=None, transfer_data=True, drop_relu=True, sigmoid=False,
                 global_pool=None, export_pool=False, ffn=False):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.ffn = None
        if ffn == True:
            self.ffn = FourierFeatureTransform(down_convs[0])
            down_convs[0] += 256 * 2 # FFN concats Fourier features to the input
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks, fcs=fcs, global_pool=global_pool, export_pool=export_pool)
        self.sigmoid = sigmoid
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data, drop_relu=drop_relu)

    def forward(self, x, meshes, layer, export_pool=False):
        if self.ffn is not None:
            x = self.ffn(x)
        fe, before_pool, conv_unpool, conv_unpool_weights = self.encoder((x, meshes), export_pool)
        if layer == "prepool":
            # IMPT: last encoding layer does NOT have pooling
            before_pool[-1] = fe
            return before_pool

        fe = self.decoder((fe, meshes), before_pool, conv_unpool, conv_unpool_weights, layer)
        if self.sigmoid == True:
            fe = nn.functional.hardsigmoid(fe)
        return fe

    def __call__(self, x, meshes, layer):
        return self.forward(x, meshes, layer)

# Only correct version of floodfill -- recursive  
# TODO: This creates islands... WHY????
def floodfill_scalar_rec(face, topo, scores, visited):
    visited.append(face)
    # NOTE: Can still only take VISITED nodes 
    fneighbors = torch.tensor([f.index for f in topo.faces[face].adjacentFaces() if f.index in visited])
    if len(fneighbors) == 0:
        neighbor_score = torch.tensor(1)
    else:
        neighbor_score = torch.max(scores[fneighbors])
    scores[face] = scores[face] - torch.nn.functional.relu(scores[face].detach() - neighbor_score.detach(), 0)
    new_faces = list(set(fneighbors.tolist()).difference(set(visited)))
    if len(new_faces) == 0:
        return True 
    for f in new_faces:
        floodfill_scalar_rec(f, topo, scores, visited)

# Good version (fully guarantess connectedness)
def floodfill_scalar_v2(mesh, scores, starting_face, previous_preds = None, threshold=0.5, debug=False):
    # if previous_preds is not None:
    #     new_scores = torch.clone(previous_preds)
    # else:
    new_scores = torch.zeros(len(scores), device=scores.device)
    new_scores[starting_face] = scores[starting_face]
    
    # new_scores = torch.clone(scores)
    # Max b/w current and previous preds 
    if previous_preds is not None:
        new_scores[starting_face] = scores[starting_face] + torch.nn.functional.relu(previous_preds[starting_face].detach() - scores[starting_face].detach())
        
    boundary = [f.index for f in mesh.topology.faces[starting_face].adjacentFaces() if scores[f.index] >= threshold]
    done = set([starting_face])
    # Debugging 
    if debug:
        boundary_count = 0
        import polyscope as ps 
        ps.init() 
        vs, fs, _ = mesh.export_soup()
        ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1, material='flat')
        anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[starting_face].adjacentVertices()], axis=0, keepdims=True)
        ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
    # ps_mesh.add_scalar_quantity("predscores", pred_scores.numpy(), defined_on='faces', enabled=True, cmap='coolwarm')
    while len(boundary) > 0:
        if debug:
            ps_mesh.add_scalar_quantity(f"scores{boundary_count}", new_scores.detach().cpu().numpy(), defined_on='faces', enabled=True)
            boundary_colors = np.zeros(len(fs))
            boundary_colors[boundary] = 1
            ps_mesh.add_scalar_quantity(f"boundary{boundary_count}", boundary_colors, defined_on='faces', enabled=True)
        # if previous_preds is not None:
        #     # If previous predictions available, then we consider full neighbor set 
        #     neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces()] for f in boundary]
        #     neighbor_scores = torch.max(new_scores[torch.tensor(neighbors)], dim=1)[0]
        # else:
        # NOTE: Need to consider only visited neighbors to avoid islands  
        # visited_neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces() if ftmp.index in done] for f in boundary]
        # NOTE: Using threshold lets us consider all neighbors without fear of islands 
        visited_neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces()] for f in boundary]
        neighbor_scores = torch.stack([torch.max(scores[torch.tensor(neighbors)]) if len(neighbors) > 0 else torch.tensor([1]).to(scores.device) for neighbors in visited_neighbors]).to(scores.device) 
        boundary_scores = scores[torch.tensor(boundary)]
        if previous_preds is not None:
            boundary_scores = boundary_scores + torch.nn.functional.relu(previous_preds[torch.tensor(boundary)].detach() - boundary_scores.detach())
        new_scores[torch.tensor(boundary)] = boundary_scores - torch.nn.functional.relu(boundary_scores.detach() - neighbor_scores.detach())
        done = done.union(set(boundary))
        # Get next set of boundary faces: only grow out to faces above threshold 
        boundary = list(set([f.index for n in boundary for f in mesh.topology.faces[n].adjacentFaces() if f.index not in done and scores[f.index] >= threshold]))
        
        if debug:
            boundary_count += 1
            ps.show() 
            
    return new_scores 

# Hard constraint: all previous prediction values must be kept 
# Basically no more guarantee of contiguity with consecutive anchors 
def floodfill_scalar_v1(mesh, scores, starting_face, previous_preds = None):
    # if previous_preds is not None:
    #     new_scores = torch.clone(previous_preds)
    # else:
    new_scores = torch.clone(scores)
    if previous_preds is not None:
        new_scores[starting_face] = torch.max(new_scores[starting_face], previous_preds[starting_face])
        # Hard enforcement: new scores should never be lower than previous preds 
        # new_scores = torch.maximum(new_scores, previous_preds)  
        
    boundary = [f.index for f in mesh.topology.faces[starting_face].adjacentFaces()]
    done = set([starting_face])
    # Debugging 
    # boundary_count = 0
    # import polyscope as ps 
    # ps.init() 
    # vs, fs, _ = mesh.export_soup()
    # ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1, material='flat')
    # anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[starting_face].adjacentVertices()], axis=0, keepdims=True)
    # ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
    # ps_mesh.add_scalar_quantity("predscores", pred_scores.numpy(), defined_on='faces', enabled=True, cmap='coolwarm')
    while len(boundary) > 0:
        # ps_mesh.add_scalar_quantity(f"scores{boundary_count}", new_scores.numpy(), defined_on='faces', enabled=True, cmap='coolwarm')
        # if previous_preds is not None:
        #     # If previous predictions available, then we consider full neighbor set 
        #     neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces()] for f in boundary]
        #     neighbor_scores = torch.max(new_scores[torch.tensor(neighbors)], dim=1)[0]
        # else:
        # NOTE: We ONLY consider neighbors whom we've already visited to propagate scores -> guarantees NO islands 
        visited_neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces() if ftmp.index in done] for f in boundary]
        neighbor_scores = torch.stack([torch.max(new_scores[torch.tensor(neighbors)]) if len(neighbors) > 0 else torch.tensor([1]).to(scores.device) for neighbors in visited_neighbors]) 
        boundary_scores = scores[torch.tensor(boundary)]
        if previous_preds is not None:
            boundary_scores = torch.maximum(boundary_scores, previous_preds[torch.tensor(boundary)])
        new_scores[torch.tensor(boundary)] = torch.minimum(boundary_scores, neighbor_scores)
        done = done.union(set(boundary))
        # Get next set of boundary faces 
        boundary = list(set([f.index for n in boundary for f in mesh.topology.faces[n].adjacentFaces() if f.index not in done]))
        # boundary_count += 1
        # ps.show() 
    
    if previous_preds is not None:
        new_scores = torch.maximum(new_scores, previous_preds) 
        assert torch.all(new_scores >= previous_preds)
        # prev_greater = torch.where(previous_preds > scores)[0]
        # prev_less = torch.where(previous_preds < scores)[0]
        # print(scores[prev_greater[:5]])
        # print(previous_preds[prev_greater[:5]])
        
        # print(scores[prev_less[:5]])
        # print(previous_preds[prev_less[:5]])
        
    return new_scores 

# Iteratively updates both scores and face features using local attention 
# Can initialize scores by passing them through, or initialize with full attention map using anchor query  
class FloodFillNetwork(nn.Module):
    def __init__(self, channels, num_heads, atn_layers=1, q_layers=1, k_layers=1, v_layers=1, pass_scores=False, mode="resid",
                 visit_scores = False, fftype='global', 
                 mlp=True, selectwidth = 256, selectdepth = 3,
                 device=torch.device("cpu"), time=False):
        super(FloodFillNetwork, self).__init__()
        self.time = time 
        self.fftype = fftype 
        self.pass_scores = pass_scores
        self.visit_scores = visit_scores
        self.mlp = mlp 
        atn = [] 
        for _ in range(atn_layers):
            atn.append(AnchorAttentionBlock(channels, num_heads, q_layers, k_layers, v_layers, mode=mode, 
                                            channel_add=int(self.visit_scores)))
        self.attention = nn.ModuleList(atn).to(device)
        self.num_heads = num_heads
        self.device = device
        
        if mlp:
            self.classification = SelectionPrediction(n_input=channels, width=selectwidth, depth=selectdepth)
        else:
            raise ValueError("We require floodfill network to have selection prediction MLP!")
            
    # If multiple anchors: run through each anchor iteratively and average scores 
    # Joint conditioning will take place in the first network 
    def forward(self, x, meshes, init_scores = None):        
        if self.time: 
            import time 
            t0 = time.time() 
            
        # Extract anchors from each mesh 
        # NOTE: We just take first anchor for now 
        init_anchors = [mesh.anchor_fs for mesh in meshes]
        
        # Need initial scores if building selection simultaneously 
        if self.pass_scores and init_scores is None: 
            init_scores = self.classification(x.transpose(2,1)) # M x F x 1
       
        # TODO: Fix below for multiple anchors 
        # if self.pass_scores:
        #     # Set initial scores by running full attention with anchors as keys/values 
        #     if init_scores is None and not self.mlp:
        #         anchor_x = torch.stack([x[i,:,init_anchors[i]] for i in range(len(init_anchors))]).unsqueeze(2)
        #         q_x = torch.clone(x)
        #         if self.visit_onehot == True: 
        #             anchor_x = torch.cat([anchor_x, torch.ones(anchor_x.shape[0], 1, anchor_x.shape[2])], dim=1)
        #         for atn in self.attention[:-1]:
        #             q_x, atnmap = atn(q_x, anchor_x, meshes)
        #         atnmap = self.attention[-1](q_x, anchor_x, return_map=True, normalize=True) # M*H x F x 1
        #         init_scores = torch.mean(atnmap.reshape(x.shape[0], self.num_heads, atnmap.shape[1], -1), dim=1) # M x F x 1
        #         # Map from [-1,1] to [0,1]
        #         init_scores = (init_scores + 1)/2
        #         assert torch.all(torch.logical_and(init_scores >= 0, init_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(init_scores, torch.linspace(0,1,5).float())}"
        #     if self.mlp and init_scores is None:
        #         init_scores = x.transpose(2,1)
        #         init_scores = self.classification(init_scores)
        #     if self.time: 
        #         print(f"Initial scores time: {time.time() - t0:0.5f} sec")
        #         t0 = time.time()
        
        if self.pass_scores:
            final_scores = torch.zeros_like(init_scores)
            
        final_features = torch.zeros_like(x) 
        for i in range(len(meshes)):
            mesh = meshes[i] 
            features = x[i,:,:len(mesh.topology.faces)] # C x F 
            facemat = torch.from_numpy(mesh.facemat)
            
            # NOTE: anchors is a LIST 
            anchors = init_anchors[i]
            if self.pass_scores:
                new_scores = torch.clone(init_scores[i][:len(mesh.topology.faces)]) # F x 1
            current_features = torch.clone(features)
            # visit_onehot = torch.zeros(1, current_features.shape[1], device=self.device)
            
            # Iterate until no more neighbors to visit 
            # New features: only consider features/scores from previous boundary iteration 
            # Update score based on min/max algos 
            boundary = anchors
            done = set(anchors)
            
            # Debugging 
            # boundary_count = 0
            # import polyscope as ps 
            # ps.init() 
            # vs, fs, _ = mesh.export_soup()
            # ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1, material='flat')
            # anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[anchor].adjacentVertices()], axis=0, keepdims=True)
            # ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
            # current_boundary = np.zeros(len(fs))
            # current_boundary[facemat[anchor]] = 1 
            # ps_mesh.add_scalar_quantity("bdryneighbors", current_boundary, defined_on='faces', enabled=True)
            # ps.show() 
            
            while len(boundary) > 0:
                # ps_mesh.add_scalar_quantity(f"scores{boundary_count}", new_scores[:len(fs)].detach().squeeze().numpy(), defined_on='faces', enabled=True, cmap='coolwarm', vminmax=(0,1))
                # For all boundary features: attend to using every face, including itself
                boundary_features = current_features[:, boundary].unsqueeze(0).transpose(2,0)
                if self.fftype == "local":
                    others = facemat[boundary]
                elif self.fftype == "global":
                    others = torch.stack([torch.arange(features.shape[1])] * len(boundary)) 
                else:
                    raise ValueError(f"No known ff type {self.fftype}")
                
                # Set one-hot for visited every iteration (can't let this dimension transform) 
                # Only set for other features b/c boundary are always new faces 
                if self.visit_scores == True: 
                    score_features = torch.cat([current_features, new_scores.transpose(1,0)], dim=0)
                    other_features = score_features[:, others].transpose(1,0) # B x C x F
                else:
                    other_features = current_features[:,others].transpose(1,0) # B x C x F

                for atn in self.attention:
                    boundary_features, atnmap = atn(boundary_features, other_features, [mesh] * other_features.shape[0])
                
                if self.pass_scores:
                    pred_scores = self.classification(boundary_features.squeeze(2))
                    assert torch.all(torch.logical_and(pred_scores >= 0, pred_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(pred_scores, torch.linspace(0,1,5, device=x.device).float())}"
                        
                    # We take max of neighbors and run min against predicted score that still backprops 
                    neighbors = torch.tensor([[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces()] for f in boundary])
                    neighbor_scores = torch.max(new_scores[neighbors], dim=1)[0] # B x 3
                    
                    boundary_scores = pred_scores - torch.nn.functional.relu(pred_scores.detach() - neighbor_scores)
                                      
                    # Only update features/scores at the end of each boundary iteration 
                    new_scores[boundary] = boundary_scores
                    
                done = done.union(set(boundary))
                current_features[:, torch.tensor(boundary)] = boundary_features.transpose(1,0).squeeze(2)
                # Get next set of boundary faces 
                boundary = list(set([f.index for n in boundary for f in mesh.topology.faces[n].adjacentFaces() if f.index not in done]))
                # boundary_count += 1
                # ps.show() 

            # Fill out features/scores for current mesh 
            if self.pass_scores:
                final_scores[i] = torch.nn.functional.pad(new_scores, (0,0,0,final_scores.shape[1] - new_scores.shape[0]))
            # Pad features back to original shape 
            final_features[i] = torch.nn.functional.pad(current_features, (0,x.shape[2] - current_features.shape[1],0,0))
        
        if not self.pass_scores:
            final_scores = self.classification(final_features.transpose(2,1))
            assert torch.all(torch.logical_and(final_scores >= 0, final_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(final_scores, torch.linspace(0,1,5, device=x.device).float())}"
        
        if self.time: 
            print(f"Floodfill ({final_features.shape}): {time.time() - t0:0.5f} sec")
            
        return final_features, final_scores

    def __call__(self, x, meshes, init_scores=None):
        return self.forward(x, meshes, init_scores)

# Floodfill using full attention for each anchor 
class FullFloodFillNetwork(nn.Module):
    def __init__(self, channels, num_heads, atn_layers=1, q_layers=1, k_layers=1, v_layers=1, pred_layers=0, mode="resid",
                 visit_onehot=False, device=torch.device("cpu"), time=False):
        super(FullFloodFillNetwork, self).__init__()
        self.time = time 
        self.num_heads = num_heads 
        self.visit_onehot = visit_onehot 
        self.device = device 
        atn = [] 
        for _ in range(atn_layers):
            # NOTE: We condition on ALL attention layers 
            atn.append(AnchorAttentionBlock(channels, num_heads, q_layers, k_layers, v_layers, mode=mode, kv_onehot=visit_onehot))
        self.attention = nn.ModuleList(atn).to(device)
        self.device = device
        self.mlp = None 
        if pred_layers > 0: 
            mlp = [] 
            for _ in range(pred_layers-1):
                mlp.append(nn.Linear(channels, channels))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(channels, 1))
            mlp.append(nn.Sigmoid())
            self.mlp = nn.ModuleList(mlp).to(device)
            
    def forward(self, x, meshes, init_scores = None):
        if self.time: 
            import time 
            t0 = time.time() 
                
        # Extract anchors from each mesh 
        # NOTE: We just take first anchor for now 
        init_anchors = [mesh.anchor_fs[0] for mesh in meshes]
        if init_scores is None and self.mlp:
            # M x C x F => M x F x C 
            init_scores = x.transpose(2,1)
            for mlp in self.mlp:
                init_scores = mlp(init_scores)
        elif init_scores is None:
            anchor_x = torch.stack([x[i,:,init_anchors[i]] for i in range(len(init_anchors))]).unsqueeze(2)
            if self.visit_onehot == True: 
                anchor_x = torch.cat([anchor_x, torch.ones(anchor_x.shape[0], 1, anchor_x.shape[2])], dim=1)
            q_x = torch.clone(x)
            for atn in self.attention[:-1]:
                q_x, atnmap = atn(q_x, anchor_x, meshes)
            atnmap = self.attention[-1](q_x, anchor_x, return_map=True, normalize=True) # M*H x F x 1
            # Map from [-1,1] to [0,1]
            atnmap = (atnmap + 1)/2
            init_scores = torch.mean(atnmap.reshape(x.shape[0], self.num_heads, atnmap.shape[1], -1), dim=1) # M x F x 1
            assert torch.all(torch.logical_and(init_scores >= 0, init_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(init_scores, torch.linspace(0,1,5, device=x.device).float())}"
        if self.time: 
            print(f"Initial scores time: {time.time() - t0:0.5f} sec")
            t0 = time.time()
        
        final_scores = [] 
        final_features = [] 
        for i in range(len(meshes)):
            mesh = meshes[i] 
            features = x[i] # C x F 
            anchor = init_anchors[i]
            scores = init_scores[i]

            new_scores = torch.clone(scores)
            new_features = torch.clone(features)
            
            # Iterate until no more neighbors to visit 
            # New scores: only consider scores from previous boundary iteration 
            # Update score based on min/max algos 
            boundary = [anchor] 
            done = set([anchor])
            
            # Debugging 
            # boundary_count = 0
            # import polyscope as ps 
            # ps.init() 
            # vs, fs, _ = mesh.export_soup()
            # ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1, material='flat')
            # anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[anchor].adjacentVertices()], axis=0, keepdims=True)
            # ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
            # ps_mesh.add_scalar_quantity("initscores", new_scores[:len(fs)].detach().squeeze().numpy(), defined_on='faces', enabled=True, cmap='coolwarm', vminmax=(0,1))
            
            while len(boundary) > 0:
                # ps_mesh.add_scalar_quantity(f"scores{boundary_count}", new_scores[:len(fs)].detach().squeeze().numpy(), defined_on='faces', enabled=True, cmap='coolwarm', vminmax=(0,1))
                # For all boundary features: attend to using every other face 
                boundary_features = new_features[:, boundary].unsqueeze(0)
                others = list(set(range(len(features))).difference(set(boundary)))
                # Set one-hot for visited every iteration (can't let this dimension transform) 
                # Only set for other features b/c boundary are always new faces 
                if self.visit_onehot == True: 
                    visit_onehot = torch.zeros(1, new_features.shape[1]).to(self.device)
                    visit_onehot[:,list(done)] = 1
                    onehot_features = torch.cat([new_features, visit_onehot], dim=0)
                    other_features = onehot_features[:, others].unsqueeze(0)
                else:
                    other_features = new_features[:,others].unsqueeze(0)
                
                for atn in self.attention[:-1]:
                    boundary_features, atnmap = atn(boundary_features, other_features, [mesh] * other_features.shape[0])
                if not self.mlp:
                    # Final attention map needs to be normalized to interpret as similarity score 
                    atnmap = self.attention[-1](boundary_features, other_features, normalize=True, return_map=True)
                    # Map from [-1,1] to [0,1]
                    atnmap = (atnmap + 1)/2
                    # H x B x N => => M x H * B * N => M x 1
                    pred_scores = torch.mean(atnmap.reshape(len(boundary_features), -1), dim=1)
                else: 
                    boundary_features, atnmap = self.attention[-1](boundary_features, other_features)
                    pred_scores = self.mlp[0](boundary_features.transpose(2,1))
                    for mlp in self.mlp[1:]:
                        pred_scores = mlp(pred_scores)
                assert torch.all(torch.logical_and(pred_scores >= 0, pred_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(pred_scores, torch.linspace(0,1,5, device=x.device).float())}"
                    
                # NOTE: When we consider the max of the neighbors, we only want to look at the neighbors already traversed 
                visited_neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces() if ftmp.index in done] for f in boundary]
                neighbor_scores = torch.stack([torch.max(new_scores[torch.tensor(neighbors)]) if len(neighbors) > 0 else torch.tensor([1]).to(x.device) for neighbors in visited_neighbors]).unsqueeze(1)
                # |B| length tensors 
                # Each face is only seen once, so the predicted scores are always compared against the init scores 
                boundary_scores = torch.minimum(torch.maximum(pred_scores, new_scores[torch.tensor(boundary)]),
                                                                    neighbor_scores)
                done = done.union(set(boundary))
                
                # Only update features/scores at the end of each boundary iteration 
                new_features[:, torch.tensor(boundary)] = boundary_features 
                new_scores[boundary] = boundary_scores
                # Get next set of boundary faces 
                boundary = list(set([f.index for n in boundary for f in mesh.topology.faces[n].adjacentFaces() if f.index not in done]))
                # boundary_count += 1
                # ps.show() 
            final_scores.append(new_scores)
            final_features.append(new_features)
            
        final_scores = torch.stack(final_scores)
        final_features = torch.stack(final_features)
        
        if self.time: 
            print(f"Floodfill ({final_features.shape}): {time.time() - t0:0.5f} sec")
            
        return final_features, final_scores

    def __call__(self, x, meshes, init_scores=None):
        return self.forward(x, meshes, init_scores)
    
# Testing same setup except no score passing  
class FullFloodFillNetworkv2(nn.Module):
    def __init__(self, channels, num_heads, atn_layers=1, q_layers=1, k_layers=1, v_layers=1, pred_layers=0, mode="resid",
                 visit_onehot=False, device=torch.device("cpu"), time=False):
        super(FullFloodFillNetworkv2, self).__init__()
        self.time = time 
        self.num_heads = num_heads 
        self.visit_onehot = visit_onehot 
        self.device = device 
        atn = [] 
        for _ in range(atn_layers):
            # NOTE: We condition on ALL attention layers 
            atn.append(AnchorAttentionBlock(channels, num_heads, q_layers, k_layers, v_layers, mode=mode, kv_onehot=visit_onehot))
        self.attention = nn.ModuleList(atn).to(device)
        self.mlp = None 
        if pred_layers > 0: 
            mlp = [] 
            for _ in range(pred_layers-1):
                mlp.append(nn.Linear(channels, channels))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(channels, 1))
            mlp.append(nn.Sigmoid())
            self.mlp = nn.ModuleList(mlp).to(device)
            
    def forward(self, x, meshes, init_scores = None):
        if self.time: 
            import time 
            t0 = time.time() 
        
        init_anchors = [mesh.anchor_fs[0] for mesh in meshes]
        final_scores = [] 
        final_features = [] 
        for i in range(len(meshes)):
            mesh = meshes[i] 
            features = x[i] # C x F 
            anchor = init_anchors[i]
            new_features = torch.clone(features)
            visit_onehot = torch.zeros(1, new_features.shape[1], device=self.device)
            # Iterate until no more neighbors to visit 
            # New scores: only consider scores from previous boundary iteration 
            # Update score based on min/max algos 
            boundary = [anchor] 
            done = set([anchor])
            
            # Debugging 
            # boundary_count = 0
            # import polyscope as ps 
            # ps.init() 
            # vs, fs, _ = mesh.export_soup()
            # ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1, material='flat')
            # anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[anchor].adjacentVertices()], axis=0, keepdims=True)
            # ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
            
            while len(boundary) > 0:
                # boundary_inds = np.zeros(len(fs))
                # boundary_neighbors = np.array([[f] + [ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces()] for f in boundary]).flatten()
                # boundary_inds[boundary_neighbors] = 2
                # boundary_inds[boundary] = 1 
                # ps_mesh.add_scalar_quantity(f"boundary{boundary_count}", boundary_inds, defined_on='faces', enabled=True)
                
                # For all boundary features: attend to using every other face 
                boundary_features = new_features[:, boundary].unsqueeze(0)
                # others = list(set(range(len(features))).difference(set(boundary)))
                # NOTE: We do FULL SELF ATTENTION
                others = list(set(range(len(features))))
                # Set one-hot for visited every iteration (can't let this dimension transform) 
                # Only set for other features b/c boundary are always new faces 
                if self.visit_onehot == True: 
                    visit_onehot[:,list(done)] = 1
                    onehot_features = torch.cat([new_features, visit_onehot], dim=0)
                    other_features = onehot_features[:, others].unsqueeze(0)
                else:
                    other_features = new_features[:,others].unsqueeze(0)
                
                # TODO: Do we care about padding here??? 
                for atn in self.attention[:-1]:
                    boundary_features, atnmap = atn(boundary_features, other_features, [mesh] * other_features.shape[0])
                    
                done = done.union(set(boundary))
                
                # Only update features/scores at the end of each boundary iteration 
                new_features[:, torch.tensor(boundary)] = boundary_features 
                # Get next set of boundary faces 
                boundary = list(set([f.index for n in boundary for f in mesh.topology.faces[n].adjacentFaces() if f.index not in done]))
                # boundary_count += 1
                # ps.show() 
            final_features.append(new_features)
        
        # Get final scores by simply running one last attention over final features 
        final_features = torch.stack(final_features)
        if not self.mlp:
            anchor_x = torch.stack([final_features[i,:,init_anchors[i]] for i in range(len(init_anchors))]).unsqueeze(2)
            if self.visit_onehot == True: 
                anchor_x = torch.cat([anchor_x, torch.ones(anchor_x.shape[0], 1, anchor_x.shape[2], device=self.device)], dim=1)
            q_x = torch.clone(x)
            for atn in self.attention[:-1]:
                q_x, atnmap = atn(q_x, anchor_x, meshes)
            # NOTE: QKV normalized across each head 
            atnmap = self.attention[-1](q_x, anchor_x, return_map=True, normalize=True) # M*H x F x 1
            # Map from [-1,1] to [0,1]
            atnmap = (atnmap + 1)/2
            final_scores = torch.mean(atnmap.reshape(x.shape[0], self.num_heads, atnmap.shape[1], -1), dim=1) # M x F x 1
        else:
            final_scores = self.mlp[0](final_features.transpose(2,1))
            for mlp in self.mlp[1:]:
                final_scores = mlp(final_scores)
        assert torch.all(torch.logical_and(final_scores >= 0, final_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(init_scores, torch.linspace(0,1,5, device=x.device).float())}"
                
        if self.time: 
            print(f"Floodfill ({final_features.shape}): {time.time() - t0:0.5f} sec")
            
        return final_features, final_scores

    def __call__(self, x, meshes, init_scores=None):
        return self.forward(x, meshes, init_scores)
    
# Floodfill using conv operation using linear layers in practice concatenated with neighbors 
class ConvFloodFillNetwork(nn.Module):
    def __init__(self, channels, n_layers=1, pred_layers=1, device=torch.device("cpu"), time=False):
        super(ConvFloodFillNetwork, self).__init__()
        self.time = time 
        convs = [] 
        for _ in range(n_layers-1):
            convs.append(nn.Linear(channels * 4, channels*4)) # Convolve 4 face features, neighbors + itself 
            convs.append(nn.InstanceNorm1d(channels))
            convs.append(nn.Relu())
        convs.append(nn.Linear(channels*4, channels))
        convs.append(nn.InstanceNorm1d(channels))
        self.convs = nn.ModuleList(convs).to(device)
        
        self.device = device
        self.mlp = None 
        if pred_layers > 0: 
            mlp = [] 
            for _ in range(pred_layers-1):
                mlp.append(nn.Linear(channels, channels))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(channels, 1))
            mlp.append(nn.Sigmoid())
            self.mlp = nn.ModuleList(mlp).to(device)
        else:
            raise ValueError("Need at least one MLP layer to map conv ff to scalar!")
            
    # NOTE: MLP to scalar is REQUIRED for this version 
    def forward(self, x, meshes, init_scores = None):
        if self.time: 
            import time 
            t0 = time.time() 
        
        # Extract anchors from each mesh 
        # NOTE: We just take first anchor for now 
        init_anchors = [mesh.anchor_fs[0] for mesh in meshes]
        if init_scores is None:
            # M x C x F => M x F x C 
            init_scores = x.transpose(2,1)
            for mlp in self.mlp:
                init_scores = mlp(init_scores)
        if self.time: 
            print(f"Initial scores time: {time.time() - t0:0.5f} sec")
            t0 = time.time()
            
        final_scores = [] 
        final_features = [] 
        # TODO: Batch updating over all meshes 
        for i in range(len(meshes)):
            mesh = meshes[i] 
            features = x[i].transpose(1,0) # C x F 
            anchor = init_anchors[i]
            new_scores = torch.clone(init_scores[i])
            current_features = torch.clone(features)

            # Iterate until no more neighbors to visit 
            # New scores: only consider scores from previous boundary iteration 
            # Update score based on min/max algos 
            boundary = [anchor] 
            done = set([anchor])
            
            # Debugging 
            # boundary_count = 0
            # import polyscope as ps 
            # ps.init() 
            # vs, fs, _ = mesh.export_soup()
            # ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1, material='flat')
            # anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[anchor].adjacentVertices()], axis=0, keepdims=True)
            # ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
            # ps_mesh.add_scalar_quantity("initscores", new_scores[:len(fs)].detach().squeeze().numpy(), defined_on='faces', enabled=True, cmap='coolwarm', vminmax=(0,1))
            
            while len(boundary) > 0:
                # ps_mesh.add_scalar_quantity(f"scores{boundary_count}", new_scores[:len(fs)].detach().squeeze().numpy(), defined_on='faces', enabled=True, cmap='coolwarm', vminmax=(0,1))
                boundary_neighbors = [[f] + [ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces()] for f in boundary]
                neighbor_features = features[boundary_neighbors,:]
                boundary_features = features[boundary,:]
                
                # Stack accordingly 
                boundary_features = torch.cat([boundary_features, neighbor_features], dim=-1)
                for conv in self.convs:
                    boundary_features = conv(boundary_features)
                
                pred_scores = self.mlp[0](boundary_features)
                for mlp in self.mlp[1:]:
                    pred_scores = mlp(pred_scores)
                assert torch.all(torch.logical_and(pred_scores >= 0, pred_scores <= 1)), f"Not all scores within [0,1]! \nQuantile: {torch.quantile(pred_scores, torch.linspace(0,1,5, device=x.device).float())}"
                
                # NOTE: When we consider the max of the neighbors, we only want to look at the neighbors already traversed 
                visited_neighbors = [[ftmp.index for ftmp in mesh.topology.faces[f].adjacentFaces() if ftmp.index in done] for f in boundary]
                neighbor_scores = torch.stack([torch.max(new_scores[torch.tensor(neighbors)]) if len(neighbors) > 0 else torch.tensor([1]).to(x.device) for neighbors in visited_neighbors]).unsqueeze(1)
                # |B| length tensors 
                # In this case: new_scores = old scores up to previous iteration 
                boundary_scores = torch.minimum(torch.maximum(pred_scores, new_scores[torch.tensor(boundary)]),
                                                                    neighbor_scores)
                done = done.union(set(boundary))
                
                # Only update features/scores at the end of each boundary iteration 
                current_features[:, torch.tensor(boundary)] = boundary_features 
                new_scores[boundary] = boundary_scores
                # Get next set of boundary faces 
                boundary = list(set([f.index for n in boundary for f in mesh.topology.faces[n].adjacentFaces() if f.index not in done]))
                # boundary_count += 1
                # ps.show() 
            final_scores.append(new_scores)
            final_features.append(current_features)
            
        final_scores = torch.stack(final_scores)
        final_features = torch.stack(final_features)
        
        if self.time: 
            print(f"Floodfill ({final_features.shape}): {time.time() - t0:0.5f} sec")
            
        return final_features, final_scores

    def __call__(self, x, meshes, init_scores=None):
        return self.forward(x, meshes, init_scores)

class MeshSegmentation(nn.Module):
    """Network for interactive segmentation
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, fcs=None, transfer_data=True, drop_relu=True,
                 export_pool=False, condition=None, unpool="original", time=False, poolorder='norm', ffn=False,
                 transform_condition=False, cond_output_dim=None, extrinsics=None, clamp="sigmoid", predict="face",
                 selectwidth=256, selectdepth=3, extrinsic_cond = "post", selection_prediction=False, binary_conv=False,
                 softmax=False, attention_cutoff=None, n_attention_heads=0, resconv=False, leakyrelu=False, ln=False, 
                 dropout=False):
        super(MeshSegmentation, self).__init__()
        self.condition = condition
        self.unpool = unpool
        self.transfer_data = transfer_data
        self.ffn = None
        self.time = time
        self.selection = selection_prediction
        self.binary_conv = binary_conv 
        self.softmax = softmax 
        self.predict = predict 
        self.norm = InstanceNormPad()
        
        # NOTE: # convs MUST be at least one greater than # pools 
        assert len(down_convs) > len(pools), f"# of convs {len(down_convs)} must be greater than # pools {len(pools)}"
        
        condition_dims = 0
        if condition == "input":
            condition_dims += down_convs[0]
        elif condition == "coarse":
            condition_dims += down_convs[-1]
        elif condition == "upsample":
            condition_dims += up_convs[-1]
        elif condition == "position":
            condition_dims += 3
        elif condition is not None:
            raise NotImplementedError(f"Condition setting {condition} not implemented!")
        
        # Whether to pass conditioning feature through MLP 
        if transform_condition == False:
            cond_output_dim = None
        elif cond_output_dim is None and condition is not None:
            cond_output_dim = condition_dims
                        
        # Additional conditioning from extrinsic features 
        self.extrinsics = extrinsics 
        self.extrinsic_cond = extrinsic_cond 
        extrinsic_cond_dims = 0 
        if extrinsics is not None and self.extrinsic_cond == "post": 
            for extr in extrinsics: 
                if extr == "coord":
                    extrinsic_cond_dims += 3
                if extr == "geodesic":
                    extrinsic_cond_dims += 1
                if extr == "normal":
                    extrinsic_cond_dims += 3
                if extr == "anchornormal":
                    extrinsic_cond_dims += 3
                if extr == "dihedral":
                    extrinsic_cond_dims += 1
        if ffn == True:
            self.ffn = FourierFeatureTransform(down_convs[0])
            down_convs[0] += 256 * 2 # FFN concats Fourier features to the input
        # Sometimes will condition MeshCNN input by extrinsic features 
        # down_convs[0] += extrinsic_input_conds
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks, fcs=fcs, export_pool=export_pool, poolorder=poolorder, 
                                   attention_cutoff=attention_cutoff, n_attention_heads=n_attention_heads, resconv=resconv,
                                   leakyrelu=leakyrelu, ln=ln)
        unrolls = pools.copy()
        unrolls.reverse()
        # Pad out unrolls to match number of convolutions
        unrolls = [0] * max((len(up_convs) - len(unrolls) - 1), 0) + unrolls 
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data, drop_relu=drop_relu, 
                                   binary_conv=binary_conv, attention_cutoff=attention_cutoff, n_attention_heads=n_attention_heads,
                                   resconv=resconv, leakyrelu=leakyrelu, ln=ln)
        if self.selection == True: 
            self.classification = SelectionPrediction(n_input=up_convs[-1] + condition_dims + extrinsic_cond_dims, clamp=clamp, width=selectwidth, 
                                                    depth=selectdepth, dropout=dropout)
    def forward(self, x, meshes, layer, export_pool):
        if self.condition == "input":
            input_x = torch.clone(x) 
        
        x = self.norm(x, meshes) 
        if self.ffn is not None:
            x = self.ffn(x)
        
        #### ENCODER ####
        fe, before_pool, conv_unpool, conv_unpool_weights = self.encoder((x, meshes), export_pool, time=self.time)
        
        if self.unpool == "original":
            conv_unpool_weights = None
        if layer == "prepool":
            # IMPT: last encoding layer does NOT have pooling
            before_pool[-1] = fe
            return before_pool
        # condition_source = None
        # if self.condition == "input":
        #     condition_source = x
        # elif self.condition == "coarse":
        #     condition_source = fe
        # elif self.condition == "upsample":
        #     condition_source = "upsample"
        # Pad unpool edits and weights if necessary 
        conv_unpool = [None] * max((len(before_pool) - len(conv_unpool) - 1), 0) + conv_unpool
        if conv_unpool_weights:
            conv_unpool_weights = [None] * max((len(before_pool) - len(conv_unpool_weights) - 1), 0) + conv_unpool_weights
         
        #### DECODER ####
        fe = self.decoder((fe, meshes), before_pool[1:], conv_unpool, conv_unpool_weights, time=self.time)
    
        if self.binary_conv == True: 
            fe, pre_binary = fe 
            
        if self.time==True:
            import time
            t0 = time.time()
        # Condition 
        if self.condition is not None:
            # if self.condition == "position": 
            #     # M x 3 x E 
            #     condition_fe = torch.zeros(fe.shape[0], 3, fe.shape[2])
            #     for i, mesh in enumerate(meshes):
            #         anchor_fs = mesh.anchor_fs
            #         # Take average of positions of all anchors faces
            #         anchor_pos = [mesh.vertices[v.index] for f in anchor_fs for v in mesh.topology.faces[f].adjacentVertices()]
            #         avg_anchor_pos = torch.from_numpy(np.mean(anchor_pos, axis=0))
            #         condition_fe[i] = avg_anchor_pos # broadcasts to shape 
            if self.condition == "input":
                condition_fe = condition_edge_features(fe, meshes, input_x, device=fe.device)
            if self.condition == "upsample":
                condition_fe = condition_edge_features(fe, meshes, fe, device=fe.device)
            # M x (C + C_cond) x E 
            fe = torch.cat([fe, condition_fe], dim=1) 
                  
        # Condition with extrinsics 
        if self.extrinsics is not None and self.extrinsic_cond == "post": 
            upconv_c = fe.shape[1]
            for extr in self.extrinsics: 
                if extr == "onehot":
                    condition_fe = torch.zeros(fe.shape[0], 1, fe.shape[2], device=fe.device)
                    for i, mesh in enumerate(meshes):
                        anchor_edges = [e.index for f in mesh.anchor_fs for e in mesh.topology.faces[f].adjacentEdges()]
                        condition_fe[i][:,anchor_edges] = 1
                if extr == "geodesic":
                    condition_fe = torch.zeros(fe.shape[0], 1, fe.shape[2], device=fe.device)
                    for i, mesh in enumerate(meshes):
                        edge_geodesics = torch.from_numpy(mesh.anchor_edge_geodesics).reshape(1, len(mesh.anchor_edge_geodesics))
                        # Pad out to standard length 
                        edge_geodesics = torch.nn.functional.pad(edge_geodesics, (0, fe.shape[2] - edge_geodesics.shape[1], 0, 0))
                        condition_fe[i] = edge_geodesics
                if extr == "normal":
                    condition_fe = torch.zeros(fe.shape[0], 3, fe.shape[2], device=fe.device)
                    for i, mesh in enumerate(meshes):
                        edgenormals = torch.from_numpy(mesh.edgenormals).transpose(1, 0)
                        # Pad out to standard length 
                        edgenormals = torch.nn.functional.pad(edgenormals, (0, fe.shape[2] - edgenormals.shape[1], 0, 0))
                        condition_fe[i] = edgenormals
                if extr == "anchornormal":
                    # Also condition using anchor faces
                    condition_fe = torch.zeros(x.shape[0], 3, x.shape[2], device=fe.device)
                    for i, mesh in enumerate(meshes):
                        anchor_edgenormals = torch.from_numpy(np.mean(mesh.facenormals[mesh.anchor_fs], axis=0))
                        condition_fe[i] = anchor_edgenormals.unsqueeze(1)
                if extr == "dihedral": 
                    condition_fe = torch.zeros(fe.shape[0], 1, fe.shape[2], device=fe.device)
                    for i, mesh in enumerate(meshes):
                        # Condition on dihedral b/w anchor and edge 
                        facenormals = torch.from_numpy(mesh.facenormals)
                        anchor_normal = torch.mean(facenormals[mesh.anchor_fs], dim=0)
                        cosTheta = torch.clip((torch.from_numpy(mesh.edgenormals) * anchor_normal.reshape(1,3)).sum(dim=1), -1, 1)
                        dihedrals = (np.pi - torch.arccos(cosTheta)).reshape(1, len(cosTheta))
                        # Pad out to standard length 
                        dihedrals = torch.nn.functional.pad(dihedrals, (0, fe.shape[2] - dihedrals.shape[1], 0, 0))
                        condition_fe[i] = dihedrals 
                fe = torch.cat([fe, condition_fe], dim=1)
            # Debug: check values of anchors 
            # for i in range(len(meshes)): 
            #     mesh = meshes[i] 
            #     anchor_fs = mesh.anchor_fs 
            #     anchor_e = [e.index for e in mesh.topology.faces[anchor_fs[0]].adjacentEdges()]
            #     anchor_fe = fe[i,:,anchor_e].transpose(1,0)
            #     print(anchor_fe[:, upconv_c:])
            # raise 
        if self.time==True:
            print(f"Edge to vertex feature conversion time: {time.time() - t0} sec.")
            
        # Reshape edge features to apply classification MLP 
        fe = fe.transpose(2, 1) # M x C x E => M x E x C 
        
        # Convert edge features to face features (if toggled)
        if self.predict == 'face':
            fe = edgetoface(fe, meshes)
            
        # Segmentation prediction: allow only deep features 
        fe_seg = None 
        if self.selection == True:
            fe_seg = self.classification(fe, self.time)
        elif self.binary_conv == True:
            # Cast to sigmoid  
            fe_seg = torch.sigmoid(fe)
            fe = pre_binary.transpose(2,1)

        # Returns both deep features and segmentation prediction  
        return fe_seg, fe.transpose(2,1)

    def __call__(self, x, meshes, layer = None, export_pool = False):
        return self.forward(x, meshes, layer, export_pool)
    
class InstanceNormPad(nn.Module):
    def __init__(self):
        super(InstanceNormPad, self).__init__()
        
    def __call__(self, x, meshes):
        return self.forward(x, meshes)

    def forward(self, x, meshes):
        # Don't do anything if x has only 1 spatial dimension
        if x.shape[2] <= 1:
            return x 
        
        # First construct mask 
        mask = torch.zeros_like(x) 
        for i in range(len(meshes)):
            mesh = meshes[i] 
            mask[i, :, :len(mesh.topology.edges)] = 1 
        counts = torch.sum(mask, dim = 2, keepdim=True)
        assert torch.all(counts > 0), f"Found zero count rows. {torch.where(counts == 0)[0]}. X shape: {x.shape}. Meshes: {len(meshes)}."
        
        # Compute means 
        means = torch.sum(x, dim=2, keepdim=True)/counts 
        
        # Compute stds manually 
        stds = torch.sqrt(torch.sum((x - means)**2/counts * mask, dim=2, keepdim=True)) + 1e-5
        
        if torch.any(stds <= 5e-5):
            # print(f"Small stds found for x shape {x.shape}. Quantile: {torch.quantile(stds, torch.linspace(0,1,5).float())}")
            return x 
        
        # Padded normalization 
        x = (x - means)/stds
        return x 

class LayerNormPad(nn.Module):
    def __init__(self):
        super(LayerNormPad, self).__init__()
        
    def __call__(self, x, meshes):
        return self.forward(x, meshes)

    def forward(self, x, meshes):
        # Don't do anything if x has only 1 spatial dimension and 1 channel 
        if x.shape[2] <= 1 and x.shape[1] <= 1:
            return x 
        
        # First construct mask 
        mask = torch.zeros_like(x) 
        for i in range(len(meshes)):
            mesh = meshes[i]
            mask[i, :, :len(mesh.topology.edges)] = 1 
        counts = torch.sum(mask, dim = (1,2), keepdim=True)
        assert torch.all(counts > 0), f"Found zero count rows. {torch.where(counts == 0)[0]}. X shape: {x.shape}. Meshes: {len(meshes)}."
        
        # Compute means 
        means = torch.sum(x, dim=(1,2), keepdim=True)/counts 
        
        # Compute stds manually 
        stds = torch.sqrt(torch.sum((x - means)**2/counts * mask, dim=(1,2), keepdim=True)) + 1e-8
        
        if torch.any(stds <= 1e-7):
            # print(f"Small stds found for x shape {x.shape}. Quantile: {torch.quantile(stds, torch.linspace(0,1,5).float())}")
            return x 
        
        # Padded normalization 
        x = (x - means)/stds
        return x 

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, leakyrelu=False, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        if leakyrelu:
            self.relu = nn.ReLU()
        else: 
            self.relu = nn.LeakyReLU()
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = self.relu(x)
        return x
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0, poolorder="norm", resconv=False, leakyrelu=False, ln=False):
        super(DownConv, self).__init__()
        self.bn = []
        self.pool = None
        if resconv:
            convop = MResConv 
        else: 
            convop = MeshConv 
        if leakyrelu: 
            self.relu = nn.LeakyReLU() 
        else: 
            self.relu = nn.ReLU()
        self.conv1 = convop(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(convop(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            if ln: 
                self.bn.append(LayerNormPad())
            else:
                self.bn.append(InstanceNormPad())
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool, order=poolorder)

    def __call__(self, x, time=False):
        return self.forward(x, time)

    def forward(self, x, time_debug):
        fe, meshes = x
        if time_debug == True:
            import time
            t0 = time.time()
        x1 = self.conv1(fe, meshes)
        if time_debug==True:
            print(f"Conv ({x1.shape}): {time.time()-t0} sec")
        if self.bn and x1.shape[2] > 1:
            x1 = self.bn[0](x1, meshes)
        x1 = self.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            if time_debug==True:
                t0=time.time()
            x2 = conv(x1, meshes)
            if self.bn and x2.shape[2] > 1:
                x2 = self.bn[idx + 1](x2, meshes)
            x2 = x2 + x1
            x2 = self.relu(x2)
            x1 = x2
            if time_debug==True:
                print(f"Conv ({x1.shape}): {time.time() - t0} sec")
        x2 = x2.squeeze(3)
        before_pool = x2
        unpools = None
        unpool_weights = None
        if self.pool:
            if time_debug==True:
                t0=time.time()
            before_pool = x2
            x2, unpools, unpool_weights = self.pool(x2, meshes)
            if len(unpool_weights) == 0:
                unpool_weights = None
            if len(unpools) == 0:
                unpools = None
            if time_debug==True:
                print(f"Pool ({x2.shape}): {time.time() - t0} sec")
        return x2, before_pool, unpools, unpool_weights

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 transfer_data=True, final=False, down_conv=None,
                 resconv=False, leakyrelu=False, ln=False):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.final = final 
        self.n_blocks = blocks
        self.unroll = None
        self.transfer_data = transfer_data
        # self.leakyrelu = leakyrelu
        if leakyrelu: 
            self.relu = nn.LeakyReLU()
        else: 
            self.relu = nn.ReLU()
        if resconv: 
            convop = MResConv 
        else: 
            convop = MeshConv
        self.up_conv = convop(in_channels, out_channels)
        if transfer_data:
            if down_conv is not None:
                self.conv1 = convop(out_channels + down_conv, out_channels)
            else:
                self.conv1 = convop(2 * out_channels, out_channels)
        else:
            self.conv1 = convop(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(convop(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            if ln:
                self.bn.append(LayerNormPad())
            else:
                self.bn.append(InstanceNormPad())
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, unpools = None, from_down=None, weights = None, time=False, return_pre=False):
        return self.forward(x, unpools, from_down, weights, time, return_pre)

    def forward(self, x, unpools, from_down, weights, time_debug, return_pre):
        if time_debug == True: 
            import time
            t0 = time.time()
        from_up, meshes = x
        pre_x = torch.clone(from_up)
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if time_debug==True:
            print(f"UpConv ({x1.shape}): {time.time()-t0} sec")
        if self.unroll:
            if time_debug==True:
                t0 = time.time()
            # Pass target # edges here using the skip connection data
            target_n = None
            if self.transfer_data == True and from_down is not None:
                target_n = from_down.shape[2]
            x1 = self.unroll(x1, meshes, unpools, weights, target_n)
            if time_debug==True:
                print(f"Unpool ({x1.shape}): {time.time()-t0} sec")
                t0 = time.time()
        # NOTE: there is edge case where transfer data is True but no pooling takes place 
        if self.transfer_data == True and from_down is not None:
            # Debugging: sometimes there's misalignment b/w residual connection and upconv 
            assert x1.shape[0] == from_down.shape[0], f"Incorrect residual shape: x {x1.shape}, resid {from_down.shape}"
            assert x1.shape[2] == from_down.shape[2], f"Incorrect residual shape: x {x1.shape}, resid {from_down.shape}"
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if time_debug==True:
            print(f"ResConv ({x1.shape}): {time.time()-t0} sec")
            t0 = time.time()
        if self.bn:
            x1 = self.bn[0](x1, meshes)
        
        x1 = self.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            if time_debug == True:
                t0 = time.time()
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2, meshes)
            if self.residual:
                x2 = x2 + x1
            # Don't relu the final convolution 
            if self.final == False or idx != self.n_blocks-1:
                x2 = self.relu(x2)
            x1 = x2
            if time_debug==True:
                print(f"Conv ({x1.shape}): {time.time()-t0} sec")
        x2 = x2.squeeze(3)
        if return_pre == True:
            return x2, pre_x
        return x2

class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, export_pool=False, poolorder='norm', 
                 attention_cutoff=None, n_attention_heads=0, resconv=False, leakyrelu=False, ln=False):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.export_pool = export_pool
        self.convs = []
        self.atns = [] 
        for i in range(len(convs) - 1):
            if i < len(pools):
                pool = pools[i]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool, poolorder=poolorder, resconv=resconv, leakyrelu=leakyrelu, ln=ln))
            # Weave in attention layers if set 
            if isinstance(attention_cutoff, int) and n_attention_heads > 0:
                if attention_cutoff <= i: 
                    assert convs[i+1] % n_attention_heads == 0, f"Feature dims {convs[i+1]} must be divisible by # attention heads {n_attention_heads}"
                    self.atns.append(AttentionBlock(convs[i+1], num_heads=n_attention_heads, condition_channels=convs[i+1]))
                else: 
                    self.atns.append(None)
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            # Final linear to map back to original convolution output features
            self.fcs.append(nn.Linear(last_length, convs[-1]))
            self.fcs_bn.append(nn.InstanceNorm1d(convs[-1]))
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        self.atns = nn.ModuleList(self.atns)
        reset_params(self)

    def forward(self, x, export_pool=False, time=False):
        fe, meshes = x
        encoder_outs = []
        conv_unpools = []
        conv_unpool_weights = []
        for i in range(len(self.convs)):
            conv = self.convs[i] 
            fe, before_pool, unpools, unpool_weights = conv((fe, meshes), time)
            if self.export_pool == True or export_pool==True:
                for mesh in meshes:
                    mesh.export_obj(edge_count = len(list(mesh.topology.edges.keys())))
            encoder_outs.append(before_pool)
            if unpools is not None:
                conv_unpools.append(unpools)
                conv_unpool_weights.append(unpool_weights)
            # Attention residual 
            if len(self.atns) > 0: 
                atn = self.atns[i] 
                if atn: 
                    # Extract anchor features from batched input 
                    anchor_features = get_anchor_features(fe, meshes)
                    fe = atn(fe, condition_out=anchor_features, meshes=meshes)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            # fe = fe.contiguous().view(fe.size()[0], -1) # Flattens all edge features per input mesh
            fe = fe.transpose(2,1) # Transpose edge and feature dimensions for linear
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    fe = self.fcs_bn[i](fe)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
                # print(f"Encoder linear {i} output: {fe.shape}")
                # print(f"Encoder linear output peek: {fe[0, 0, :5]}")
            fe = fe.transpose(2,1) # Transpose back for convolution
        return fe, list(reversed(encoder_outs)), list(reversed(conv_unpools)), list(reversed(conv_unpool_weights))

    def __call__(self, x, export_pool=False, time=False):
        return self.forward(x, export_pool, time)

class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, transfer_data=True, drop_relu = True, down_convs=None,
                 cond_input_dim=None, cond_output_dim=None, binary_conv=False, attention_cutoff=None, 
                 n_attention_heads = 0, resconv=False, leakyrelu=False, ln=False):
        super(MeshDecoder, self).__init__() 
        self.up_convs = []
        self.atns = [] 
        self.binary_conv = binary_conv 
        if binary_conv == True: 
            convs += [1] 
        for i in range(len(convs) - 2):
            # Weave in attention layers if set 
            if isinstance(attention_cutoff, int) and n_attention_heads > 0:
                if attention_cutoff > i: 
                    assert convs[i] % n_attention_heads == 0, f"Feature dims {convs[i]} must be divisible by # attention heads {n_attention_heads}"
                    self.atns.append(AttentionBlock(convs[i], num_heads=n_attention_heads, condition_channels=convs[i]))
                else: 
                    self.atns.append(None)
                    
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            # NOTE: Need original downconv dims b/c up conv dimensions are different!
            down_conv = None
            if down_convs is not None:
                down_conv = down_convs[-(i+2)]
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        down_conv=down_conv, transfer_data=transfer_data,
                                        resconv=resconv, leakyrelu=leakyrelu, ln=ln))
        
        # Might need one more attention block
        if isinstance(attention_cutoff, int) and n_attention_heads > 0:
            if attention_cutoff > len(convs)-2: 
                assert convs[-2] % n_attention_heads == 0, f"Feature dims {convs[-2]} must be divisible by # attention heads {n_attention_heads}"
                self.atns.append(AttentionBlock(convs[-2], num_heads=n_attention_heads, condition_channels=convs[-2]))
            else: 
                self.atns.append(None)
                    
        unroll = 0
        if len(unrolls) > len(convs)-2:
            unroll = unrolls[len(convs)-2]
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=unroll,
                                 transfer_data=False, final = drop_relu, 
                                 resconv=resconv, leakyrelu=leakyrelu, ln=ln)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.atns = nn.ModuleList(self.atns)
        self.featuretransform = None
        if cond_input_dim is not None and cond_output_dim is not None:
            self.featuretransform = ConditionTransform(cond_input_dim, cond_output_dim)
        reset_params(self)

    def forward(self, x, encoder_outs, conv_unpools, conv_unpool_weights, time=False):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            # Attention residual 
            if len(self.atns) > 0: 
                atn = self.atns[i] 
                if atn: 
                    # Extract anchor features from batched input 
                    anchor_features = get_anchor_features(fe, meshes)
                    fe = atn(fe, condition_out=anchor_features, meshes=meshes)
                    
            before_pool = None
            if encoder_outs is not None and i < len(encoder_outs):
                before_pool = encoder_outs[i]
            # NOTE: Later upconvs may not have unrolling
            unpool_weights = None
            if conv_unpool_weights is not None and i < len(conv_unpool_weights):
                unpool_weights = conv_unpool_weights[i]
            conv_unpool = None
            if i < len(conv_unpools):
                conv_unpool = conv_unpools[i]
            # print(f"Upconv {i}: {fe.shape}")
            fe = up_conv((fe, meshes), conv_unpool, before_pool, unpool_weights, time)

        # Might need one more right before final upconv 
        if len(self.atns) > len(self.up_convs): 
            atn = self.atns[-1] 
            if atn: 
                # Extract anchor features from batched input 
                anchor_features = get_anchor_features(fe, meshes)
                fe = atn(fe, condition_out=anchor_features, meshes=meshes)
                            
        conv_unpool = None
        before_pool = None
        unpool_weights = None
        if conv_unpool_weights is not None and len(conv_unpool_weights) > len(self.up_convs):
            unpool_weights = conv_unpool_weights[len(self.up_convs)]
        if len(conv_unpools) > len(self.up_convs):
            conv_unpool = conv_unpools[len(self.up_convs)]
        fe = self.final_conv((fe, meshes), conv_unpool, before_pool, unpool_weights, time, return_pre=self.binary_conv)
        # condition_fe = condition_source
        # if condition_source == "upsample":
        #     condition_fe = fe
        # if self.featuretransform is not None:
        #     # M x C x E => M x E x C 
        #     condition_fe = condition_fe.view(condition_fe.shape[0], condition_fe.shape[2], condition_fe.shape[1])
        #     condition_fe = self.featuretransform(condition_fe, time)
        #     condition_fe = condition_fe.view(condition_fe.shape[0], condition_fe.shape[2], condition_fe.shape[1])
        # fe_cond = condition_edge_features(fe, meshes, condition_fe, time)
        return fe

    def __call__(self, x, encoder_outs, conv_unpools, conv_unpool_weights = None, time=False):
        return self.forward(x, encoder_outs, conv_unpools, conv_unpool_weights, time)

class MeshDecoderGenerator(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True, drop_relu = True, down_convs=None,
                 cond_input_dim=None, cond_output_dim=None):
        super(MeshDecoderGenerator, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            # NOTE: Need original downconv dims b/c up conv dimensions are different!
            down_conv = None
            if down_convs is not None:
                down_conv = down_convs[-(i+2)]
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        down_conv=down_conv,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        # TODO: Consider KEEPING RELU 
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False, final = drop_relu)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.featuretransform = None
        if cond_input_dim is not None and cond_output_dim is not None:
            self.featuretransform = ConditionTransform(cond_input_dim, cond_output_dim)
        reset_params(self)

    def forward(self, x, encoder_outs, conv_unpools, conv_unpool_weights, pred_upsample, condition_source, time=False):
        fe, meshes = x
        if pred_upsample == True:
            condition_fe = condition_source
            if condition_source == "upsample":
                condition_fe = fe
            if self.featuretransform is not None:
                condition_fe = condition_fe.transpose(2,1) # M x C x E -> M x E x C 
                condition_fe = self.featuretransform(condition_fe, time)
                condition_fe = condition_fe.transpose(2,1) # M x E x C  -> M x C x E 
            fe_cond = condition_edge_features(fe, meshes, condition_fe, time)
            yield False, fe_cond
        # Debugging: check size of encoder outs
        # for out in encoder_outs:
        #     if out is not None:
        #         print(out.shape)
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            # NOTE: Later upconvs may not have unrolling
            unpool_weights = None
            if conv_unpool_weights is not None and i < len(conv_unpool_weights):
                unpool_weights = conv_unpool_weights[i]
            conv_unpool = None
            if i < len(conv_unpools):
                conv_unpool = conv_unpools[i]
            # print(f"Upconv {i}: {fe.shape}")
            fe = up_conv((fe, meshes), conv_unpool, before_pool, unpool_weights, time)
            if pred_upsample == True:
                condition_fe = condition_source
                if condition_source == "upsample":
                    condition_fe = fe
                if self.featuretransform is not None:
                    condition_fe = condition_fe.transpose(2,1)
                    condition_fe = self.featuretransform(condition_fe, time)
                    condition_fe = condition_fe.transpose(2,1)
                fe_cond = condition_edge_features(fe, meshes, condition_fe, time)
                yield False, fe_cond
        fe = self.final_conv((fe, meshes), time)
        condition_fe = condition_source
        if condition_source == "upsample":
            condition_fe = fe
        if self.featuretransform is not None:
            condition_fe = condition_fe.transpose(2,1)
            condition_fe = self.featuretransform(condition_fe, time)
            condition_fe = condition_fe.transpose(2,1)
        fe_cond = condition_edge_features(fe, meshes, condition_fe, time)
        yield True, fe_cond

    def __call__(self, x, encoder_outs, conv_unpools, conv_unpool_weights = None, pred_upsample=False, condition_source=None,
                 time=False):
        return self.forward(x, encoder_outs, conv_unpools, conv_unpool_weights, pred_upsample, condition_source, time)

class SelectionPrediction(nn.Module):
    def __init__(self, n_input, width=512, depth=1, clamp="sigmoid", dropout=False):
        super(SelectionPrediction, self).__init__()
        self.clamp = clamp
        layers = []
        layers.append(nn.Linear(n_input, width))
        if dropout: 
            layers.append(nn.Dropout(0.2))
        layers.append(nn.ReLU())
        for i in range(depth-1):
            layers.append(nn.Linear(width, width))
            if dropout: 
                layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, width))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))
        if clamp == "sigmoid":
            layers.append(nn.Sigmoid())
        if clamp == "tanh":
            layers.append(nn.Tanh())
        if clamp == "softmax":
            layers.append(nn.ReLU())
            layers.append(nn.Softmax(dim=1))
        self.mlp = nn.ModuleList(layers)

    def forward(self, x, time_debug=False):
        if time_debug == True:
            import time
            t0=time.time()
        for layer in self.mlp:
            x = layer(x)
        if self.clamp == "tanh":
            x = x / 2 + 0.5
        if self.clamp == "clamp":
            x = torch.clamp(x, 0, 1)
        if time_debug==True:
            print(f"Selection prediction: {time.time() - t0} sec")
        return x
    
# Baseline network: we should be able to get perfect training results by just fitting MLP on intrinsic features + geodesics + normals 
class BasicSelection(nn.Module):
    def __init__(self, n_input, fcs, clamp="sigmoid"):
        super(BasicSelection, self).__init__()
        self.clamp = clamp
        layers = []
        layers.append(nn.Linear(n_input, fcs[0]))
        layers.append(nn.ReLU())
        for i in range(1,len(fcs)):
            layers.append(nn.Linear(fcs[i-1], fcs[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(fcs[i], 1))
        if clamp == "sigmoid":
            layers.append(nn.Sigmoid())
        if clamp == "tanh":
            layers.append(nn.Tanh())
        self.mlp = nn.ModuleList(layers)

    def forward(self, x, meshes, layer=None, export_pool=None):
        # Convert M x C x E => M x E x C
        x = x.transpose(2, 1)
        for layer in self.mlp:
            x = layer(x)
        if self.clamp == "tanh":
            x = x / 2 + 0.5
        if self.clamp == "clamp":
            x = torch.clamp(x, 0, 1)
        # Convert to face predictions 
        x = edgetoface(x, meshes)
        return x

# Module to transform conditioning feature prior to concatenation
# NOTE: 1x1 convolution is SAME as fully connected layer
class ConditionTransform(nn.Module):
    def __init__(self, n_input, n_output, depth=1):
        super(ConditionTransform, self).__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_output))
        layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(n_output, n_output))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_output, n_output))
        self.mlp = nn.ModuleList(layers)

    def forward(self, x, time_debug=False):
        if time_debug == True:
            import time
            t0=time.time()
        for layer in self.mlp:
            x = layer(x)
        if time_debug==True:
            print(f"Condtional feature transformation: {time.time() - t0} sec")
        return x

# Return features concatenated with mean-pooled conditioned features
def condition_edge_features(fe, meshes, condition_source, time_debug=False, device=torch.device('cpu')):
    # Input: M x C x E
    # Output: M x (C + C_cond) x E
    if time_debug==True:
        import time
        t0 = time.time()
    if condition_source is None:
        return fe 
    # elif condition_source == "upsample":
    #     condition_source = fe
    condition_fe = torch.zeros(fe.shape[0], condition_source.shape[1], fe.shape[2], device=device)
    for i, mesh in enumerate(meshes):
        anchor_fs = mesh.anchor_fs
        # Aggregate all edge features associated with anchor vertices
        cond_edges = []
        for f in anchor_fs:
            cond_edges += list(mesh.topology.faces[f].adjacentEdges())
        cond_edges = [edge.index for edge in cond_edges]
        cond_edges, sort_edge_indices = torch.sort(torch.unique(torch.tensor(cond_edges)))
        mesh_cond = torch.mean(condition_source[i, :, cond_edges], dim=1)
        mesh_cond = torch.column_stack([mesh_cond] * condition_fe.shape[2])
        condition_fe[i] = mesh_cond
    if time_debug==True:
        print(f"Feature conditioning: {time.time()-t0} sec")
    return condition_fe

def condition_vertex_features(fe, meshes, condition_source, time_debug=False):
    # Input: M x V x C
    # Output: M x V x (C + C_cond)
    if time_debug==True:
        import time
        t0 = time.time()
    if condition_source is None:
        return fe 
    # elif condition_source == "upsample":
    #     condition_source = fe
    condition_fe = torch.zeros(fe.shape[0], fe.shape[1], condition_source.shape[2], device=fe.device)
    for i, mesh in enumerate(meshes):
        anchor_fs = mesh.anchor_fs
        # Aggregate all edge features associated with anchor vertices
        cond_v_inds = [mesh.topology.vertices[v.index].index for f in anchor_fs for v in mesh.topology.faces[f].adjacentVertices()]
        mesh_cond = torch.mean(condition_source[i, cond_v_inds, :], dim=0)
        mesh_cond = torch.row_stack([mesh_cond] * condition_fe.shape[1])
        condition_fe[i] = mesh_cond
    if time_debug==True:
        print(f"Feature conditioning: {time.time()-t0} sec")
    return condition_fe

def condition_face_features(fe, meshes, condition_source, time_debug=False):
    # Input: M x F x C
    # Output: M x F x (C + C_cond)
    if time_debug==True:
        import time
        t0 = time.time()
    if condition_source is None:
        return fe 
    condition_fe = torch.zeros(fe.shape[0], fe.shape[1], condition_source.shape[2], device=fe.device)
    for i, mesh in enumerate(meshes):
        anchor_fs = mesh.anchor_fs
        # Aggregate all features associated with anchor vertices
        cond_f_inds = [mesh.topology.faces[f].index for f in anchor_fs]
        mesh_cond = torch.mean(condition_source[i, cond_f_inds, :], dim=0)
        mesh_cond = torch.row_stack([mesh_cond] * condition_fe.shape[1])
        condition_fe[i] = mesh_cond
    if time_debug==True:
        print(f"Feature conditioning: {time.time()-t0} sec")
    return condition_fe

# Return batched anchor features 
def get_anchor_features(fe, meshes):
    # Input: M x C x E
    # Output: M x C
    condition_fe = torch.zeros(fe.shape[0], fe.shape[1], device=fe.device)
    for i, mesh in enumerate(meshes):
        anchor_fs = mesh.anchor_fs
        # Aggregate all edge features associated with anchor vertices
        cond_edges = []
        for f in anchor_fs:
            cond_edges += list(mesh.topology.faces[f].adjacentEdges())
        cond_edges = [edge.index for edge in cond_edges]
        cond_edges, sort_edge_indices = torch.sort(torch.unique(torch.tensor(cond_edges)).long())
        edge_inds = list(sorted(mesh.topology.edges.keys()))
        edge_topo = torch.zeros(np.max(edge_inds)+1).long()
        edge_topo[edge_inds] = torch.arange(len(edge_inds))  
        cond_edges = edge_topo[cond_edges]
        mesh_cond = torch.mean(fe[i, :, cond_edges], dim=1)
        condition_fe[i] = mesh_cond
    return condition_fe

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        # NOTE: row-wise sorting (per-channel) 
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)  # for sape

    def forward(self, x):
        # Expected input: M x C x E
        # B shape: C x C'

        channels = x.shape[1]

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        res = x.transpose(2, 1) @ self._B.to(x.device).unsqueeze(0)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        res = 2 * np.pi * res.transpose(2,1)
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)
import torch
from . import networks
import os
from os.path import join
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib as mpl
from matplotlib import cm
from models.layers.meshing import Mesh
from models.layers.meshing.analysis import computeFaceNormals, computeFaceAreas
from models.networks import floodfill_scalar_v2
from data.intseg_data import condition_and_cache, augment
from util.util import print_network, export_views, get_local_tris, pad, ZeroNanGrad
from util.util import dclamp, cut_to_disk, cut_to_disk_single, cut_vertex, ss_torch
from util.parameterization import weightedlscm
from util.losses import arap, count_loss, gcsmoothness
torch.autograd.set_detect_anomaly(False)

class DAWand:
    """ Conditional mesh segmentation """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.machine = "gpu" if self.gpu_ids else "cpu"
        self.save_dir = join(opt.export_save_path, opt.name)
        self.network_load_dir = opt.network_load_path

        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.optimizer = None
        self.edge_features = None
        self.mesh = None
        self.input_nc = opt.input_nc
        
        # zero grad 
        self.zerograd = ZeroNanGrad()
        
        # load/define networks
        self.net = networks.define_classifier(opt, self.input_nc, opt.ncf, self.gpu_ids, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        
        self.loss_fcn = torch.nn.BCELoss()
        if self.opt.loss == "ce":
            self.loss_fcn = torch.nn.CrossEntropyLoss()
        
        self.distortion_fcn = arap
        self.uv_history = defaultdict(lambda: defaultdict(dict))
        self.distortion_loss_fcn = None 
        if self.opt.distortion_loss == "count":
            self.distortion_loss_fcn = count_loss 

        # TRAINING OPTIONS
        if self.is_train == True:
            self.savetraindata = opt.savetraindata
            if opt.continue_train == True: 
                # NOTE: Continue train != load pretrain (save dir vs network load path)
                self.load_network(opt.which_epoch, self.save_dir)
            elif opt.load_pretrain == True: 
                self.load_network(opt.which_epoch, self.network_load_dir)
                
            self.net.train()
            # Set gradient clipping 
            if self.opt.max_grad is not None: 
                for p in self.net.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -self.opt.max_grad, self.opt.max_grad))
                    
            if opt.finetune_mlp:
                # Finetune the pretrained MLP but freeze rest of network 
                tmp_net = self.net 
                if isinstance(self.net, torch.nn.DataParallel):
                    tmp_net = self.net.module 
                for p in tmp_net.parameters():
                    p.requires_grad = False 
                for p in tmp_net.classification.parameters():
                    p.requires_grad = True 
                if opt.vectoradam: 
                    from util.vectoradam import VectorAdam
                    self.optimizer = VectorAdam([{'params': tmp_net.classification.parameters(), 'axis':-1}], lr=opt.lr, betas=(opt.beta1, 0.999))
                else:
                    self.optimizer = torch.optim.Adam(tmp_net.classification.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                
            if opt.continue_train == True:
                self.load_optimizer(opt.which_epoch)

            self.scheduler = None 
            if opt.lr_policy:
                self.scheduler = networks.get_scheduler(self.optimizer, opt)
                
            if self.savetraindata == True:
                self.losses = defaultdict(list)
                self.count_losses = defaultdict(list)
                self.distortions = defaultdict(list)
                self.uvs = defaultdict(list)
            print_network(self.net)
        else:
            try:
                self.load_network(opt.which_epoch, self.save_dir)
            except Exception as e: 
                self.load_network(opt.which_epoch, self.network_load_dir)
            self.net.eval()

    def set_input(self, data):
        # NOTE: We assume the input data is already padded 
        input_edge_features = np.stack(data['edge_features']) # B x F x E 
        input_edge_features = torch.from_numpy((input_edge_features - data['mean'][0][None, :, None])/data['std'][0][None, :, None])
        input_edge_features = input_edge_features.float().to(self.device)
        skip_inputs = [] 
        
        self.anchor_fs = data['anchor_fs']
        self.anchor_fs_labels = None 
            
        self.files = data['file']
        # NOTE: We build mesh here to avoid pickling issues 
        meshdatas = data['meshdata']
        anchordatas = data['anchordata']
        augs = data['aug'] 
        self.mesh = [] 
        # Apply augs if necessary 
        if self.opt.time:
            import time 
            t0 = time.time() 

        # NOTE: By default augs will be None 
        # NOTE: Have to build mesh structure here because it is unpicklable 
        for i in range(len(augs)):
            # NOTE: Building from serialization means arbitrary edge -> HE mapping
            mesh = Mesh(meshdata=meshdatas[i], meshname=os.path.splitext(self.files[i])[0])
            mesh.no = data['no'][i] 
            mesh.export_dir = data['export_dir'][i]
            mesh.anchor_fs = data['anchor_fs'][i]
            
            # Anchor features 
            for key, val in anchordatas[i].items(): 
                setattr(mesh, key, val)
            
            # Debugging
            # old_vs, old_fs, _ = mesh.export_soup()
            
            # NOTE: If testing, then augmentations are precomputed offline 
            if augs[i] is not None and self.opt.is_train == True: 
                augment(mesh, self.opt)
                
                # If overwriting edge features, then also need to overwrite preconditioning 
                mesh.computeEdgeFeatures(overwrite=True, intrinsics=self.opt.edgefeatures)
                
                if self.opt.extrinsic_features is not None and self.opt.extrinsic_condition_placement == "pre":
                    condition_and_cache(mesh, self.opt, evals=data['evals'][i], evecs=data['evecs'][i])
                
                new_edge_features = (mesh.edgefeatures - data['mean'][i][:, None])/data['std'][i][:, None]
                input_edge_features[i] = torch.from_numpy(pad(new_edge_features, self.opt.ninput_edges))
                
            # Sometimes augmentations cause heat geodesic to compute NAs
            # Skip data in these cases 
            if not torch.all(torch.isfinite(input_edge_features[i])):
                print(f"Warning: non-finite inputs found for mesh {mesh.no}. Skipping...")
                skip_inputs.append(i)
                continue 

            self.mesh.append(mesh)
        
        # Skip bad inputs 
        input_edge_features = input_edge_features[list(set(range(len(input_edge_features))).difference(set(skip_inputs)))]
        
        if self.opt.time:
            print(f"set_input: {time.time() - t0:0.5f}") 
        
        # Edge case: all inputs skipped b/c non-finite 
        if len(self.mesh) == 0:
            print(f"Warning: batch skipped because no finite inputs")
            return False 
        
        # SET EDGE FEATURES 
        self.edge_features = input_edge_features

        # Labels
        self.labels = [torch.from_numpy(label).long().to(self.device) if label is not None else None for label in data['label']]
            
        # Pad labels
        max_len = max([len(label) if label is not None else 0 for label in self.labels])
        padded_labels = []
        padded_faces = [] 
        for label in self.labels:
            if label is not None: 
                padded_labels.append(torch.nn.functional.pad(label, (0, max_len - len(label))))
            else:
                padded_labels.append(None)
        self.labels = padded_labels
        # Convert to class probability labels if necessary 
        # if len(torch.unique(self.labels)) > 2: 
        #     prob_labels = torch.zeros(self.labels.shape[0], self.labels.shape[1], 2)
        #     prob_labels[:,:,1] = self.labels[i] 
        #     prob_labels[:,:,0] = 1 - prob_labels[:,:,1]
        #     self.labels = prob_labels             
            
        # Re-weight loss function based on size of ground truth
        # weights = torch.ones(self.labels.shape)
        # for i in range(len(self.labels)):
        #     label = self.labels[i]
        #     weights[i][label == 1] = torch.sum(label == 0).item() / torch.sum(label == 1).item()
        if self.labels is not None:
            good_labels = [label for label in self.labels if label is not None]
            if len(good_labels) > 0:
                tot_labels = torch.vstack(good_labels)
                weights = torch.ones(len(torch.unique(tot_labels))).float().to(self.device)
                if self.opt.loss == "ce":
                    if self.opt.reweight_loss == True:
                        weights = torch.tensor([1., torch.sum(tot_labels == 0).item() / torch.sum(tot_labels == 1).item()]).to(self.device)
                    self.loss_fcn = torch.nn.CrossEntropyLoss(weight=weights)

        # Debugging: make sure anchor, ground truth, and geodesics correspond correctly 
        # import polyscope as ps
        # i = 0
        # mesh = self.mesh[i] 
        # vertices, faces, _ = mesh.export_soup()
        # anchor_fs = mesh.anchor_fs
        # ps.init()
        # ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=1)
        # gt = self.labels[i][:len(faces)].detach().numpy()
        # gt[anchor_fs] = 2
        # ps_mesh.add_scalar_quantity("gt", gt, defined_on="faces", enabled=True)
        # anchor_pos = np.mean([mesh.vertices[v.index] for find in anchor_fs for v in mesh.topology.faces[find].adjacentVertices()], axis=0, keepdims=True)
        # ps_curve = ps.register_curve_network("anchor", anchor_pos, np.array([[0,0]]))
        # ps.show()
        # raise
        
        return True 

    def forward(self, layer=None, export_pool=False):        
        out, deep_features = self.net(self.edge_features, self.mesh, layer, export_pool=export_pool)
        
        if self.opt.time == True and torch.cuda.is_available():
            import time
            # Get GPU memory usage 
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            m = torch.cuda.max_memory_allocated(0)
            f = r-a  # free inside reserved
            print(f"{a/1024**3:0.3f} GB allocated. \nGPU max memory alloc: {m/1024**3:0.3f} GB. \nGPU total memory: {t/1024**3:0.3f} GB.\n")
            t0 = time.time()
            
        return out, deep_features 

    def optimize_parameters(self, epoch=0, evalmode=False):
        # TODO: Not sure if this helps yet 
        torch.cuda.empty_cache()
        
        if not evalmode:
            self.optimizer.zero_grad(set_to_none=True)

        # Check if switch to distortion-only supervision 
        distortion_mode = False 
        if epoch >= self.opt.delayed_distortion_epochs and self.opt.solo_distortion==True:
            distortion_mode = True 
                
        out, deep_features = self.forward()
        
        self.total_loss = 0.0
        # M x F x 1 OR M x E x 1
        assert len(self.mesh) == out.shape[0]
        assert torch.all(torch.isfinite(out)), f"{out.flatten()[~torch.isfinite(out.flatten())][:2]}"

        loss_dict = {}
        for i in range(len(self.mesh)):
            if self.opt.time == True:
                import time
                t0 = time.time()
            meshdict = {}
            mesh = self.mesh[i]
            vertices, faces, _ = mesh.export_soup()
            mesh_out = out[i][:len(faces)]
            # NOTE: padded faces should always have ZERO loss
            # (1) Supervision loss 
            # Convert 1-D binary probs to 2D for cross entropy loss
            if self.opt.supervised == True and distortion_mode == False and self.labels is not None and self.labels[i] is not None: 
                if self.opt.time == True:
                    import time
                    t0 = time.time()
                    
                if self.opt.loss == "ce":
                    if out.shape[2] == 1:
                        class_preds = torch.cat([1 - mesh_out, mesh_out], dim=1).float()  # F x 2
                    else:
                        class_preds = mesh_out 
                else:
                    class_preds = mesh_out.squeeze()

                tmp_sup_loss = self.loss_fcn(class_preds, self.labels[i][:len(faces)])
                self.total_loss += tmp_sup_loss
                meshdict["supervision_loss"] = tmp_sup_loss.detach().item()
                
                if self.opt.time:
                    print(f"Mesh {i}; Labels L2 loss: {time.time() - t0:0.5f} sec")
                    
                # If running mixed training, then L2 supervision loss is solo 
                if self.opt.mixedtraining:
                    loss_dict[mesh.no] = meshdict
                    continue 
                            
            # (2) Graphcuts L2 loss 
            if self.opt.gcsupervision:
                if self.opt.time == True:
                    import time
                    t0 = time.time()
                from util.util import graphcuts 
                
                gcpreds = graphcuts(mesh_out.detach().cpu().numpy().squeeze(), mesh, anchors=mesh.anchor_fs[0])
                gcpreds = torch.from_numpy(gcpreds).long().to(self.device)
            
                if self.opt.loss == "ce":
                    if out.shape[2] == 1:
                        class_preds = torch.cat([1 - mesh_out, mesh_out], dim=1).float()  # F x 2
                    else:
                        class_preds = mesh_out 
                else:
                    class_preds = mesh_out.squeeze()
                    
                gc_loss = self.opt.gcsupervision_weight * self.loss_fcn(class_preds, gcpreds)
                self.total_loss += gc_loss
                meshdict["gc_loss"] = gc_loss.detach().item()  
                
                if self.opt.time:
                    print(f"Mesh {i}; GC L2 loss: {time.time() - t0:0.5f} sec")
            
            # Render views of pooled segmentation (soft and hard), and ground truth
            if self.opt.export_view_freq > 0 and epoch % self.opt.export_view_freq == 0:
                if self.opt.time == True:
                    import time
                    t0 = time.time()                
                epoch_dir = os.path.join(self.save_dir, f"epoch{epoch}", self.opt.phase)
                Path(epoch_dir).mkdir(parents=True, exist_ok=True)                
                tmp_pred = mesh_out.detach().cpu() 

                if self.opt.export_preds:
                    torch.save(tmp_pred.detach().cpu(), os.path.join(epoch_dir, f"{mesh.no}.pt"))
                    
                    # Also save GC predictions if set 
                    if self.opt.gcsupervision:
                        torch.save(gcpreds.detach().cpu(), os.path.join(epoch_dir, f"{mesh.no}_gc.pt"))
        
                if self.opt.plot_preds:
                    import matplotlib.pyplot as plt 
                    hard_preds = np.round(tmp_pred.cpu().detach().numpy())
                    cmap = plt.get_cmap("coolwarm")
                    max_label = max(mesh_out.shape[1]-1, 1)
                    norm = mpl.colors.Normalize(vmin=0, vmax=max_label) # 0 -> C class labels 
                    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)
                    default_color = np.array(scalarmap.to_rgba(0)[:3])
                    soft_fcolors = scalarmap.to_rgba(tmp_pred.squeeze().numpy())[:,:3]
                    # Convert face colors to vertex colors per face 
                    # NOTE: expand vertex colors to total no. faces (repeat 3 times for each face)
                    soft_vcolors = [color for color in soft_fcolors for _ in range(3)]
                    soft_vcolors = np.vstack(soft_vcolors)
                    export_views(mesh, epoch_dir, filename=f"{mesh.no}_softseg.png",
                                plotname=f"{mesh.no}: Soft Labels",
                                vcolors=soft_vcolors, device="cpu", n_sample=24, width=200, height=200)
                    

                    hard_fcolors = np.vstack([default_color] * len(mesh.topology.faces.keys()))
                    for label in range(1,max_label+1):
                        hard_fcolors[np.where(hard_preds == label)[0], :] = np.array(scalarmap.to_rgba(label)[:3])
                    hard_vcolors = [color for color in hard_fcolors for _ in range(3)]
                    hard_vcolors = np.vstack(hard_vcolors)
                    export_views(mesh, epoch_dir, filename=f"{mesh.no}_hardseg.png",
                                plotname=f"{mesh.no}: Hard Labels",
                                vcolors=hard_vcolors, device="cpu")
                    
                    if self.labels is not None and self.labels[i] is not None:
                        ground_truth = np.vstack(
                            [default_color] * len(mesh.topology.faces.keys()))  # Segmentation will be gold/grey
                        gt = self.labels[i].detach().cpu().numpy().astype(int)
                        for label in range(1,max_label+1):
                            ground_truth[np.where(gt == label)[0], :] = np.array(scalarmap.to_rgba(label)[:3])
                        ground_truth_vcolors = [color for color in ground_truth for _ in range(3)]
                        ground_truth_vcolors = np.vstack(ground_truth_vcolors)
                        export_views(mesh, epoch_dir, filename=f"{mesh.no}_gt.png",
                                    plotname=f"{mesh.no}: Ground Truth",
                                    vcolors=ground_truth_vcolors, device="cpu", anchor_fs=mesh.anchor_fs)
                        if self.opt.time == True:
                            print(f"Mesh {i}; Fresnel render: {time.time() - t0:0.5f} sec")
                                
            # (3) Distortion supervision 
            if epoch >= self.opt.delayed_distortion_epochs:
                if self.opt.time == True:
                    import time
                    t0 = time.time()
                vertices = torch.from_numpy(vertices).float().to(self.device)
                faces = torch.from_numpy(faces.astype(int)).to(self.device)
                
                ffscores = torch.clone(mesh_out.squeeze())
                if self.opt.floodfillparam: 
                    ffscores = floodfill_scalar_v2(mesh, ffscores, mesh.anchor_fs[0], debug=False) 
                    
                    if self.opt.export_view_freq > 0 and epoch % self.opt.export_view_freq == 0 and self.opt.export_preds:
                        torch.save(ffscores.detach().cpu(), os.path.join(epoch_dir, f"{mesh.no}_ffpreds.pt"))

                    if self.opt.time == True: 
                        print(f"Mesh {i}; score floodfill {time.time() - t0:0.5f} sec")
                        
                ###### Step 1 Parameterization ##### 
                old_inds = [] 
                select_faces = [] 
                if self.opt.step1paramloss:
                    select_faces = torch.where(torch.round(ffscores).detach().long() == 1)[0]                    
                    distortion_energy_s1 = None 
                    # Edge case: no faces selected (skip straight to step 2)
                    if len(select_faces) > 0:
                        # Soft boundary option 
                        if self.opt.segboundary == "neighbor":
                            select_set = set(select_faces.detach().cpu().tolist())
                            for _ in range(self.opt.segradius):
                                neighbors = [n.index for f in select_set for n in mesh.topology.faces[f].adjacentFaces()]
                                select_set.update(neighbors)
                            select_set = np.array(list(select_set))
                            select_faces = torch.from_numpy(select_set).long().to(self.device) 
                            
                        # === Cut boundaries === 
                        # Map selection back to old vertex indices 
                        old_inds, indices = torch.sort(torch.unique(faces[select_faces]))
                        
                        sub_vertices, sub_faces = mesh.export_submesh(select_faces.detach().cpu().numpy())
                        submesh = Mesh(sub_vertices, sub_faces)
                        
                        # Don't cut if nonmanifold edges
                        if submesh.topology.hasNonManifoldEdges():
                            pass
                        else:
                            if self.opt.cut_param:
                                try:
                                    # Check for nonmanifold vertices
                                    if submesh.topology.hasNonManifoldVertices():
                                        print(f"Cutting nonmanifold vertices: {submesh.topology.nonmanifvs}")
                                        for vind in submesh.topology.nonmanifvs:
                                            cut_vertex(submesh, vind)
                                    
                                    # NOTE: don't really need this unless perfectly cylinder topology       
                                    if len(submesh.topology.boundaries) > 1:
                                        cut_to_disk(submesh)
                                        
                                    # Check for nonmanifold vertices while only one boundary 
                                    if submesh.topology.hasNonManifoldVertices():
                                        print(f"Cutting nonmanifold vertices: {submesh.topology.nonmanifvs}")
                                        for vind in mesh.topology.nonmanifvs:
                                            cut_vertex(submesh, vind)
                                    
                                    if not hasattr(submesh, "vertexangle"):
                                        from models.layers.meshing.analysis import computeVertexAngle
                                        computeVertexAngle(submesh)
                                    
                                    # If cone/sock-like structures then also cut those 
                                    singlevs = np.where(2 * np.pi - submesh.vertexangle >= np.pi/4)[0]
                                    if len(singlevs) >= 0 and len(submesh.topology.boundaries) > 0:
                                        cut_to_disk_single(submesh, singlevs)
                                    
                                    # Check for nonmanifold vertices while only one boundary 
                                    if submesh.topology.hasNonManifoldVertices():
                                        print(f"Cutting nonmanifold vertices: {submesh.topology.nonmanifvs}")
                                        for vind in submesh.topology.nonmanifvs:
                                            cut_vertex(submesh, vind)
                                except Exception as e: 
                                    print(e)

                        sub_vertices, sub_faces, _ = submesh.export_soup()
                        sub_vertices = torch.from_numpy(sub_vertices).to(self.device)
                        sub_faces = torch.from_numpy(sub_faces).to(self.device)

                        if self.opt.cut_param:
                            # NOTE: IMPORTANT -------
                            # Map cut vs, fs back to orginal mesh  
                            cutf_to_f = torch.zeros(len(sub_vertices)).to(self.device)
                            cutf_to_f[len(old_inds):] = torch.arange(len(vertices), len(vertices) + len(sub_vertices) - len(old_inds))
                            cutf_to_f[:len(old_inds)] = old_inds
                            new_subfs = cutf_to_f[sub_faces] 
                            faces[select_faces] = new_subfs.long()
                            # New cut vertices 
                            vertices = torch.cat([vertices, sub_vertices[len(old_inds):]])
                        
                        # Set softweights to be minimum 0.5 (otherwise get crazy gradients) 
                        s1_weights = dclamp(ffscores[select_faces], 0.5, 1)
                        
                        lscm_err = self.opt.lscmreg 
                        uv_res_s1 = weightedlscm(sub_vertices, sub_faces, device=self.device,
                                                        return_face_err=lscm_err,
                                                        face_weights = ZeroNanGrad.apply(s1_weights),
                                                        fixzero=True, verbose=False, timeit=self.opt.time)
                        
                        fareas = torch.from_numpy(mesh.fareas).to(self.device)[select_faces]
                        if self.opt.distortion_metric == "arap":
                            # Compute distortion energy
                            local_tris_s1 = get_local_tris(sub_vertices, sub_faces, device=self.device)
                            distortion_energy_s1 = self.distortion_fcn(local_tris_s1.float(), sub_faces, uv_res_s1.float(),
                                            device=self.device, renormalize=True,
                                            return_face_energy=True, timeit=self.opt.time)
                            if distortion_energy_s1 is not None and torch.all(torch.isfinite(distortion_energy_s1)):
                                meshdict['s1_arap_distortion'] = torch.mean(distortion_energy_s1).detach().item()
                                if self.opt.step1paramloss:
                                    if self.distortion_loss_fcn is not None:
                                        s1_countloss = self.distortion_loss_fcn(distortion_energy_s1, fareas, return_softloss=False, device=self.device, 
                                                                            threshold = self.opt.arapthreshold)
                                        meshdict['s1_distortion_loss'] = s1_countloss.detach().item()
                                        self.total_loss += s1_countloss
                                    else:
                                        s1_countloss = torch.mean(distortion_energy_s1)
                                self.total_loss += s1_countloss        
                        elif self.opt.distortion_metric is not None:
                            raise ValueError(f"No metric found for {self.opt.distortion_metric}")
                        
                        # This is a bottleneck now: save predictions instead 
                        if self.opt.export_view_freq > 0 and epoch % self.opt.export_view_freq == 0:
                            epoch_param_dir = os.path.join(self.save_dir, f"epoch{epoch}", self.opt.phase, "uv")
                            Path(epoch_param_dir).mkdir(parents=True, exist_ok=True)
                            if self.opt.export_preds:
                                # Map cut UVs back to original 
                                og_uv = uv_res_s1.detach().cpu().numpy()
                                np.save(os.path.join(epoch_param_dir, f"{mesh.no}_param1.npy"), og_uv)
                            
                            if self.opt.plot_preds:                                
                                # NOTE: We set upper bound on distortion color arbitrarily here
                                norm = mpl.colors.Normalize(vmin=0, vmax=2)
                                scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)

                                if self.opt.time == True:
                                    ttmp = time.time()
                                
                                select_faces_np = select_faces.detach().cpu().numpy()

                                s1_vertices = np.concatenate([uv_res_s1.detach().cpu().numpy(), np.zeros((len(uv_res_s1), 1))], axis=1)
                                uv_mesh = Mesh(s1_vertices, sub_faces.detach().cpu().numpy())
                                uv_mesh.normalize()
                                computeFaceNormals(uv_mesh)
                                # Remap anchors 
                                sub_anchors = [] 
                                for anchor in mesh.anchor_fs:
                                    if anchor in select_faces_np:
                                        sub_anchors.append(np.where(select_faces_np == anchor)[0])
                                if len(sub_anchors) == 0:
                                    sub_anchors = None 
                                else:
                                    sub_anchors = np.array(sub_anchors)
                                
                                ground_truth = np.vstack([default_color] * len(select_faces_np))
                                if self.labels is not None and self.labels[i] is not None:
                                    gt = self.labels[i].detach().cpu().numpy().astype(int)
                                    sub_gt = gt[select_faces_np]
                                    for label in range(1,max_label+1):
                                        ground_truth[np.where(sub_gt == label)[0], :] = np.array(scalarmap.to_rgba(label)[:3])
                                else:
                                    # If no labels, then color with face distortion 
                                    ground_truth = np.array(scalarmap.to_rgba(distortion_energy_s1.detach().cpu().numpy())[:,:3])
                                    
                                ground_truth_vcolors = [color for color in ground_truth for _ in range(3)]
                                ground_truth_vcolors = np.vstack(ground_truth_vcolors)
                                
                                export_views(uv_mesh, epoch_param_dir, filename=f"{mesh.no}_param1.png",
                                            plotname=f"{mesh.no}: Step 1 Param.",
                                            vcolors=ground_truth_vcolors, device="cpu", anchor_fs=sub_anchors) 
                            
                                if self.opt.time == True:
                                    print(f"Mesh {i}; Fresnel render: {time.time() - ttmp:0.5f} sec")
                                                        
                ###### Step 2 Parameterization ######
                # Edge case: all VERTICES selected in first step
                if self.opt.step2paramloss:
                    distortion_energy_s2 = None 
                    if len(old_inds) < len(mesh.vertices):
                        # (2) param. whole shape with uv results from prior step as pinned vertices
                        # NOTE: We map segmented cut back to full mesh even without step1 loss 
                        if self.opt.cut_param and not self.opt.step1paramloss:
                            select_faces = torch.where(torch.round(ffscores).detach().long() == 1)[0]
                            old_inds, indices = torch.sort(torch.unique(faces[select_faces]))
                            subvs, subfs = mesh.export_submesh(select_faces.detach().cpu().numpy())
                            submesh = Mesh(subvs, subfs)
                            try:
                                # Check for nonmanifold vertices
                                if submesh.topology.hasNonManifoldVertices():
                                    print(f"Cutting nonmanifold vertices: {submesh.topology.nonmanifvs}")
                                    for vind in submesh.topology.nonmanifvs:
                                        cut_vertex(submesh, vind)
                                
                                # NOTE: don't really need this unless perfectly cylinder topology       
                                if len(submesh.topology.boundaries) > 1:
                                    cut_to_disk(submesh)
                                    
                                # Check for nonmanifold vertices while only one boundary 
                                if submesh.topology.hasNonManifoldVertices():
                                    print(f"Cutting nonmanifold vertices: {submesh.topology.nonmanifvs}")
                                    for vind in mesh.topology.nonmanifvs:
                                        cut_vertex(submesh, vind)
                                
                                if not hasattr(submesh, "vertexangle"):
                                    from models.layers.meshing.analysis import computeVertexAngle
                                    computeVertexAngle(submesh)
                                
                                # If cone/sock-like structures then also cut those 
                                singlevs = np.where(2 * np.pi - submesh.vertexangle >= np.pi/4)[0]
                                if len(singlevs) >= 0 and len(submesh.topology.boundaries) > 0:
                                    cut_to_disk_single(submesh, singlevs)
                                
                                # Check for nonmanifold vertices while only one boundary 
                                if submesh.topology.hasNonManifoldVertices():
                                    print(f"Cutting nonmanifold vertices: {submesh.topology.nonmanifvs}")
                                    for vind in submesh.topology.nonmanifvs:
                                        cut_vertex(submesh, vind)
                            except Exception as e: 
                                print(e)
                            
                            sub_vertices, sub_faces, _ = submesh.export_soup()
                            sub_vertices = torch.from_numpy(sub_vertices).to(self.device)
                            sub_faces = torch.from_numpy(sub_faces).to(self.device)

                        if self.opt.cut_param and not self.opt.step1paramloss:
                            # NOTE: IMPORTANT -------
                            # Map cut vs, fs back to orginal mesh  
                            cutf_to_f = torch.zeros(len(sub_vertices)).to(self.device)
                            cutf_to_f[len(old_inds):] = torch.arange(len(vertices), len(vertices) + len(sub_vertices) - len(old_inds))
                            cutf_to_f[:len(old_inds)] = old_inds
                            new_subfs = cutf_to_f[sub_faces] 
                            faces[select_faces] = new_subfs.long()
                            # New cut vertices 
                            vertices = torch.cat([vertices, sub_vertices[len(old_inds):]])
                            
                        # NOTE: This is same as global softboundary 
                        if self.opt.softs2:
                            face_weights_s2 = dclamp(ffscores, min=0.5, max=1.0)
                        else:
                            face_weights_s2 = torch.clone(ffscores)
                        
                        pinned_vertices = None
                        pinned_vertex_vals = None
                        if len(select_faces) > 0 and self.opt.step1paramloss:
                            # NOTE: We reshape to (x1, x2, ...., y1, y2)
                            # NOTE: The cutting in step 1 should automatically map the inds in `faces` to the post-cut!!
                            pinned_vertices, _ = torch.sort(torch.unique(faces[select_faces]))
                            pinned_vertex_vals = torch.cat([uv_res_s1[:,0], uv_res_s1[:,1]])
                            assert 2 * len(pinned_vertices) == len(pinned_vertex_vals)

                        lscm_err = self.opt.lscmreg
                        uv_res_s2 = weightedlscm(vertices, faces, device=self.device,
                                                        return_face_err=lscm_err,
                                                        pinned_vertices=pinned_vertices,
                                                        pinned_vertex_vals=pinned_vertex_vals,
                                                        face_weights=ZeroNanGrad.apply(face_weights_s2),
                                                        fixzero=True, verbose=False, timeit=self.opt.time)
                        
                        fareas = torch.from_numpy(mesh.fareas).to(self.device)
                        if self.opt.distortion_metric == "arap":
                            # Compute distortion energy
                            local_tris_s2 = get_local_tris(vertices, faces, device=self.device)
                            arap_distortion_s2 = self.distortion_fcn(local_tris_s2.float(), faces, uv_res_s2.float(),
                                            device=self.device, renormalize=True, return_face_energy=True,
                                            timeit=self.opt.time)
                            if arap_distortion_s2 is not None and torch.all(torch.isfinite(arap_distortion_s2)): 
                                meshdict['s2_arap_distortion'] = torch.mean(arap_distortion_s2).detach().item()
                                
                                if self.opt.step2paramloss:
                                    if self.distortion_loss_fcn is not None:
                                        countloss = self.distortion_loss_fcn(arap_distortion_s2, fareas, return_softloss=False, device=self.device,
                                                                            threshold = self.opt.arapthreshold)
                                        meshdict['s2_distortion_loss'] = countloss.detach().item()
                                        self.total_loss += countloss
                                    else:
                                        self.total_loss += torch.mean(arap_distortion_s2)
                        elif self.opt.distortion_metric is not None:
                            raise ValueError("Need a distortion metric for parameterization distortion!")
                                                                        
                        if self.opt.export_view_freq > 0 and epoch % self.opt.export_view_freq == 0:
                            epoch_param_dir = os.path.join(self.save_dir, f"epoch{epoch}", self.opt.phase, "uv")
                            Path(epoch_param_dir).mkdir(parents=True, exist_ok=True)
                            if self.opt.export_preds:
                                # Save parameterizations from each epoch 
                                np.save(os.path.join(epoch_param_dir, f"{mesh.no}_param2.npy"), uv_res_s2.detach().cpu().numpy())
                            
                            if self.opt.plot_preds: 
                                # NOTE: We set upper bound on distortion color arbitrarily here
                                norm = mpl.colors.Normalize(vmin=0, vmax=2)
                                scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)
                                
                                if self.opt.time == True:
                                    ttmp = time.time()
                                ground_truth = np.vstack([default_color] * len(mesh.topology.faces.keys()))
                                
                                if self.labels is not None and self.labels[i] is not None:
                                    gt = self.labels[i].detach().cpu().numpy().astype(int)
                                    for label in range(1,max_label+1):
                                        ground_truth[np.where(sub_gt == label)[0], :] = np.array(scalarmap.to_rgba(label)[:3])
                                else:
                                    # If no labels, then color with face distortion 
                                    ground_truth = np.array(scalarmap.to_rgba(distortion_energy_s2.detach().cpu().numpy())[:,:3])
                                    
                                ground_truth_vcolors = [color for color in ground_truth for _ in range(3)]
                                ground_truth_vcolors = np.vstack(ground_truth_vcolors)
                                
                                s2_vertices = np.concatenate([uv_res_s2.detach().cpu().numpy(), np.zeros((len(uv_res_s2), 1))], axis=1)
                                vs, fs, _ = mesh.export_soup() 
                                uv_mesh = Mesh(s2_vertices, fs)
                                uv_mesh.normalize()
                                computeFaceNormals(uv_mesh)
                                export_views(uv_mesh, epoch_param_dir, filename=f"{mesh.no}_param2.png",
                                            plotname=f"{mesh.no}: Step 2 Param.",
                                            vcolors=ground_truth_vcolors, device="cpu", anchor_fs=mesh.anchor_fs) 

                                if self.opt.time == True:
                                    print(f"Mesh {i}; Fresnel render: {time.time() - ttmp:0.5f} sec")
                            
                    if self.opt.time == True:
                        print(f"Mesh {i}; Total distortion loss time: {time.time() - t0:0.5f} sec")
                
            # (4) Anchor loss: encourage predicted anchor vertex weights to be 1
            if self.opt.anchor_loss == True:  
                anchor_loss = torch.mean((mesh_out[mesh.anchor_fs] - 1)**2)
                meshdict['anchor_loss'] = anchor_loss.detach().item()
                self.total_loss += anchor_loss 
                       
            # (5) Graphcuts smoothness loss
            if self.opt.gcsmoothness:
                if self.opt.time:
                    t0 = time.time() 
                
                smoothness_loss = self.opt.gcsmoothness_weight * gcsmoothness(mesh_out, mesh)
                meshdict['gcsmoothness'] = smoothness_loss.detach().item()
                self.total_loss += smoothness_loss 
                
                if self.opt.time:
                    print(f"Mesh {i}; GC smoothness loss: {time.time() - t0:0.5f} sec")
                
            loss_dict[mesh.no] = meshdict

        if self.opt.time == True and torch.cuda.is_available():
            import time
            # Get GPU memory usage 
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            m = torch.cuda.max_memory_allocated(0)
            f = r-a  # free inside reserved
            print(f"{a/1024**3:0.3f} GB allocated. \nGPU max memory alloc: {m/1024**3:0.3f} GB. \nGPU total memory: {t/1024**3:0.3f} GB.")
            
            # Get CPU RAM usage too 
            import psutil 
            print(f'RAM memory % used: {psutil.virtual_memory()[2]}')
            t0 = time.time()

        if evalmode==False:
            if not self.total_loss.requires_grad:
                print(f"No loss computed for epoch {epoch}.")
                return self.total_loss, loss_dict
            
            self.total_loss.backward()
            
            if self.opt.clip_grad: 
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.opt.clip_grad)
                
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.opt.time == True:
                print(f"Optimizer step: {time.time() - t0:0.5f} sec")
                
            return self.total_loss.detach().item(), loss_dict
        else:
            if isinstance(self.total_loss, float):            
                return self.total_loss, loss_dict, out.detach()
            
            return self.total_loss.detach().item(), loss_dict, out.detach()

    def test(self, epoch=0):
        """tests model
        returns: number correct and total number
        """
        self.net.eval()
        with torch.no_grad():
            loss, loss_dict, out = self.optimize_parameters(epoch, evalmode=True)
        np_labels = [label.long().detach().cpu().numpy() if label is not None else None for label in self.labels]
        
        return np_labels, out.detach().cpu().numpy(), loss, loss_dict

    ##################
    def load_optimizer(self, which_epoch):
        optim_filename = '%s_optim.pth' % which_epoch
        # NOTE: Network load dir will ONLY be used for finetuning 
        if os.path.exists(join(self.save_dir, optim_filename)):
            try:
                load_path = join(self.save_dir, optim_filename)
                optim_state = torch.load(load_path, map_location=self.device)
                self.optimizer.load_state_dict(optim_state)
                print(f"Loaded optimizer from {optim_filename}")
            except Exception as e:
                print(e)
                print(f"Optimizer loading failed. Starting training from initial optimizer settings...")
            
    def load_network(self, which_epoch, loaddir):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(loaddir, save_filename)
        if os.path.exists(load_path): 
            print('loading the model from %s' % load_path)
        else:
            raise ValueError(f"No saved model {load_path}")
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)
        
    def save_network(self, which_epoch, wipe=False):
        """save model to disk"""
        if wipe == True: 
            import re 
            # Wipe all previous epochs 
            search = re.compile(r"\d+_[a-zA-Z]+.pth")
            deletefiles = list(filter(search.match, os.listdir(self.save_dir)))
            for file in deletefiles: 
                os.unlink(os.path.join(self.save_dir, file))
            
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)
        
        # Also save the optimizer state 
        if self.is_train == True:
            save_filename = '%s_optim.pth' % (which_epoch)
            save_path = join(self.save_dir, save_filename)
            torch.save(self.optimizer.state_dict(), save_path)
                            
    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
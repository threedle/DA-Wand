import torch
import torch.nn as nn
from torch.nn import ConstantPad2d

class MeshUnpool(nn.Module):
    def __init__(self, unroll_target, type="unweight"):
        super(MeshUnpool, self).__init__()
        # NOTE: Padding to this target is problematic when an input mesh has irreducible triangulation w/ # edges > unroll_target
        self.unroll_target = unroll_target
        self.type = type

    def __call__(self, features, meshes, unpools, weights=None, target_n = None):
        return self.forward(features, meshes, unpools, weights, target_n)

    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows = unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def forward(self, features, meshes, unpools, weights=None, target_n = None):
        # Edge case: unpools is None 
        if unpools is None: 
            return features 
        new_features = []
        for i in range(len(meshes)):
            mesh = meshes[i]
            # NOTE: We assume weights and unpools are already reversed here
            mesh_unpools = unpools[i]
            mesh_features = features[i]
            old_keys_unsorted = torch.tensor(list(mesh.topology.edges.keys())).long().to(mesh_features.device)
            old_keys = torch.tensor(list(sorted(mesh.topology.edges.keys()))).long().to(mesh_features.device)
            if weights is not None:
                mesh_weights = weights[i]
                assert len(mesh_weights) == len(mesh_unpools)
            # Need to loop through each unpool iteratively, b/c coarse edge features also change with sequence
            for unpool in mesh_unpools:
                unpool.apply()
                
            edge_keys = torch.tensor(list(sorted(mesh.topology.edges.keys()))).long().to(mesh_features.device)
            edges_count = len(edge_keys)
            topo_to_inds = torch.zeros(torch.max(edge_keys) + 1).long().to(mesh_features.device)
            topo_to_inds[edge_keys] = torch.arange(edges_count).to(mesh_features.device)
            old_indices = topo_to_inds[old_keys]
            new_mesh_features = torch.zeros((mesh_features.shape[0], edges_count)).to(mesh_features.device)
            # NOTE: Here we assume the input features are edge 0-padded on the right
            new_mesh_features[:, old_indices] = mesh_features[:,:len(old_keys)]
            # Now split each old edge into weighted components based on split sequence
            for i in range(len(mesh_unpools)):
                unpool = mesh_unpools[i]
                left_edge_index = topo_to_inds[unpool.e_left_id]
                right_edge_index = topo_to_inds[unpool.e_right_id]
                new_e_index = topo_to_inds[unpool.new_e_bundle[0]]
                new_e_left_index = topo_to_inds[unpool.new_e_left_bundle[0]]
                new_e_right_index = topo_to_inds[unpool.new_e_right_bundle[0]]

                # MeshCNN original upsample: copy adjacent edges and rebuilt center edge is mean
                if weights is None:
                    # NOTE: we have to update edge features ITERATIVELY (since new edge feature may be dependent on a previously collapsed edge)
                    assert torch.all(new_mesh_features[:, new_e_left_index] == 0), f"Error: overwriting already visited edge"
                    assert torch.all(new_mesh_features[:, new_e_right_index] == 0), f"Error: overwriting already visited edge"
                    assert torch.all(new_mesh_features[:, new_e_index] == 0), f"Error: overwriting already visited edge"
                    assert not torch.all(new_mesh_features[:,left_edge_index] == 0), f"Error: unvisited edge encountered in unpool"
                    assert not torch.all(new_mesh_features[:,right_edge_index] == 0), f"Error: unvisited edge encountered in unpool"
                    new_mesh_features[:, new_e_left_index] = new_mesh_features[:, left_edge_index]
                    new_mesh_features[:, new_e_right_index] = new_mesh_features[:, right_edge_index]
                    new_mesh_features[:, new_e_index] = (new_mesh_features[:, left_edge_index] + new_mesh_features[:, right_edge_index]) / 2                    
                # Inverse weighting upsample
                else:
                    left_weights, right_weights = mesh_weights[i]
                    left_weights = torch.tensor(left_weights, device=mesh_features.device)
                    right_weights = torch.tensor(right_weights, device=mesh_features.device)
                    left_features = new_mesh_features[:, left_edge_index].unsqueeze(1) * left_weights
                    right_features = new_mesh_features[:, right_edge_index].unsqueeze(1) * right_weights
                    # Assign new feature values
                    new_mesh_features[:, left_edge_index] = left_features[:,0]
                    new_mesh_features[:, new_e_left_index] = left_features[:,1]
                    new_mesh_features[:, right_edge_index] = right_features[:, 0]
                    new_mesh_features[:, new_e_right_index] = right_features[:, 1]
                    new_mesh_features[:, new_e_index] = (left_features[:,2] + right_features[:,2])/2
            new_features.append(new_mesh_features)
            
        # Pad each set of new mesh features to either input target or max edges
        new_target = target_n
        if target_n is None:
            new_target = max([fe.shape[1] for fe in new_features])
        for i in range(len(new_features)):
            fe = new_features[i]
            diff = new_target - fe.shape[1]
            if diff > 0:
                pad = ConstantPad2d((0, diff, 0, 0), 0)
                new_features[i] = pad(fe)
        new_features = torch.stack(new_features)
        
        return new_features
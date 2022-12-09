import torch
import torch.nn as nn
from threading import Thread
from models.layers.meshing.analysis import computeEdgeNeighborMatrix
import numpy as np
from heapq import heappop, heapify
from .meshing.edit import EdgeCollapse
from torch.nn import ConstantPad2d
from collections import defaultdict

class MeshPool(nn.Module):

    def __init__(self, target, order="norm", multi_thread=False, save_pool_weights = False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]
        self.__save_pool_weights = save_pool_weights
        self.__order = order 

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__unpools = defaultdict(list)
        self.__pool_weights = defaultdict(list)
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        
        # Debugging: show old and new pooled meshes
        # import polyscope as ps 
        # ps.init() 
        # for i in range(len(meshes)): 
        #     vs, fs, _ = meshes[i].export_soup() 
        #     ps_mesh = ps.register_surface_mesh(f"mesh{i}", vs, fs, edge_width=1)

        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        # Pad updated features to new max-edge mesh
        new_target = max([fe.shape[1] for fe in self.__updated_fe])
        
        # Debug
        # for i in range(len(meshes)): 
        #     vs, fs, _ = meshes[i].export_soup() 
        #     ps_mesh = ps.register_surface_mesh(f"mesh{i}_pool", vs, fs, edge_width=1)
        # ps.show()
        
        for i in range(len(self.__updated_fe)):
            fe = self.__updated_fe[i]
            diff = new_target - fe.shape[1]
            if diff > 0:
                self.__updated_fe[i] = torch.nn.functional.pad(fe, (0, diff, 0, 0))
        out_features = torch.stack(self.__updated_fe)
        # Feature channels always stays same
        assert out_features.shape[1] == self.__fe.shape[1], f"MeshPool: Expected channel dims {self.__fe.shape[1]}, got shape {out_features.shape}" 
        # Reverse unpool lists
        for key, val in self.__unpools.items():
            self.__unpools[key] = list(reversed(val))
        for key, val in self.__pool_weights.items():
            self.__pool_weights[key] = list(reversed(val))
        # print(f"Pooled features shape: {out_features.shape}")
        return out_features, self.__unpools, self.__pool_weights

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        edge_keys = torch.tensor(list(sorted(mesh.topology.edges.keys()))).long().to(self.__fe.device)
        edges_count = len(edge_keys)
        # NOTE: This is padded -- take only the non-padded features
        fe = self.__fe[mesh_index][:,:edges_count]
        
        topo_to_inds = torch.zeros(torch.max(edge_keys) + 1).long().to(fe.device)
        topo_to_inds[edge_keys] = torch.arange(edges_count).to(fe.device)
        inds_to_topo = edge_keys 
        
        if self.__order == "similarity":
            edge_neighbors = torch.tensor([list(set([edge.index for edge in mesh.topology.edges[e.item()].halfedge.face.adjacentEdges()] + 
                                           [edge.index for edge in mesh.topology.edges[e.item()].halfedge.twin.face.adjacentEdges()]))[1:] for e in edge_keys]).to(fe.device)
            edge_neighbor_inds = topo_to_inds[edge_neighbors]
            similarities = torch.mean(torch.sum((fe.unsqueeze(2) - fe[:,edge_neighbor_inds])**2, dim=0), dim=1) 
            assert len(similarities) == fe.shape[1], f"MeshPool similarity: expected edge count {fe.shape[1]}, got {len(similarities)} edge comparisons"
            sorted_edge_ids = torch.argsort(similarities)
            # Debugging
            # print(fe[:,sorted_edge_ids[0]])
            # print(similarities[sorted_edge_ids[0]])
            # print(edge_neighbor_inds[sorted_edge_ids[0]])
            # print(torch.sum((fe[:,sorted_edge_ids[0]].unsqueeze(1) - fe[:,edge_neighbor_inds[sorted_edge_ids[0]]])**2, dim=0))
            # raise 
        elif self.__order == "neighborsim":
            # Collapse edges for which the corresponding neighbors are most similar 
            edge_neighbors = torch.stack([
                                        torch.tensor([[edge.index for edge in mesh.topology.edges[e.item()].halfedge.face.adjacentEdges() if e != edge.index] for e in edge_keys]), 
                                        torch.tensor([[edge.index for edge in mesh.topology.edges[e.item()].halfedge.twin.face.adjacentEdges() if e != edge.index] for e in edge_keys])
                                          ], dim=-1).to(fe.device)
            edge_neighbor_inds = topo_to_inds[edge_neighbors]
            fe_neighbor = fe[:, edge_neighbor_inds]
            similarities = torch.sum(torch.sum((fe_neighbor[:,:,:,0] - fe_neighbor[:,:,:,1])**2, dim=-1), dim=0)
            assert len(similarities) == fe.shape[1], f"MeshPool similarity: expected edge count {fe.shape[1]}, got {len(similarities)} edge comparisons"
            sorted_edge_ids = torch.argsort(similarities)
            # Debugging
            # print(fe[:,sorted_edge_ids[0]])
            # print(fe_neighbor[:,sorted_edge_ids[0],:, :])
            # print(similarities[sorted_edge_ids[0]])
            # print(edge_neighbor_inds[sorted_edge_ids[0]])
            # raise 
        elif self.__order == "anchordot":
            # Rank based on average dot product between anchor edge features and other edge features 
            # First normalize all features 
            fe_norm = fe/torch.linalg.norm(fe, dim=0)
            anchor_fs = mesh.anchor_fs
            # Aggregate all edge features associated with anchor vertices
            edge_keys = torch.tensor(list(sorted(mesh.topology.edges.keys()))).long()
            topo_edge_map = torch.zeros(torch.max(edge_keys) + 1).long()
            topo_edge_map[edge_keys] = torch.arange(len(edge_keys))
            cond_edges = []
            for f in anchor_fs:
                cond_edges += list(e.index for e in mesh.topology.faces[f].adjacentEdges())
            cond_edges, sort_edge_indices = torch.sort(torch.unique(torch.tensor(cond_edges)).long())
            cond_edges = topo_edge_map[cond_edges]
            cond_edge_features = fe_norm[:,cond_edges]
            anchorscore = torch.matmul(fe_norm.transpose(1,0), cond_edge_features.unsqueeze(0))
            anchorscore = torch.mean(anchorscore.squeeze(), dim=1)
            sorted_edge_ids = torch.argsort(anchorscore)
        elif self.__order == "revanchordot":
            # Pool highest similarity first 
            # First normalize all features 
            fe_norm = fe/torch.linalg.norm(fe, dim=0)
            anchor_fs = mesh.anchor_fs
            # Aggregate all edge features associated with anchor vertices
            edge_keys = torch.tensor(list(sorted(mesh.topology.edges.keys()))).long()
            topo_edge_map = torch.zeros(torch.max(edge_keys) + 1).long()
            topo_edge_map[edge_keys] = torch.arange(len(edge_keys))
            cond_edges = []
            for f in anchor_fs:
                cond_edges += list(e.index for e in mesh.topology.faces[f].adjacentEdges())
            cond_edges, sort_edge_indices = torch.sort(torch.unique(torch.tensor(cond_edges)).long())
            cond_edges = topo_edge_map[cond_edges]
            cond_edge_features = fe_norm[:,cond_edges]
            anchorscore = torch.matmul(fe_norm.transpose(1,0), cond_edge_features.unsqueeze(0))
            anchorscore = torch.mean(anchorscore.squeeze(), dim=1)
            sorted_edge_ids = torch.argsort(anchorscore, descending=True)
        elif self.__order == "norm":
            sorted_edge_ids = torch.argsort(torch.sum(fe ** 2, dim=0))
        
        ordered_edge_keys = inds_to_topo[sorted_edge_ids].cpu().tolist()
        assert edges_count == len(ordered_edge_keys), f"MeshPool similarity: expected edge count {edges_count}, got {len(ordered_edge_keys)} sorted keys"
        # NOTE: With this method, we will not COMPUTE the pooling iteratively (i.e. pooling will all be done with the ORIGINAL features)
        pool_mat = torch.eye(fe.shape[1], device=fe.device) # E x E (original edge count)
        # print(f"Pooling for mesh {mesh_index}. Initial edges count: {edges_count}. Target: {self.__out_target}.")
        pool_count = 0 
        # Don't let mesh flatten beyond tetrahedron
        while pool_count < self.__out_target and edges_count >= 4:
            success = False
            for edge_key in ordered_edge_keys:
                edt = EdgeCollapse(mesh, edge_key)
                if edt.do_able:
                    # Only necessary external check: edge doesn't collapse an anchor face
                    incident_faces = [mesh.topology.edges[edge_key].halfedge.face.index, 
                                      mesh.topology.edges[edge_key].halfedge.twin.face.index]
                    if incident_faces[0] in mesh.anchor_fs or incident_faces[1] in mesh.anchor_fs:
                        continue
                    self.__unpools[mesh_index].append(edt.inverse())
                    edt.apply()
                    edges_count = len(list(mesh.topology.edges.keys()))

                    # Save collapse indices for pooling
                    v_top_id, v_bottom_id, e_left_id, e_right_id, v_top_coord, v_bottom_coord, deleted_e_bundle, \
                    deleted_e_left_bundle, deleted_e_right_bundle, deleted_f_bundle = edt.record
                    # Convert topology (fixed) indices to current (collapsed) indices
                    e_left_index = topo_to_inds[e_left_id]
                    deleted_e_left_index = topo_to_inds[deleted_e_left_bundle[0]]
                    e_right_index = topo_to_inds[e_right_id]
                    deleted_e_right_index = topo_to_inds[deleted_e_right_bundle[0]]
                    deleted_e_index = topo_to_inds[deleted_e_bundle[0]]
                    left_pool = [e_left_index, deleted_e_left_index,
                                 deleted_e_index]
                    right_pool = [e_right_index, deleted_e_right_index,
                                  deleted_e_index]
                    pool_mat[left_pool, e_left_index] = 1
                    pool_mat[right_pool, e_right_index] = 1

                    # Remove collapsed edge from iteration
                    ordered_edge_keys.remove(edge_key)
                    success=True
                    pool_count += 3
                    break
            # If we run through all valid keys, then collapsing is over
            if success == False:
                break
            
        # Debugging: values collapsed correctly
        unpool = self.__unpools[mesh_index][0]
        left_edge_index = topo_to_inds[unpool.e_left_id]
        right_edge_index = topo_to_inds[unpool.e_right_id]
        deleted_e_index = topo_to_inds[unpool.new_e_bundle[0]]
        deleted_e_left_index = topo_to_inds[unpool.new_e_left_bundle[0]]
        deleted_e_right_index = topo_to_inds[unpool.new_e_right_bundle[0]]
        print(fe[:,left_edge_index])
        print(fe[:,right_edge_index])
        print(fe[:,deleted_e_index])
        print(fe[:,deleted_e_left_index])
        print(fe[:,deleted_e_right_index])
        print(pool_mat[[left_edge_index, right_edge_index, deleted_e_index, deleted_e_left_index, deleted_e_right_index], left_edge_index])
        print(fe.shape) 
        print(pool_mat.shape)
        
        fe = torch.matmul(fe, pool_mat)
        fe /= torch.sum(pool_mat, dim=0, keepdim=True) # Mean 
        self.__updated_fe[mesh_index] = fe[:, topo_to_inds[list(sorted(mesh.topology.edges.keys()))]]
        # print(f"Done. # edges: {len(keepcols)}")
        print(fe[:, left_edge_index])
        print(fe[:, right_edge_index])
        raise 
    
        # Recompute edge neighborhood matrix 
        computeEdgeNeighborMatrix(mesh)
        
        # Edges consistent for topology and features
        assert edges_count == self.__updated_fe[mesh_index].shape[1], f"MeshPool: expected edge counts {edges_count}, got features with {self.__updated_fe[mesh_index].shape[1]} edges"

    def __build_queue(self, features, edge_ids):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids.unsqueeze(-1).to(squared_magnitude.device)), dim=-1).tolist()
        heapify(heap)
        return heap
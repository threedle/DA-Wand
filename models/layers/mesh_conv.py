import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        # TODO: RETRAIN WITHOUT OG CONV BELOW 
        self.ogconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=True)
        self.conv = nn.Linear(in_features=5 * in_channels, out_features=out_channels)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        # Build neighborhood image: M x E x C*5 (5 edge neighbors)
        G = self.build_GeMM_linear(x, mesh)
        # Debugging: fix initialization of these layers and compare the calculations 
        # import time 
        # og_G = self.build_GeMM(x, mesh)
        # for initval in torch.linspace(-0.5, 0.5, 10):
        #     og_random_weights = torch.rand((32, 7, 1, 5))
        #     og_random_bias = torch.rand(32)
        #     random_weights = og_random_weights.transpose(3,1).reshape(32, -1)
        #     random_bias = og_random_bias
        #     self.ogconv.weight.data = og_random_weights
        #     self.ogconv.bias.data = og_random_bias
        #     self.conv.weight.data = random_weights
        #     self.conv.bias.data = random_bias
        #     # self.ogconv.weight.data.fill_(initval)
        #     # self.ogconv.bias.data.fill_(initval)
        #     # self.conv.weight.data.fill_(initval)
        #     # self.conv.bias.data.fill_(initval)
        #     t0 = time.time() 
        #     og_x = self.ogconv(og_G)
        #     print(f"MeshConv2D time: {time.time() - t0:0.5f} seconds")
        #     t0 = time.time() 
        #     # For linear: collapse all neighbor channels into one dimension
        #     print(og_G[0,:,0,:])
        #     print(G[0,0,:])
        #     lin_x = self.conv(G)
        #     lin_x = lin_x.transpose(2,1).unsqueeze(3)
        #     print(f"Linear meshconv time: {time.time() - t0:0.5f} seconds")
        #     print(og_x.shape)
        #     print(lin_x.shape)
        #     assert torch.allclose(og_x, lin_x, atol=1e-5), f"Max channel difference: {torch.max(og_x - lin_x)}"
        # raise 
        x = self.conv(G).transpose(2,1).unsqueeze(3)
        return x
    
    def build_GeMM_linear(self, x, mesh):
        # Input dim: M x C x E 
        x = x.transpose(2,1)
        # def get_edge_nbr_index(edge, index_map):
        #     inds = [edge.index, edge.halfedge.next.edge.index, edge.halfedge.next.next.edge.index,
        #                 edge.halfedge.twin.next.edge.index, edge.halfedge.twin.next.next.edge.index]
        #     return index_map[inds]
        
        image = [] 
        for i in range(len(mesh)):
            # Build edge index map
            tmp_mesh = mesh[i]
            tmp_x = x[i]
            
            edge_keys = torch.tensor(list(sorted(tmp_mesh.topology.edges.keys())))
            topo_edge_map = torch.zeros(torch.max(edge_keys)+1).long()
            topo_edge_map[edge_keys] = torch.arange(len(edge_keys))            
            mesh_image = tmp_x[topo_edge_map[torch.from_numpy(tmp_mesh.edgemat)]].transpose(2,1)
            mesh_image = torch.nn.functional.pad(mesh_image, (0,0,0,0,0,tmp_x.shape[0] - mesh_image.shape[0]))
            image.append(mesh_image)
        
        # Debugging 
        # def get_edge_nbr_index(edge):
        #     inds = [edge.index, edge.halfedge.next.edge.index, edge.halfedge.next.next.edge.index,
        #                 edge.halfedge.twin.next.edge.index, edge.halfedge.twin.next.next.edge.index]
        #     return inds  
        
        # og_image = []
        # for i in range(len(mesh)):
        #     # Build edge index map
        #     tmp_mesh = mesh[i]
        #     tmp_x = x[i]
        #     edge_keys = torch.tensor(list(sorted(tmp_mesh.topology.edges.keys())))
        #     topo_edge_map = torch.zeros(torch.max(edge_keys)+1).long()
        #     topo_edge_map[edge_keys] = torch.arange(len(edge_keys))
            
        #     # Get edge neighbors and build initial feature neighborhood
        #     mesh_neighborhood = []
        #     for e_id in edge_keys:
        #         edge = tmp_mesh.topology.edges[e_id.item()]
        #         mesh_neighborhood.append(tmp_x[topo_edge_map[get_edge_nbr_index(edge)],:].transpose(1,0)) # 5 x C
        #     mesh_neighborhood = torch.stack(mesh_neighborhood) # E x C x 5
        #     # Pad to fixed number of edges
        #     mesh_neighborhood = torch.nn.functional.pad(mesh_neighborhood, (0,0,0,tmp_x.shape[0] - mesh_neighborhood.shape[0],0,0))
        #     og_image.append(mesh_neighborhood) # E x C x 5
        # og_image = torch.stack(og_image)
        
        try:
            image = torch.stack(image).contiguous() # M x E x C x 5
        except Exception as e: 
            print(e)
            image = torch.stack(image)
        
        # Apply symmetric operations
        x_1 = image[:, :, :, 1] + image[:, :, :, 3]
        x_2 = image[:, :, :, 2] + image[:, :, :, 4]
        x_3 = torch.abs(image[:, :, :, 1] - image[:, :, :, 3])
        x_4 = torch.abs(image[:, :, :, 2] - image[:, :, :, 4])
        
        # Debug: pooled convs 
        # if x.shape[1] > 5:
        #     tmp_x = x[0] 
        #     tmp_mesh = mesh[0]
        #     edge_keys = torch.tensor(list(sorted(tmp_mesh.topology.edges.keys())))
        #     edge = tmp_mesh.topology.edges[edge_keys[0].item()]
        #     print(tmp_x[:,topo_edge_map[get_edge_nbr_index(edge)]])
        #     print(x_1[0,:,0])
        #     print(x_2[0,:,0])
        #     print(x_3[0,:,0])
        #     print(x_4[0,:,0])
        #     raise 
        image = torch.cat([image[:, :, :, 0], x_1, x_2, x_3, x_4], dim=2) # M x E x C*5
        return image
    
    def build_GeMM(self, x, mesh):
        # def get_edge_nbr_index(edge, index_map):
        #     inds = [edge.index, edge.halfedge.next.edge.index, edge.halfedge.next.next.edge.index,
        #                 edge.halfedge.twin.next.edge.index, edge.halfedge.twin.next.next.edge.index]
        #     return index_map[inds]
        def get_edge_nbr_index(edge):
            inds = [edge.index, edge.halfedge.next.edge.index, edge.halfedge.next.next.edge.index,
                        edge.halfedge.twin.next.edge.index, edge.halfedge.twin.next.next.edge.index]
            return inds  
        image = []
        for i in range(len(mesh)):
            # Build edge index map
            tmp_mesh = mesh[i]
            tmp_x = x[i]
            edge_keys = torch.tensor(list(sorted(tmp_mesh.topology.edges.keys())))
            topo_edge_map = torch.zeros(torch.max(edge_keys)+1).long()
            topo_edge_map[edge_keys] = torch.arange(len(edge_keys))

            # Get edge neighbors and build initial feature neighborhood
            mesh_neighborhood = []
            for e_id in edge_keys:
                edge = tmp_mesh.topology.edges[e_id.item()]
                mesh_neighborhood.append(tmp_x[:,topo_edge_map[get_edge_nbr_index(edge)]]) # C x 5
                
                # Visualize edge neighbors 
                # import sys 
                # sys.path.append("../../")
                # from util.util import polyscope_edge_perm
                # eperm = polyscope_edge_perm(tmp_mesh)
                # # Map ordered features to original edge order in mesh 
                # edge_topo_map = torch.zeros(tmp_x.shape[1]).long()
                # edge_topo_map[edge_indices] = edge_keys 
                
                # import polyscope as ps 
                # import numpy as np
                # ps.init() 
                # vs, fs, _ = tmp_mesh.export_soup() 
                # ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width = 1)
                # ps_mesh.set_edge_permutation(eperm)
                # edge_nbrs = get_edge_nbr_index(edge)
                # nbr_colors = np.zeros(len(edge_keys))
                # nbr_colors[edge_nbrs] = 2
                # nbr_colors[e_id] = 1
                # ps_mesh.add_scalar_quantity("nbrs", nbr_colors, defined_on='edges', enabled=True)
                # # Show the different edge features 
                # topo_x = tmp_x[:,edge_topo_map]  
                # ps_mesh.add_scalar_quantity("dihedrals", topo_x[0].detach().numpy(), defined_on='edges', enabled=True) 
                # ps_mesh.add_scalar_quantity("angle1", topo_x[1].detach().numpy(), defined_on='edges', enabled=True) 
                # ps_mesh.add_scalar_quantity("angle2", topo_x[2].detach().numpy(), defined_on='edges', enabled=True) 
                # ps_mesh.add_scalar_quantity("ratio1", topo_x[3].detach().numpy(), defined_on='edges', enabled=True) 
                # ps_mesh.add_scalar_quantity("ratio2", topo_x[4].detach().numpy(), defined_on='edges', enabled=True) 
                # ps.show()
                # raise 
            mesh_neighborhood = torch.stack(mesh_neighborhood) # E x C x 5
            # Pad to fixed number of edges
            mesh_neighborhood = torch.nn.functional.pad(mesh_neighborhood, (0,0,0,0,0,tmp_x.shape[1] - mesh_neighborhood.shape[0]))
            image.append(mesh_neighborhood.transpose(1,0)) # C x E x 5

        image = torch.stack(image) # M x C x E x 5
        # Apply symmetric operations
        x_1 = image[:, :, :, 1] + image[:, :, :, 3]
        x_2 = image[:, :, :, 2] + image[:, :, :, 4]
        x_3 = torch.abs(image[:, :, :, 1] - image[:, :, :, 3])
        x_4 = torch.abs(image[:, :, :, 2] - image[:, :, :, 4])
        
        # Debug: pooled convs 
        # if x.shape[1] > 5:
        #     tmp_x = x[0] 
        #     tmp_mesh = mesh[0]
        #     edge_keys = torch.tensor(list(sorted(tmp_mesh.topology.edges.keys())))
        #     edge = tmp_mesh.topology.edges[edge_keys[0].item()]
        #     print(tmp_x[:,topo_edge_map[get_edge_nbr_index(edge)]])
        #     print(x_1[0,:,0])
        #     print(x_2[0,:,0])
        #     print(x_3[0,:,0])
        #     print(x_4[0,:,0])
        #     raise 

        image = torch.stack([image[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return image

    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift

        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        edges_count = len(list(m.topology.edges.keys()))
        padded_gemm = torch.cat((torch.arange(edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm

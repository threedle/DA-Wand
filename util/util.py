from __future__ import print_function
import torch
import numpy as np
import os
import time
from pathlib import Path
import scipy
import shutil
import fresnel

def time_function(func):
    def wrapping_fun(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('Run time of %s is %4.2fs' % (func.__name__, (end - start)))
        return result
    return wrapping_fun

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def norm_to_range(arr, new_range):
	return np.interp(np.array(arr), (np.amin(arr), np.amax(arr)), new_range)

def signed_volume(v, f):
    # Add up signed volume of tetrahedra for each face
    # If triangles, then one of these vertices is the origin
    if f.shape[1] == 3:
        f = np.hstack([f, np.ones(len(f)).reshape(len(f), 1) * len(v)]).astype(int)
        v = np.vstack([v, np.zeros(3).reshape(1, 3)])
    fverts = v[f]
    fvectors = fverts - fverts[:,3, None,:]
    # Triple scalar product
    volume = 1/6 * np.sum(np.sum(fvectors[:,0,:] * np.cross(fvectors[:,1,:], fvectors[:,2,:], axis=1), axis=1))
    return volume

# Returns new faces array with orientation fixed
def fix_orientation(vertices, faces):
    from igl import bfs_orient
    new_faces, c = bfs_orient(faces)
    new_faces = new_faces.astype(int)

    # Edge case: only one face
    if len(new_faces.shape) == 1:
        new_faces = new_faces.reshape(1,3)

    volume = signed_volume(vertices, new_faces)
    if volume < 0:
        new_faces = np.fliplr(new_faces)
    return new_faces

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

MESH_EXTENSIONS = [
    '.obj'
]

class ZeroNanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad[grad != grad] = 0
        return grad

def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    if target_length <= shp[dim]:
        return input_arr
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)

def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy

# Compute HKS
def compute_hks(evals, evecs, scales):
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1) # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out

# Run graphcuts
def graphcuts(preds, mesh, pairwise=None, unary=-15, anchors=None):
    from pygco import cut_from_graph
    face_adj = np.array([[edge.halfedge.face.index, edge.halfedge.twin.face.index] for key, edge in sorted(mesh.topology.edges.items())])

    if not hasattr(mesh, "dihedrals"):
        from models.layers.meshing.analysis import computeDihedrals
        computeDihedrals(mesh)

    dihedrals = np.clip(np.pi - mesh.dihedrals, 0, np.pi).squeeze()

    # TODO: maybe send all dihedrals past 90* to smoothness cost 0
    # Maps dihedrals from 0 => infty
    smoothness = -np.log(dihedrals/np.pi + 1e-15)
    edges = np.ceil(np.concatenate([face_adj, smoothness[:,None]], axis=1)).astype(np.int32)

    if pairwise is None:
        pairwise = 0 * np.eye(2, dtype=np.int32)
        pairwise[0,1] = 10
        pairwise[1,0] = 10

    # TODO: scale unary costs by geodesic distance with gaussian dropoff
    # Selection nearby: super high weight, rapidly drop down as moves further away
    # Non-selection: fixed cost regardless of distance
    unaries = np.ones((len(preds), 2))
    unaries[:, 1] = (preds - 0.5) * unary
    unaries[:, 0] = -unaries[:, 1]
    # Assign large value to anchor
    if anchors is not None:
        unaries[anchors, 1] = -10000
        unaries[anchors,0] = 10000
    unaries = unaries.astype(np.int32)
    cut_graph = cut_from_graph(edges, unaries, pairwise, n_iter=-1, algorithm='swap')
    return cut_graph

# NOTE: This assumes default view direction of (0, 0, -r)
def get_camera_from_view(elev, azim, r=2.0):
    x = r * np.cos(azim) *  np.sin(elev)
    y = r * np.sin(azim) * np.sin(elev)
    z = r * np.cos(elev)

    pos = np.array([x, y, z])
    look_at = -pos
    direction = np.array([0.0, 1.0, 0.0])
    return pos, look_at, direction

def polyscope_edge_perm(mesh):
    # Need to map edge ordering based on polyscope's scheme
    vs, fs, _ = mesh.export_soup()
    polyscope_edges = []
    for f in fs:
        for i in range(len(f)):
            e_candidate = {f[i], f[(i+1)%3]}
            if e_candidate not in polyscope_edges:
                polyscope_edges.append(e_candidate)
    mesh_edges = [set([v.index for v in e.two_vertices()]) for e in mesh.topology.edges.values()]
    # Build permutation
    edge_p = []
    for edge in polyscope_edges:
        found = 0
        for i in range(len(mesh_edges)):
            meshe = mesh_edges[i]
            if edge == meshe:
                edge_p.append(i)
                found = 1
                break
        if found == 0:
            raise ValueError(f"No match found for polyscope edge {edge}")
    return np.array(edge_p)

def export_views(mesh, savedir, n=5, n_sample=20, width=150, height=150, plotname="Views", filename="test", vcolors=None,
                 device="cpu", outline_width=0.005, anchor_fs=None):
    import matplotlib, matplotlib.cm

    fresnel_device = fresnel.Device(mode=device)
    scene = fresnel.Scene(device=fresnel_device)
    vertices, faces, _ = mesh.export_soup()
    fnormals = mesh.facenormals
    fverts = vertices[faces].reshape(3 * len(faces), 3)
    mesh = fresnel.geometry.Mesh(scene, vertices=fverts, N=1)
    mesh.material = fresnel.material.Material(color=fresnel.color.linear([0.25, 0.5, 0.9]), roughness=0.1)
    mesh.outline_material = fresnel.material.Material(color=(0., 0., 0.), roughness=0.1, metal=1.)
    if vcolors is not None:
        mesh.color[:] = fresnel.color.linear(vcolors)
    mesh.material.primitive_color_mix = 1.0
    mesh.outline_width=outline_width

    # New primitives for anchors
    if anchor_fs is not None:
        anchor_f_inds = faces[anchor_fs]
        # Offset from surface to make visible
        anchor_fverts = vertices[anchor_f_inds] + (fnormals[anchor_fs]*0.01).reshape(len(anchor_fs), 3, 1)
        anchor_fverts = anchor_fverts.reshape(3 * len(anchor_f_inds), 3)
        anchor_mesh = fresnel.geometry.Mesh(scene, vertices=anchor_fverts, N=1)
        anchor_mesh.material = fresnel.material.Material(color=fresnel.color.linear([1.0, 1.0, 1.0]), roughness=0.1)
        anchor_mesh.outline_material = fresnel.material.Material(color=(0., 0., 0.), roughness=0.1, metal=1.)
        anchor_mesh.outline_width = outline_width * 1.5
        anchor_mesh.material.primitive_color_mix = 0.0

    scene.lights = fresnel.light.cloudy()

    # TODO: maybe initializing with fitting gives better camera angles
    scene.camera = fresnel.camera.Orthographic.fit(scene, margin=0)
    # TODO: initialize to viewing ray that connects origin and average of anchor positions
    # Radius is just largest vertex norm
    r = np.max(np.linalg.norm(vertices))
    elevs = torch.linspace(0, 2 * np.pi, n+1)[:n]
    azims = torch.linspace(-np.pi, np.pi, n+1)[:n]
    renders = []
    for i in range(len(elevs)):
        elev = elevs[i]
        azim = azims[i]
        # Then views are just linspace
        # Loop through all camera angles and collect outputs
        pos, lookat, _ = get_camera_from_view(elev, azim, r=r)
        scene.camera.look_at = lookat
        scene.camera.position = pos
        out = fresnel.pathtrace(scene, samples=n_sample, w=width,h=height)
        renders.append(out[:])
    # Plot and save in matplotlib using imshow
    import matplotlib.pyplot as plt
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig, axs = plt.subplots(nrows=1, ncols=len(renders), gridspec_kw={'wspace':0, 'hspace':0}, figsize=(15, 4), squeeze=True)
    for i in range(len(renders)):
        render = renders[i]
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(render, interpolation='lanczos')
    fig.suptitle(plotname)
    fig.tight_layout()
    plt.savefig(os.path.join(savedir, filename))
    plt.cla()
    plt.close()

# Convert each triangle into local coordinates: A -> (0,0), B -> (x2, 0), C -> (x3, y3)
def get_local_tris(vertices, faces, device=torch.device("cpu")):
    fverts = vertices[faces].to(device)
    e1 = fverts[:, 1, :] - fverts[:, 0, :]
    e2 = fverts[:, 2, :] - fverts[:, 0, :]
    s = torch.linalg.norm(e1, dim=1)
    t = torch.linalg.norm(e2, dim=1)
    angle = torch.acos(torch.sum(e1 / s[:, None] * e2 / t[:, None], dim=1))
    x = torch.column_stack([torch.zeros(len(angle)).to(device), s, t * torch.cos(angle)])
    y = torch.column_stack([torch.zeros(len(angle)).to(device), torch.zeros(len(angle)).to(device), t * torch.sin(angle)])
    local_tris = torch.stack((x, y), dim=-1).reshape(len(angle), 3, 2)
    return local_tris

# Get rotation matrix about vector through origin
def getRotMat(axis, theta):
    """
    axis: np.array, normalized vector
    theta: radians
    """
    import math

    axis = axis/np.linalg.norm(axis)
    cprod = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    rot = math.cos(theta) * np.identity(3) + math.sin(theta) * cprod + \
            (1 - math.cos(theta)) * np.outer(axis, axis)
    return rot

# Map vertices and subset of faces to 0-indexed vertices, keeping only relevant vertices
def trimMesh(vertices, faces):
    if len(faces.shape) < 2:
        faces = faces.reshape(1, len(faces))
    unique_v = np.sort(np.unique(faces.flatten()))
    v_val = np.arange(len(unique_v))
    v_map = dict(zip(unique_v, v_val))
    new_faces = np.array([v_map[i] for i in faces.flatten()]).reshape(faces.shape[0], faces.shape[1])
    new_v = vertices[unique_v]

    return new_v, new_faces

# Compute area of triangulated surface using cross product formula
def meshArea3D(vertices, faces):
    # Get edge vectors
    fverts = vertices[faces]
    edge1 = fverts[:,1,:] - fverts[:,0,:]
    edge2 = fverts[:,2,:] - fverts[:,0,:]

    # Cross product
    cross = torch.cross(edge1, edge2, dim = 1)
    area = torch.sum(0.5 * torch.norm(cross, dim = 1))

    return area

# 2D version of triangulated surface area
def meshArea2D(vertices, faces, return_fareas = False):
    # Get edge vectors
    fverts = vertices[faces]
    edge1 = fverts[:,1,:] - fverts[:,0,:]
    edge2 = fverts[:,2,:] - fverts[:,0,:]

    # Determinant definition of area
    area = 0.5 * torch.abs(edge1[:,0] * edge2[:,1]  - edge1[:,1] * edge2[:,0])

    # Debugging
    # print(area[0])
    # print(fverts[0])
    # exit()

    if return_fareas == True:
        return area
    else:
        return torch.sum(area)

# ====================== Meshing =======================
# Given face topology, return list of contiguous patches indexing into input face array
def contiguous_patches(faces):
    from models.layers.meshing import Topology
    # Remap faces and build halfedge topology
    vkeys = np.sort(np.unique(faces))
    topo_map = np.zeros(np.max(vkeys)+1)
    topo_map[vkeys] = np.arange(len(vkeys))
    faces = topo_map[faces]
    topo = Topology()
    topo.build(len(vkeys), faces)

    # If topology contains isolated faces, then don't use
    if topo.hasIsolatedFaces():
        return None

    # Extract contiguous patches using boundaries
    patches = []
    for i, boundary in topo.boundaries.items():
        init_f = boundary.halfedge.twin.face.index
        patchlist = []
        _get_contig_inds(init_f, topo, patchlist)
        patches.append(np.unique(patchlist))

    # Check: full coverage, no overlaps
    assert len(set(range(len(faces))).difference(set(np.concatenate(patches)))) == 0
    assert len(set(np.concatenate(patches)).difference(set(range(len(faces))))) == 0

    # NOTE: Sometimes boundaries will produce duplicate patches
    delete_patches = []
    from itertools import combinations
    for patchpair in combinations(range(len(patches)), 2):
        n_intersect = len(set(patches[patchpair[0]]).intersection(set(patches[patchpair[1]])))
        # NOTE: This might mean patch is closed: it's fine don't worry about it for now
        if n_intersect == len(patches[patchpair[0]]) == len(patches[patchpair[1]]):
            delete_patches.append(patchpair[1])
        elif n_intersect > 0:
            print(f"Found {n_intersect} intersecting faces between patches. Patch 1: {len(patchpair[0])}. Patch 2: {len(patchpair[1])}.")
        # assert n_intersect == 0, f"Found {n_intersect} intersecting faces between patches. Patch 1: {len(patchpair[0])}. Patch 2: {len(patchpair[1])}."
    unique_patches = [patches[i] for i in range(len(patches)) if i not in delete_patches]
    return unique_patches

# Given starting face and topology, recursively add face neighbors until none left
def _get_contig_inds(face, topo, patchlist):
    patchlist.append(face)
    he = topo.faces[face].halfedge
    fneighbors = [f.index for f in list(topo.faces[face].adjacentFaces())]
    new_faces = list(set(fneighbors).difference(set(patchlist)))
    if len(new_faces) == 0:
        return True
    for f in new_faces:
        _get_contig_inds(f, topo, patchlist)

# ====================== UV Stuff ========================
def make_cut(mesh, cutlist, b0=None, b1=None):
    from models.layers.meshing.edit import EdgeCut

    for i in range(len(cutlist)-1):
        # Cut: 1 new vertex, 1 new edge, two new halfedges
        # Find the halfedge associated with each successive cut
        vsource = cutlist[i]
        vtarget = cutlist[i+1]

        # Instead of assert, just continue if broken
        if not mesh.topology.vertices[vsource].onBoundary():
            continue
        if not vtarget in [v.index for v in mesh.topology.vertices[vsource].adjacentVertices()]:
            continue
        # assert mesh.topology.vertices[vsource].onBoundary()
        # assert vtarget in [v.index for v in mesh.topology.vertices[vsource].adjacentVertices()]

        for he in mesh.topology.vertices[vsource].adjacentHalfedges():
            if he.tip_vertex().index == vtarget:
                break
        edt = EdgeCut(mesh, he.index)
        edt.apply()
        del edt

        # assert np.all(mesh.vertices[vsource] == mesh.vertices[-1])

    # b0 and b1 should iterate through same set of vertices
    if b0 is not None and b1 is not None:
        b0_v = set([v.index for v in mesh.topology.boundaries[b0].adjacentVertices()])
        b1_v = set([v.index for v in mesh.topology.boundaries[b1].adjacentVertices()])

        # Debugging
        if b0_v != b1_v:
            import polyscope as ps
            ps.init()
            ps.remove_all_structures()
            ps_mesh = ps.register_surface_mesh("mesh", mesh.vertices, mesh.faces, edge_width=1)
            b0_colors = np.zeros(len(mesh.vertices))
            b0_colors[list(b0_v)] = 1
            b1_colors = np.zeros(len(mesh.vertices))
            b1_colors[list(b1_v)] = 1
            ps_mesh.add_scalar_quantity("b0", b0_colors, enabled=True)
            ps_mesh.add_scalar_quantity("b1", b1_colors, enabled=True)
            ps.show()

        # assert b0_v == b1_v, f"Boundaries {b0} and {b1} do not coincide after cut!"

        # Delete second boundary
        del mesh.topology.boundaries[b1]

def cut_to_disk(mesh, verbose=False):
    count = 0

    # Don't allow cut if mesh has isolated faces
    if mesh.topology.hasIsolatedFaces():
        return

    while len(mesh.topology.boundaries) > 1:
        if verbose:
            import time
            t0 = time.time()

        # Draw cut starting from longest boundary to nearest boundary -> repeat until only 1 boundary left
        # Get longest boundary
        current_b = 0
        max_b_length = 0
        for b in mesh.topology.boundaries.values():
            b_edge_vs = np.array([list(v.index for v in e.two_vertices()) for e in b.adjacentEdges()])
            b_v_pos = mesh.vertices[b_edge_vs]
            b_length = np.sum(np.linalg.norm(b_v_pos[:,0,:] - b_v_pos[:,1,:], axis=1))
            if b_length > max_b_length:
                current_b = b.index
                max_b_length = b_length

        # Get closest boundary to current boundary from current cut point
        current_boundary = mesh.topology.boundaries[current_b]
        avail_b = list(k for k in mesh.topology.boundaries.keys() if k != current_b)
        subboundary_vs = np.array([v.index for v in current_boundary.adjacentVertices()])

        import igraph as ig
        vs, fs, es = mesh.export_soup()
        edgeweights = [mesh.length(e) for e in mesh.topology.edges.values()]
        graph = ig.Graph(len(vs), es)

        # Compute shortest paths from current boundary vertices to all other boundary vertices
        b_vs = np.array([v.index for b in avail_b for v in mesh.topology.boundaries[b].adjacentVertices()])

        if len(b_vs) == 0:
            print(f"Overlapping boundaries!")
            break

        if verbose:
            print(f"Iteration {count}: graph construct time {time.time() - t0:0.2f} sec.")
            t0 = time.time()

        # Heuristic: initialize to first vertex in subboundary, and compute all shortest paths to all other boundaries
        # Choose shortest path and cut
        cutlists = []
        for init_v in subboundary_vs:
            tmpcutlists = graph.get_shortest_paths(init_v, b_vs, edgeweights)
            if len(tmpcutlists) > 0:
                cutlists.extend(tmpcutlists)
        if len(cutlists) == 0:
            print("No more paths found.")
            break
        # Remove all 0 length paths
        cutlists = [cutlist for cutlist in cutlists if len(cutlist) > 0]
        if len(cutlists) == 0:
            print("No more paths found.")
            break
        cutlens = [len(cut) for cut in cutlists]
        cutlist = cutlists[np.argmin(cutlens)]
        if verbose:
            print(f"Iteration {count}: shortest path calc {time.time() - t0:0.2f} sec.")
            print(f"\tCutlist {cutlist}. # boundaries: {len(mesh.topology.boundaries)}")
            t0 = time.time()

        # Get boundary of target
        shortest_target = cutlist[-1]
        for b in avail_b:
            if shortest_target in [v.index for v in mesh.topology.boundaries[b].adjacentVertices()]:
                next_b = b
                break
        make_cut(mesh, cutlist, current_b, next_b)
        count += 1
        if verbose:
            print(f"Iteration {count}: cutting took {time.time() - t0:0.2f} sec.")
            t0 = time.time()

        if mesh.topology.hasNonManifoldEdges():
            print(f"Mesh became non-manifold from cuts!")
            break

        if count >= 10:
            print(f"Cuts infinite loop.")
            break

def cut_to_disk_single(mesh, singular_vs, verbose=False):
    count = 0

    for target_v in singular_vs:
        if verbose:
            import time
            t0 = time.time()

        # Build weighted edgelist
        weighted_edgelist = [[v.index for v in e.two_vertices()] + [mesh.length(e)] for \
                                e in mesh.topology.edges.values()]

        import igraph as ig
        vs, fs, es = mesh.export_soup()
        edgeweights = [mesh.length(e) for e in mesh.topology.edges.values()]
        graph = ig.Graph(len(vs), es)

        # Compute shortest paths from current target vertex to all other boundary vertices
        b_vs = np.array([v.index for b in mesh.topology.boundaries.values() for v in b.adjacentVertices()])
        cutlists = graph.get_shortest_paths(target_v, b_vs, edgeweights)

        if len(cutlists) == 0:
            print("No path found from vertex to boundary.")
            continue

        cutlens = [len(cut) for cut in cutlists]
        cutlist = cutlists[np.argmin(cutlens)]
        if verbose:
            print(f"Iteration {count}: shortest path calc {time.time() - t0:0.2f} sec.")
            print(f"\tCutlist {cutlist}. # boundaries: {len(mesh.topology.boundaries)}")
            t0 = time.time()

        # Generate cut
        make_cut(mesh, cutlist)
        count += 1

        graph.clear()
        del graph

        if mesh.topology.hasNonManifoldEdges():
            print(f"Mesh became non-manifold from cuts!")
            break

def tutte_embedding(vertices, faces):
    import igl
    bnd = igl.boundary_loop(faces)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(vertices, bnd)

    ## Harmonic parametrization for the internal vertices
    assert not np.isnan(bnd).any()
    assert not np.isnan(bnd_uv).any()
    uv_init = igl.harmonic_weights(vertices, faces, bnd, np.array(bnd_uv, dtype=vertices.dtype), 1)

    return uv_init

def SLIM(mesh, v_with_holes = None, f_with_holes = None):
    # SLIM parameterization
    # Initialize using Tutte embedding
    import igl
    from models.layers.meshing import Mesh

    vs, fs, _ = mesh.export_soup()
    uv_init = tutte_embedding(vs, fs)

    # Need to subset back non-disk topology if filled hole
    if v_with_holes is not None and f_with_holes is not None:
        # Only select UVs relevant to the
        uv_init = uv_init[v_with_holes]
        sub_faces = fs[f_with_holes]
        # NOTE: vertices should now be indexed in the same way as the original sub_faces
        submesh = Mesh(uv_init, sub_faces)
        vs, fs, _ = submesh.export_soup()

    slim = igl.SLIM(vs, fs, uv_init, np.ones((1,1)),np.expand_dims(uv_init[0,:],0), igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, 1.0e1)
    slim.solve(500)
    slim_uv = slim.vertices()
    slim_uv -= slim_uv.mean(axis = 0)
    return slim_uv, slim.energy()

# Duplicates a vertex for each disjoint fan it belongs to
# Use to solve problem of nonmanifold vertices
def cut_vertex(mesh, vind):
    # Find all halfedges that contain the vertex
    heset = set()
    for he in mesh.topology.halfedges.values():
        if he.vertex.index == vind:
            heset.update([he.index])

    # Remove original vertex adjacent halfedge set
    heset = heset.difference(set([he.index for he in mesh.topology.vertices[vind].adjacentHalfedges()]))
    while len(heset) > 0:
        startind = heset.pop()
        starthe = mesh.topology.halfedges[startind]
        currenthe = mesh.topology.halfedges[startind]

        # Duplicate and add new vertex to mesh
        mesh.vertices = np.append(mesh.vertices, [mesh.vertices[vind]], axis=0)
        newv = mesh.topology.vertices.allocate()
        newv.halfedge = currenthe
        assert newv.index == len(mesh.vertices) - 1
        visited = set([currenthe.index])
        while currenthe.twin.next != starthe:
            currenthe.vertex = newv
            currenthe = currenthe.twin.next
            visited.update([currenthe.index])
        currenthe.vertex = newv

        heset = heset.difference(visited)

def run_slim(mesh, cut=True, verbose=False, time=False):
    did_cut = False
    if mesh.topology.hasNonManifoldEdges():
        print(f"run_slim: Non-manifold edges found.")
        return None, None, did_cut

    if cut:
        if time:
            import time
            t0 = time.time()

        # Check for nonmanifold vertices while only one boundary
        if mesh.topology.hasNonManifoldVertices():
            print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
            for vind in mesh.topology.nonmanifvs:
                cut_vertex(mesh, vind)

        if len(mesh.topology.boundaries) > 1:
            cut_to_disk(mesh, verbose)
            did_cut = True

        # Check for nonmanifold vertices while only one boundary
        if mesh.topology.hasNonManifoldVertices():
            print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
            for vind in mesh.topology.nonmanifvs:
                cut_vertex(mesh, vind)

        if not hasattr(mesh, "vertexangle"):
            from models.layers.meshing.analysis import computeVertexAngle
            computeVertexAngle(mesh)

        # Cut cones
        # Only cut cones if one boundary exists
        singlevs = np.where(2 * np.pi - mesh.vertexangle >= np.pi/2)[0]
        if len(singlevs) >= 0 and len(mesh.topology.boundaries) == 1: # Edge case: no boundaries
            cut_to_disk_single(mesh, singlevs, verbose)
            did_cut = True

        # Check for nonmanifold vertices while only one boundary
        if mesh.topology.hasNonManifoldVertices():
            print(f"Cutting nonmanifold vertices: {mesh.topology.nonmanifvs}")
            for vind in mesh.topology.nonmanifvs:
                cut_vertex(mesh, vind)

        # Don't parameterize nonmanifold after cut
        if mesh.topology.hasNonManifoldEdges():
            print(f"run_slim: Cut mesh has nonmanifold edges.")
            return None, None, did_cut

        if time:
            import time
            print(f"Cutting took {time.time() - t0:0.3f} sec.")

    # Compute SLIM
    try:
        uvmap, energy = SLIM(mesh)
    except Exception as e:
        print(e)
        return None, None, did_cut

    assert len(uvmap) == len(mesh.vertices), f"UV: {uvmap.shape}, vs: {mesh.vertices.shape}"

    return uvmap, energy, did_cut

def get_ss(mesh, uvmap):
    import igl

    # mesh.normalize(copy_v=True)
    sub_vs, sub_fs, _ = mesh.export_soup()
    grad = igl.grad(sub_vs, sub_fs) # F*3 x V

    # SLIM singular values
    # Jacobian: need to organize the x,y,z back together
    grad_x = grad @ uvmap[:,0]
    grad_x = np.stack([grad_x[:len(sub_fs)], grad_x[len(sub_fs):2*len(sub_fs)], grad_x[2*len(sub_fs):]], axis=-1)
    grad_y = grad @ uvmap[:,1]
    grad_y = np.stack([grad_y[:len(sub_fs)], grad_y[len(sub_fs):2*len(sub_fs)], grad_y[2*len(sub_fs):]], axis=-1)

    jacob = np.stack([grad_x, grad_y], axis=-1)
    ss = np.linalg.svd(jacob, compute_uv=False) # ss: F x 2

    return ss

# Run differentiable singular value computation
# TODO: Sparse tensors don't work yet with pytorch linalg lstsq backwards I think....
def ss_torch(vertices, faces, uv, gradop = None):
    if gradop is None:
        import igl
        gradop = igl.grad(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()).toarray()
        gradop = torch.from_numpy(gradop).float().to(uv.device)
        # values = gradop.data
        # indices = np.vstack((gradop.row, gradop.col))
        # gradop = torch.sparse_coo_tensor(indices, values, gradop.shape).float()
        # gradop = torch.from_numpy(gradop)

    uv_grad = gradop @ uv # 3F x 2
    jacob = torch.stack([uv_grad[:len(faces)], uv_grad[len(faces):2*len(faces)], uv_grad[2*len(faces):]], dim=1)
    fform = torch.bmm(jacob.transpose(1,2), jacob) # First fundamental form

    # NOTE: We keep sqrt gradient valid by checking for zeros
    discr = fform[:,0,1] * fform[:,1,0] + ((fform[:,0,0] - fform[:,1,1])/2)**2
    buffer = torch.zeros(discr.shape, device = discr.device)
    buffer[torch.where(discr <= 1e-8)] += 1e-8

    eig1 = (fform[:,0,0] + fform[:,1,1])/2 + torch.sqrt(discr + buffer)
    buffer1 = torch.zeros(eig1.shape, device = eig1.device)
    buffer1[torch.where(eig1 <= 1e-8)] += 1e-8
    eig2 = (fform[:,0,0] + fform[:,1,1])/2 - torch.sqrt(discr + buffer)
    buffer2 = torch.zeros(eig2.shape, device = eig2.device)
    buffer2[torch.where(eig2 <= 1e-8)] += 1e-8

    ss1 = torch.sqrt(eig1 + buffer1)
    ss2 = torch.sqrt(eig2 + buffer2)
    ss = torch.stack([ss1,ss2], dim=-1) # F x 2

    if not torch.all(torch.isfinite(ss)):
        print(f"Non-finite singular values from analytical. Trying pytorch SVD...")
        ss = torch.linalg.svdvals(jacob) # ss: F x 2

    return ss

# ====================== Graphical utilities =======================
# Back out camera parameters from view transform matrix
def extract_from_gl_viewmat(gl_mat):
    gl_mat = gl_mat.reshape(4, 4)
    s = gl_mat[0, :3]
    u = gl_mat[1, :3]
    f = -1 * gl_mat[2, :3]
    coord = gl_mat[:3, 3]  # first 3 entries of the last column
    camera_location = np.array([-s, -u, f]).T @ coord
    target = camera_location + f * 10  # any scale
    return camera_location, target

# Polyscope rotation and screenshotting function
def psScreenshot(vertices, faces, axes, angles, save_path, name = "mesh", frame_folder = "frames", scalars = None, colors = None,
                    defined_on = "faces", highlight_faces = None, highlight_color = [1,0,0], highlight_radius=None,
                    cmap = None, sminmax = None, cpos = None, clook = None, up=(0,1,0), save_video = False, save_base = False,
                    show_plane = True, debug=False, edge_color=[0, 0, 0], edge_width=1, overwrite = False):
    import polyscope as ps

    ps.init()
    # Set camera to look at same fixed position in centroid of original mesh
    # center = np.mean(vertices, axis = 0)
    # pos = center + np.array([0, 0, 3])
    # ps.look_at(pos, center)
    if show_plane == False:
        ps.set_ground_plane_mode("none")

    frame_path = f"{save_path}/{frame_folder}"
    Path(frame_path).mkdir(parents=True, exist_ok=True)
    if overwrite == True:
        clear_directory(frame_path)

    if save_base == True:
        ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, enabled=True,
                                        edge_color=edge_color, edge_width=edge_width, material='flat')
        ps.screenshot(f"{frame_path}/{name}.png")
        ps.remove_all_structures()
    # Convert 2D to 3D by appending Z-axis
    if vertices.shape[1] == 2:
        vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis = 1)

    count = 0
    for axis in axes:
        for i in range(len(angles)):
            rot = getRotMat(axis, angles[i])
            rot_verts = np.transpose(rot @ np.transpose(vertices))

            ps_mesh = ps.register_surface_mesh("mesh", rot_verts, faces, enabled=True,
                                            edge_color=edge_color, edge_width=edge_width, material='flat')
            if scalars is not None:
                ps_mesh.add_scalar_quantity(f"scalar", scalars, defined_on=defined_on,
                                            cmap=cmap, enabled=True, vminmax = sminmax)
            if colors is not None:
                ps_mesh.add_color_quantity(f"color", colors, defined_on=defined_on,
                                            enabled=True)
            if highlight_faces is not None:
                # Create curve to highlight faces
                curve_v, new_f = trimMesh(rot_verts, faces[highlight_faces])
                curve_edges = []
                for face in new_f:
                    curve_edges.extend(
                        [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]])
                curve_edges = np.array(curve_edges)
                ps_curve = ps.register_curve_network("curve", curve_v, curve_edges, color = highlight_color,
                                                    radius = highlight_radius)

            if cpos is None or clook is None:
                ps.reset_camera_to_home_view()
            else:
                ps.look_at_dir(cpos, clook, up)

            if debug == True:
                ps.show()
            ps.screenshot(f"{frame_path}/{name}_{count:03}.png")
            ps.remove_all_structures()
            count += 1
    if save_video == True:
        import glob
        from PIL import Image
        fp_in = f"{frame_path}/{name}_*.png"
        fp_out = f"{save_path}/{name}.gif"
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=200, loop=0, disposal=2)

# =================== Network Profiling =============================
def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad is None:
            print(f"Warning. Found null gradient: layer {n}")
            continue
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

# ==================== Memory Profiling =============================
from itertools import chain
from collections import deque
from sys import getsizeof, stderr

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
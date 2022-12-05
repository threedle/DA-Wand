# Generate developable dataset by running CSG with deformed polyhedrons  
from multiprocessing.sharedctypes import Value
from turtle import clear
from models.layers.meshing import Mesh
from models.layers.meshing.io import PolygonSoup
from models.layers.meshing.analysis import computeVertexAngle
import numpy as np
import os 
import torch 
import random 
from pathlib import Path
import polyscope as ps
import sys
from util.util import run_slim, get_ss
import re 
from util.util import getRotMat, fix_orientation
import dill as pickle 
from util.util import clear_directory
sys.setrecursionlimit(10000)

# Pipeline: union up to 5 deformed shapes (normalize => rotation + translation augs => wildmeshing) => fix orientations => 
#               save as objs => map labels to new faces using correspondence (check vertices for each face) + 
#                                 sample anchors from each label set in source folder

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
    # fneighbors = [] 
    # for _ in range(3):
    #     if he.twin.face.isBoundaryLoop():
    #         he = he.next 
    #         continue 
    #     fneighbors.append(he.twin.face.index)
    #     he = he.next 
    fneighbors = [f.index for f in list(topo.faces[face].adjacentFaces())]
    new_faces = list(set(fneighbors).difference(set(patchlist)))
    if len(new_faces) == 0:
        return True 
    for f in new_faces:
        _get_contig_inds(f, topo, patchlist)

def generate_union(meshes):
    # Test mesh boolean using PyMesh  
    import pymesh 
    pymeshes = [] 
    for mesh in meshes: 
        vs, fs, _ = mesh.export_soup() 
        pymeshes.append(pymesh.form_mesh(vs, fs))
    csgtree = pymesh.CSGTree({"union": [{"mesh": mesh} for mesh in pymeshes]})
    csgmesh = csgtree.mesh 
    source_inds = csgmesh.get_attribute("source")
    source_faces = csgmesh.get_attribute("source_face")
    assert len(csgmesh.faces) == len(source_inds) == len(source_faces), f"Error: PyMesh CSG produced faces/source mappings of unequal length. Faces: {len(csgmesh.faces)}. Inds: {len(source_inds)}. Face maps: {len(source_faces)}."       

    # Cleanup bad triangles 
    vs, fs, info = pymesh.remove_duplicated_vertices_raw(csgmesh.vertices, csgmesh.faces)
    print(f"# vertices removed: {info['num_vertex_merged']}")
    assert len(fs) == len(csgmesh.faces)
    vertices, faces, info = pymesh.collapse_short_edges_raw(vs, fs, rel_threshold=0.3, preserve_feature=True)
    print(f"# edges collaped: {info['num_edge_collapsed']}")
    sources = info['source_face_index'][:len(faces)]
    assert len(faces) == len(sources), f"Edge collapse error: new faces {len(faces)}, source array {len(sources)}"
    
    # Remap source faces 
    source_inds = source_inds[sources]
    source_faces = source_faces[sources]

    # Fix face orientations: use breadth first search, then use signed volume calculation to decide whether to flip 
    new_faces = fix_orientation(vertices, faces)
    csg_mesh = Mesh(vertices, new_faces)
    vs, fs, _ = csg_mesh.export_soup()
    assert len(fs) == len(source_inds) == len(source_faces), f"Error: new mesh faces/source mappings of unequal length. Faces: {len(fs)}. Inds: {len(source_inds)}. Face maps: {len(source_faces)}. New faces: {len(faces)}."       
    # Debugging: visualize mappings 
    # from models.layers.meshing.analysis import computeFaceNormals
    # computeFaceNormals(csg_mesh)
    # # Make sure face normals fit within mesh 
    # csg_mesh.facenormals *= 0.1
    # import polyscope as ps
    # ps.init()
    # ps_mesh = ps.register_surface_mesh("csgmesh", csg_mesh.vertices, new_faces, edge_width=1)
    # ps_mesh.add_vector_quantity("normals", csg_mesh.facenormals, defined_on='faces', enabled=True, vectortype='ambient')
    # ps_mesh.add_scalar_quantity("sources", source_inds, defined_on='faces', enabled=True)
    # # Show mappings to original meshes (split concatenated faces)
    # face_min = 0 
    # face_max = 0 
    # for i in range(len(meshes)):
    #     mesh = meshes[i] 
    #     vs, fs, _ = mesh.export_soup() 
    #     face_max += len(fs)
    #     source_mesh_inds = (source_faces < face_max) & (source_faces >= face_min)
    #     face_maps = source_faces[source_mesh_inds]
    #     face_maps -= np.min(face_maps)
    #     source_f_color = np.zeros(len(source_faces)) * -1 
    #     source_f_color[source_mesh_inds] = face_maps 
    #     ps_mesh.add_scalar_quantity(f"source faces {i}", source_f_color, defined_on='faces', enabled=True)
    #     og_psmesh = ps.register_surface_mesh(f"ogmesh{i}", vs, fs, edge_width=1)
    #     og_psmesh.add_scalar_quantity("enumfaces", np.arange(len(fs)), defined_on='faces', enabled=True)
    #     face_min += len(fs)
    # # Visualize unoriented mesh to double check pymesh 
    # pycsg_mesh = Mesh(vertices, faces)
    # pycsg_mesh.normalize()
    # computeFaceNormals(pycsg_mesh)
    # ps_mesh = ps.register_surface_mesh("pycsgmesh", pycsg_mesh.vertices, faces, edge_width=1)
    # ps_mesh.add_vector_quantity("normals", pycsg_mesh.facenormals * 0.1, defined_on='faces', enabled=True, vectortype='ambient')
    # ps_mesh.add_scalar_quantity("sources", source_inds, defined_on='faces', enabled=True)
    # ps.show() 
    # raise 
    csg_mesh.normalize() 
    return csg_mesh, source_inds, source_faces


if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--defdir', default="./datasets/meshgen/polyhedrons_def_highres", help='path to deformed simple shapes')
    parser.add_argument('--sourcedir', default="./datasets/meshgen/source_highres", help='path to source directory')
    parser.add_argument('--savedir', default="./datasets/meshgen/developable_highres", help='path to export')
    parser.add_argument('--n_sample', type=int, default=30)
    parser.add_argument('--max_union', type=int, default=5)
    parser.add_argument('--ratio', type=float, default=0.05, help='percent of faces to sample as anchors')
    parser.add_argument('--max_anchors', type=int, default=np.inf, help="max # anchors allowed per patch")
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--max_distort', type=float, default=0.05)
    parser.add_argument('--additional_primitives', nargs="+", default=[])
    parser.add_argument('--overwrite', action="store_true")

    args = parser.parse_args()
    defdir = args.defdir
    sourcedir = args.sourcedir
    savedir = args.savedir
    n_sample = args.n_sample 
    ratio = args.ratio 
    max_anchors = args.max_anchors 
    max_union = args.max_union
    train_split = args.train_split 
    
    # Regenerate 
    start_i = 0
    if os.path.exists(savedir) and args.overwrite==True:
        clear_directory(savedir)
    elif os.path.exists(savedir):
        # Find the largest index to start at 
        for mode in ['train', 'test']:
            for meshfile in os.listdir(os.path.join(savedir, mode)):
                if not meshfile.endswith(".obj"):
                    continue 
                re_search = re.search("[a-zA-Z]+(\d+)\.obj", meshfile)
                mesh_i = int(re_search.group(1))
                start_i = max(mesh_i, start_i)
        if start_i > 0: 
            start_i += 1
        print(f"Starting generation at mesh index {start_i}")
        
    Path(os.path.join(savedir, "test/anchors")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(savedir, "test/labels")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(savedir, "test/maps")).mkdir(exist_ok=True, parents=True)    
    Path(os.path.join(savedir, "train/anchors")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(savedir, "train/labels")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(savedir, "train/maps")).mkdir(exist_ok=True, parents=True)
    
    defmeshes = [file for file in os.listdir(defdir) if file.endswith(".obj")]

    n_union = np.random.choice(np.arange(1, max_union+1), n_sample) 
    for i in range(start_i, n_sample): 
        meshfiles = np.random.choice(defmeshes, n_union[i])
        
        # Set output dir based on train/test 
        mode = "test"
        if i <= train_split * n_sample:
            mode = "train"
        
        # Read in mesh data: meshes, labels
        soups = [PolygonSoup.from_obj(os.path.join(defdir, file)) for file in meshfiles]
        meshes = [Mesh(soup.vertices, soup.indices) for soup in soups]
        meshnames = [os.path.splitext(file)[0] for file in meshfiles]
        meshnames = [re.sub(r'\d+', "", name) for name in meshnames]
        
        print(f"Sample {i}: {meshnames}") 
        meshlabels = [] 
        for name in meshnames: 
            labels = [] 
            # Filter out labels for cylinders and cones 
            for file in os.listdir(os.path.join(sourcedir, "labels")):
                tmpname = re.sub(r'\d+\.npy', "", file)
                if name == tmpname:
                    labels.append(np.load(os.path.join(sourcedir, "labels", file)))
            meshlabels.append(labels)
        if len(meshlabels) == 0:
            print("Warning: no labels found for this mesh combination.")
            continue 
        # Normalize, rotate, and translate (to create new wild shape)
        for mesh in meshes: 
            mesh.normalize() 
            axis = np.random.uniform(size=3)
            theta = np.random.uniform(0, 2*np.pi)
            rotmat = getRotMat(axis, theta) 
            mesh.vertices = (rotmat @ mesh.vertices.transpose(1,0)).transpose(1,0)
            mesh.vertices += np.random.uniform(-0.3, 0.3, 3)
        
        # Generate union surface mesh 
        csgmesh, source_inds, source_faces = generate_union(meshes)
        # Generate mappings to original meshes (split concatenated faces)
        face_min = 0 
        face_max = 0
        mesh_maps = [] 
        csg_labels = [] 
        vertices, faces, _ = csgmesh.export_soup()
        assert len(source_faces) == len(faces)
        for meshi in np.unique(source_inds):
            mesh = meshes[int(meshi)] 
            vs, fs, _ = mesh.export_soup()
            face_max += len(fs)
            source_mesh_inds = (source_faces < face_max) & (source_faces >= face_min)
            face_maps = source_faces[source_mesh_inds]
            face_maps -= face_min
            face_maps = face_maps.astype(int)
            mesh_maps.append(face_maps)
            if len(face_maps) <= 0:
                continue 
            assert np.max(face_maps) < len(fs) and np.min(face_maps) >= 0, f"Error: Source face {np.max(face_maps)} larger than # mesh faces {len(fs)}" 
            
            # Debugging: might be something wrong with the CSG mapping 
            # import polyscope as ps 
            # ps.init() 
            # ps_og_mesh = ps.register_surface_mesh("og mesh", vs, fs, edge_width=1)
            # og_faces = np.zeros(len(fs))
            # og_faces[face_maps] = 1 
            # ps_og_mesh.add_scalar_quantity("mapped faces", og_faces, defined_on="faces", enabled=True)
            # ps_csgmesh = ps.register_surface_mesh("csg mesh", vertices, faces, edge_width=1)
            # csg_faces = np.zeros(len(faces))
            # csg_faces[source_mesh_inds] = 1
            # ps_csgmesh.add_scalar_quantity(f"mapped faces", csg_faces, defined_on="faces", enabled=True)
            # ps.show()
            
            labels = meshlabels[int(meshi)]
            for labeli in range(len(labels)): 
                labelset = labels[labeli]
                assert len(labelset) == len(fs), f"Error: labelset length {len(labelset)} for mesh with # faces {len(fs)}"
                tmplabels = np.zeros(len(faces), dtype=int)
                tmplabels[source_mesh_inds] = labelset[face_maps]

                if np.sum(tmplabels) < 10: 
                    continue 
                
                patches = contiguous_patches(faces[np.where(tmplabels == 1)[0]])
                if patches is None: 
                    continue 
                
                # Remove patches for which isolated faces were found 
                for patch in patches: 
                    patchlabels = np.zeros(len(faces), dtype=int)
                    patchlabels[np.where(tmplabels == 1)[0][patch]] = 1
                    
                    # Need at least 1 anchor to be sampled
                    if np.sum(patchlabels) * ratio < 1: 
                        continue 
                    
                    if args.max_distort > 0:
                        selection = np.where(patchlabels == 1)[0]
                        subvs, subfs = csgmesh.export_submesh(selection)
                        submesh = Mesh(subvs, subfs)
                        
                        uvmap, slim_energy, did_cut = run_slim(submesh, cut=True)
                        ss = get_ss(submesh, uvmap)                        
                        distortion = np.mean(np.maximum(ss[:,0], 1/ss[:,1]))
                        
                        if not (np.isfinite(distortion) and distortion <= 1 + args.max_distort and distortion >= 1 - args.max_distort):
                            print(f"{meshnames}: label {labeli} distortion {distortion:0.5f} over distortion threshold {args.max_distort}.") 
                            print(f"SLIM energy: {slim_energy:0.5f}")
                            continue                        
                    
                    csg_labels.append(patchlabels)
            face_min += len(fs)

        if len(csg_labels) == 0:
            print(f"Warning: no valid anchors for mesh {i}.")
            continue 
        
        # Sample positive anchors 
        anchors = [np.random.choice(np.where(tmplabel == 1)[0], min(int(np.sum(tmplabel)*ratio), max_anchors), replace=False) for tmplabel in csg_labels]
        assert np.all([len(anchor) > 0 for anchor in anchors])
        
        # Don't let anchors be on small mesh elements (clustered near intersections ergo patch boundaries)
        total_area = csgmesh.totalArea()
        avg_area = total_area/len(csgmesh.topology.faces)
        min_area = 0.3 * avg_area 
        for anchori in range(len(anchors)):
            anchorset = anchors[anchori] 
            anchors[anchori] = [anchor for anchor in anchorset if csgmesh.area(csgmesh.topology.faces[anchor]) >= min_area]
        
        if np.all([len(anchor) == 0 for anchor in anchors]):
            print(f"Warning: no valid anchors for mesh {i}.")
            continue 
        
        # Save labels 
        labelcount = 0 
        for labeli in range(len(anchors)):
            for _ in range(len(anchors[labeli])):
                np.save(os.path.join(savedir, mode, "labels", f"sample{i}_{labelcount}.npy"), csg_labels[labeli])
                labelcount += 1
        # Convert to nested list 
        anchors_list = [] 
        for anchor in anchors:
            anchors_list.extend([[f] for f in anchor])
        print(f"Anchors for mesh {i}: {len(anchors_list)}")
        with open(os.path.join(savedir, mode, "anchors", f"sample{i}.pkl"), 'wb') as f:
            pickle.dump(anchors_list, f)
        # Save mesh maps as tuple with source indices 
        with open(os.path.join(savedir, mode, "maps", f"sample{i}.pkl"), 'wb') as f:
            pickle.dump((meshnames, source_inds, mesh_maps), f)
    
        # Export mesh 
        csgmesh.export_dir = os.path.join(savedir, mode)
        csgmesh.meshname = f"sample{i}"
        csgmesh.export_obj()
        
    # Additional manual individual shapes 
    additional_primitives = args.additional_primitives 
    if len(additional_primitives) > 0:
        print(f"Adding additional individual primitives to dataset: {additional_primitives}")
        start_index = n_sample 
        n_sample = len(additional_primitives)
        for i in range(len(additional_primitives)):
            mode = "test"
            if i <= train_split * len(additional_primitives):
                mode = "train"
            
            meshname = additional_primitives[i]
            meshfile = f"{meshname}.obj"
            
            soup = PolygonSoup.from_obj(os.path.join(defdir, meshfile))
            mesh = Mesh(soup.vertices, soup.indices)
            print(meshfile)
            
            re_res = re.search(f"^([a-zA-Z]+)(\d+)?\.obj", meshfile)
            source_meshname = re_res.group(1) 
            
            labeldir = os.path.join(sourcedir, "labels")
            labelcount = 0
            totanchors = [] 
            for labelfile in os.listdir(labeldir):
                re_res = re.search(f"^{source_meshname}\d+\.npy", labelfile)
                if re_res:
                    labels = np.load(os.path.join(labeldir, labelfile))
                    n_sample = min(int(ratio * len(np.where(labels==1)[0])), max_anchors)
                    
                    anchors = np.random.choice(np.where(labels == 1)[0], n_sample, replace=False)
                    totanchors.extend([[anchor] for anchor in anchors])
                    
                    for _ in anchors:
                        np.save(os.path.join(savedir, mode, "labels", f"sample{start_index + i}_{labelcount}.npy"), labels)
                        labelcount += 1

                    # Debugging
                    # import polyscope as ps 
                    # ps.remove_all_structures()
                    # ps.init() 
                    # ps_mesh = ps.register_surface_mesh("mesh", soup.vertices, soup.indices, edge_width=1) 
                    # ps_mesh.add_scalar_quantity("gt", labels, defined_on='faces', enabled=True)
                    # ps_curve = ps.register_curve_network(f"anchors", np.mean(soup.vertices[soup.indices[anchors]], axis=1),
                    #                                         np.array([[i,i] for i in range(len(anchors))]), color=[1,0,0], enabled=True)
                    # ps.show() 
                    
            with open(os.path.join(savedir, mode, "anchors", f"sample{start_index + i}.pkl"), 'wb') as f:
                pickle.dump(totanchors, f)

            # Export mesh 
            mesh.export_dir = os.path.join(savedir, mode)
            mesh.meshname = f"sample{start_index + i}"
            mesh.export_obj()
            
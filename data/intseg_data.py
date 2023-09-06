import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad, getRotMat, compute_hks
import numpy as np
from models.layers.meshing import Mesh
from models.layers.meshing.analysis import computeDihedrals, computeFaceAreas, computeEdgeRatios, computeOppositeAngles, computeVertexNormals
from models.layers.meshing.analysis import computeEdgeNeighborMatrix, computeFaceNeighborMatrix, computeFaceNormals, computeHKS
from models.layers.meshing.io import PolygonSoup
from pathlib import Path
import dill as pickle
from util.diffusion_net.geometry import get_operators

def compute_and_cache(mesh, cachepath=None):
    # Precompute all the standard edge features
    computeDihedrals(mesh)
    computeEdgeRatios(mesh)
    computeOppositeAngles(mesh)

    # Compute edge matrix
    computeEdgeNeighborMatrix(mesh)
    computeFaceNeighborMatrix(mesh)

    # Compute other relevant features
    computeFaceAreas(mesh)
    computeFaceNormals(mesh)

    mesh.edge_to_f = np.array([[e.halfedge.face.index, e.halfedge.twin.face.index] for key, e in sorted(mesh.topology.edges.items())])
    mesh.edgenormals = np.mean(mesh.facenormals[mesh.edge_to_f], axis=1)
    mesh.edgenormals /= np.maximum(np.linalg.norm(mesh.edgenormals, axis=1, keepdims=True), 1e-5)
    assert mesh.edgenormals.shape == (len(mesh.topology.edges.keys()), 3)

    if cachepath is not None:
        np.savez_compressed(cachepath, facenormals=mesh.facenormals, dihedrals=mesh.dihedrals,
                                        edgemat=mesh.edgemat, facemat=mesh.facemat,
                                        symmetricoppositeangles=mesh.symmetricoppositeangles,
                                        edgeratios=mesh.edgeratios, edge_to_f=mesh.edge_to_f,
                                        edgenormals=mesh.edgenormals,
                                        vertices=mesh.vertices, faces=mesh.faces,
                                        halfedge_data=mesh.topology.export_halfedge_serialization(),
                                        fareas = mesh.fareas)
    return mesh

def augment(mesh, opt):
    if opt.slideaug > 0:
        # Slide vertex positions along random incident edge
        v_ids = np.random.choice(list(mesh.topology.vertices.keys()), int(opt.slideaug * len(mesh.topology.vertices)), replace=False)
        interp = np.random.uniform(0.2, 0.5, (int(opt.slideaug * len(mesh.topology.vertices)),1))
        vn_ids = []
        for v_id in v_ids:
            vn_ids.append(np.random.choice([v.index for v in mesh.topology.vertices[v_id].adjacentVertices()]))
        mesh.vertices[v_ids] = mesh.vertices[v_ids] + interp * (mesh.vertices[vn_ids] - mesh.vertices[v_ids])

    if opt.vaug == True:
        from models.layers.meshing.edit import LaplacianSmoothing
        # Displacement: random 3D translation + smoothing filter w/ random weights for each vertex
        displ = np.random.normal(0, 0.05, size=(len(mesh.vertices), 3))

        # Smoothing filter
        mesh.vertices = mesh.vertices + displ
        weights = np.random.uniform(0.1, 0.9, len(mesh.vertices))
        LaplacianSmoothing(mesh, weights=weights).apply()

    if opt.vnormaug == True:
        from models.layers.meshing.edit import LaplacianSmoothing
        # Displacement: displace vertices randomly along their normals
        displ = np.random.normal(0, 0.05, size=(len(mesh.vertices), 1))
        if not hasattr(mesh, "vertexnormals"):
            computeVertexNormals(mesh)
        displ = mesh.vertexnormals * displ

        # Smoothing filter
        mesh.vertices = mesh.vertices + displ
        weights = np.random.uniform(0.1, 0.9, len(mesh.vertices))
        LaplacianSmoothing(mesh, weights=weights).apply()

    if opt.rotaug == True:
        # Also apply random rotation about random axis
        axis = np.random.uniform(size=3)
        theta = np.random.uniform(0, 2*np.pi)
        rotmat = getRotMat(axis, theta)
        mesh.vertices = (rotmat @ mesh.vertices.transpose(1,0)).transpose(1,0)

    if opt.flipaug > 0:
        from models.layers.meshing.edit import EdgeFlip
        # Random edge flips
        e_ids = np.random.choice(list(mesh.topology.edges.keys()), int(opt.flipaug * len(mesh.topology.edges)), replace=False)
        for e_id in e_ids:
            EdgeFlip(mesh, e_id).apply()

def condition_and_cache(mesh, opt, cachepath=None, evals=None, evecs=None):
    extrinsic_features = opt.extrinsic_features
    extrinsics = []
    cachedict = {}

    for extr in extrinsic_features:
        if extr == "onehot":
            condition_fe = torch.zeros(1, len(mesh.topology.edges))
            anchor_edges = [e.index for f in mesh.anchor_fs for e in mesh.topology.faces[f].adjacentEdges()]
            condition_fe[:,anchor_edges] = 1
        elif extr == "geodesic":
            # Geodesics
            from igl import heat_geodesic, edge_lengths
            vertices, faces, _ = mesh.export_soup()
            faces = np.array(faces, dtype=int)
            t = np.mean(edge_lengths(vertices, faces)) ** 2
            geodesics = heat_geodesic(vertices, faces, t, np.array([v.index for f in mesh.anchor_fs for v in mesh.topology.faces[f].adjacentVertices()]))
            # No negative geodesic values
            geodesics = np.maximum(geodesics, 0)
            # Normalize geodesics between 0 and 1
            geodesics /= np.max(geodesics)
            mesh.anchor_vertex_geodesics = geodesics.astype(float)
            edge_geodesics = np.mean(geodesics[[list(v.index for v in edge.two_vertices()) for \
                                                    key, edge in sorted(mesh.topology.edges.items())]], axis=1)
            mesh.anchor_edge_geodesics = edge_geodesics
            condition_fe = mesh.anchor_edge_geodesics

            cachedict['anchor_vertex_geodesics'] = mesh.anchor_vertex_geodesics
            cachedict['anchor_edge_geodesics'] = mesh.anchor_edge_geodesics

        elif extr == "normal":
            condition_fe = mesh.edgenormals.transpose(1,0)
        elif extr == "anchornormal":
            # Also condition using anchor faces
            condition_fe = np.mean(mesh.facenormals[mesh.anchor_fs], axis=0)
            condition_fe = np.column_stack([condition_fe] * len(mesh.topology.edges))
        elif extr == "position":
            # Condition on anchor xyz
            vertices, faces, edges = mesh.export_soup()
            condition_fe = np.mean(mesh.vertices[faces[mesh.anchor_fs]].reshape(len(mesh.anchor_fs) * 3, 3), axis=0)
            condition_fe = np.column_stack([condition_fe] * len(mesh.topology.edges))
        elif extr == "dihedral":
            # Condition on dihedral b/w anchor and edge
            # Just average across the two face dihedrals for all anchor faces
            e_to_f = np.array([[e.halfedge.face.index, e.halfedge.twin.face.index] for e in mesh.topology.edges.values()])
            edge_fnormals = mesh.facenormals[e_to_f] # E x 2 x 3
            anchor_fnormals = mesh.facenormals[np.array(mesh.anchor_fs)] # A x 3

            # Broadcast dot product across all anchors
            cosTheta = (edge_fnormals * anchor_fnormals.reshape(anchor_fnormals.shape[0], 1, 1, anchor_fnormals.shape[1])).sum(axis=-1).clip(-1,1)
            anchor_dihedrals = np.pi - np.arccos(cosTheta)

            # Average across anchors and neighbor faces
            condition_fe = np.mean(anchor_dihedrals.transpose(0,2,1).reshape(-1, len(edge_fnormals)), axis=0, keepdims=True)

        elif extr == "hks":
            if not hasattr(mesh, "hks"):
                # Instead of using options, use np logspace default
                scales = opt.hks_t
                if scales is None:
                    scales = torch.logspace(-2, 0, 16)
                else:
                    scales = torch.tensor(scales)
                mesh.hks = compute_hks(evals, evecs, scales).detach().numpy()

            # Vertex to edge
            vtoe = [list(v.index for v in edge.two_vertices()) for key, edge in sorted(mesh.topology.edges.items())]
            ehks = np.mean(mesh.hks[vtoe], axis=1)
            condition_fe = ehks.transpose(1,0)

            cachedict['hks'] = mesh.hks
            cachedict['ehks'] = ehks
        else:
            raise ValueError(f"No conditioning feature: {extr} impelmented")
        extrinsics.append(condition_fe)
    if len(mesh.edgefeatures) > 0:
        mesh.edgefeatures = np.vstack([mesh.edgefeatures] + extrinsics)
    else:
        mesh.edgefeatures = np.vstack(extrinsics)
    assert len(mesh.edgefeatures) > 0, f"Must have either mesh intrinsic edge features or extrinsics with preconditioning"

    cachedict['edgefeatures'] = mesh.edgefeatures
    if cachepath is not None:
        np.savez_compressed(cachepath, **cachedict)

class IntSegData(BaseDataset):
    def __init__(self, opt=None):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.save_dir = os.path.join(opt.export_save_path, opt.name, opt.phase)
        Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        self.weight_init = opt.weight_init
        if opt.is_train == True:
            self.paths, self.cachepaths, self.anchorcachepaths, self.operatorpaths, self.labelpaths, self.anchor_fs, self.anchor_fs_labels, self.augs = self.make_dataset(self.dir, self.opt.max_dataset_size)
            if opt.clip == True:
                self.texturedir = opt.texturedir
        else:
            self.dir = opt.test_dir
            self.test_dir = opt.test_dir
            self.paths, self.cachepaths, self.anchorcachepaths, self.operatorpaths, self.labelpaths, self.anchor_fs, self.anchor_fs_labels, self.augs = self.make_dataset(self.test_dir, self.opt.max_dataset_size)
        self.texturedir = None
        self.size = len(self.paths)
        # Maintain Cvxpy layers here
        self.cvxlayers = [None] * len(self.paths)

        # Pre-load all of your data instead of reading from npz (still need to run augmentations online though)
        self.data = {}
        self.precomputed_labels = None

        # NOTE: This will run through our full dataset before training starts
        self.get_mean_std()

        # NOTE: this resets the input channels automatically to the features we compute
        opt.input_nc = self.ninput_channels
        opt.ninput_edges = self.ninput_edges

    def update_anchor(self, anchor_fs):
        if anchor_fs:
            self.anchor_fs = [anchor_fs]
        # Need to regenerate cache and normalize features
        self.opt.overwriteanchorcache = True
        # self.opt.overwritecache = True
        self.opt.overwritemeanstd = True
        self.get_mean_std()

    def __getitem__(self, index):
        path = self.paths[index]
        # NOTE: will be aug specific if test
        cachepath = self.cachepaths[index]
        anchorcachepath = self.anchorcachepaths[index]
        operatorpath = self.operatorpaths[index]
        Path(operatorpath).mkdir(exist_ok=True, parents=True)

        labelpath = self.labelpaths[index]
        meshname = os.path.splitext(os.path.basename(path))[0]
        mesh_i = os.path.splitext(os.path.basename(anchorcachepath))[0] # Maintain same indexing as cache/labels
        aug = self.augs[index]

        if index in self.data.keys() and self.opt.overwritecache == False:
            meta = self.data[index]
            if meta['edge_features'].shape[1] < self.opt.ninput_edges:
                meta['edge_features'] = pad(meta['edge_features'], self.opt.ninput_edges) # F x E

            # NOTE: Currently mean/std NOT being computed (training input NOT normalized)
            # Include mean and std deviation info b/c will need to renormalize after augmenting
            meta['mean'] = self.mean
            meta['std'] = self.std

            # Replace labels with precomputed if set
            if self.precomputed_labels is not None:
                labels = self.precomputed_labels[index]
                meta['label'] = labels

            return meta
        else:
            labels = None
            if labelpath and os.path.exists(labelpath):
                labels = np.load(labelpath)

            # Replace labels with precomputed if set
            if self.precomputed_labels is not None:
                labels = self.precomputed_labels[index]

            anchor_fs = self.anchor_fs[index]

            # Recompute mesh values only if cache doesn't exist
            cacheexists = False
            if os.path.exists(cachepath) and self.opt.overwritecache == False:
                try:
                    mesh = Mesh(meshdata=dict(np.load(cachepath, allow_pickle=True)))
                    cacheexists = True

                    # Check if all necessary attributes exists
                    attributes = ["facenormals", "edgemat", "facemat", "dihedrals", "edgeratios", "edgenormals",
                                  "symmetricoppositeangles", "edge_to_f","vertices", "faces",
                                    "halfedge_data", "fareas"]
                    for attr in attributes:
                        if not hasattr(mesh, attr):
                            cacheexists = False
                            break
                except Exception as e:
                    print(e)
            if not cacheexists:
                soup = PolygonSoup.from_obj(path)
                meshname = os.path.splitext(os.path.basename(path))[0]

                if self.opt.shuffle_topo == True:
                    p = np.random.permutation(len(soup.vertices))
                    p_map = np.zeros(len(soup.vertices))
                    p_map[p] = np.arange(len(soup.vertices))

                    soup.indices = p_map[soup.indices.astype(int)]
                    soup.vertices = soup.vertices[p]

                mesh = Mesh(soup.vertices, soup.indices, meshname=meshname)
                mesh.normalize() # NOTE: normalize within unit sphere so can set view directions using azim/elev

                # Cache augmentation if set
                if aug is not None and self.opt.is_train == False:
                    augment(mesh, self.opt)

                mesh = compute_and_cache(mesh, cachepath)

            # Compute anchor extrinsics
            mesh.computeEdgeFeatures(intrinsics=self.opt.edgefeatures)

            mesh.no = mesh_i
            if aug:
                mesh.no = f"{mesh_i}_aug{aug}"

            # Load operator values
            # NOTE: The below function automatically CACHES
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(np.ascontiguousarray(mesh.faces)).long()

            frames, mass, L, evals, evecs, gradX, gradY = \
                get_operators(vertices, faces, meshname, op_cache_dir=operatorpath, overwrite_cache=self.opt.overwriteopcache)

            mesh.anchor_fs = anchor_fs
            # NOTE: NOT every value in anchor cache should be a feature
            if self.opt.extrinsic_features is not None and self.opt.extrinsic_condition_placement == "pre":
                # NOTE: The anchorcache should have the edgefeatures saved as well
                anchorcache_exists = False
                if os.path.exists(anchorcachepath) and not self.opt.overwriteanchorcache:
                    anchorfeatures = dict(np.load(anchorcachepath, allow_pickle=True))

                    # We only care about vertexfeatures existing here
                    if "edgefeatures" in anchorfeatures.keys():
                        anchorcache_exists = True

                    for key, val in anchorfeatures.items():
                        setattr(mesh, key, val)
                if not anchorcache_exists:
                    condition_and_cache(mesh, self.opt, anchorcachepath, evals, evecs)

            # Sometimes edge features are nan (b/c of heat geodesic or other bad elements)
            if not np.all(np.isfinite(mesh.edgefeatures)):
                print(f"Warning: non-finite mesh features found for mesh {mesh.no}. Skipping...")
                print(np.where(~np.isfinite(mesh.edgefeatures))[0][:10])
                return None

            meta = {}
            # meta['mesh'] = mesh
            meta['meshdata'] = dict(np.load(cachepath, allow_pickle=True))
            meta['anchordata'] = dict(np.load(anchorcachepath, allow_pickle=True))
            meta['label'] = labels

            meta['evals'] = evals
            meta['evecs'] = evecs

            edgefeatures = mesh.edgefeatures
            if self.opt.ninput_edges:
                edgefeatures = pad(edgefeatures, self.opt.ninput_edges)

            meta['edge_features'] = edgefeatures
            meta['file'] = os.path.basename(path)
            meta['texture_path'] = self.texturedir
            meta['anchor_fs'] = anchor_fs
            meta['mean'] = self.mean
            meta['std'] = self.std

            meta['aug'] = aug
            meta['no'] = mesh.no
            meta['export_dir'] = os.path.join(self.save_dir, f"{mesh_i}_pools")
            # meta['cvxlayer'] = layer

            # Save meta dictionary in larger preloaded dictionary
            self.data[index] = meta

            return meta

    def init_data(self, mode):
        # Basically just need to initialize labels to the relevant values
        precomputed_labels = []
        for i, data in enumerate(self):
            # NOTE: Building from serialization means arbitrary edge -> HE mapping
            mesh = Mesh(meshdata=data['meshdata'])
            tmplabs = np.zeros(len(mesh.faces))
            if mode == "anchor":
                tmplabs[data['anchor_fs']] = 1
                precomputed_labels.append(tmplabs)
            if mode == "neighbors":
                select_set = set(data['anchor_fs'])
                for _ in range(self.opt.init_neighbor_radius):
                    neighbors = [n.index for f in select_set for n in mesh.topology.faces[f].adjacentFaces()]
                    select_set.update(neighbors)
                tmplabs[list(select_set)] = 1
                precomputed_labels.append(tmplabs)
        self.precomputed_labels = precomputed_labels
        assert len(self.precomputed_labels) == len(self), f"Initialized labels {len(self.precomputed_labels)} should be same length as dataset {len(self)}."

    def __len__(self):
        return self.size

    def make_dataset(self, path, max_size):
        paths = []
        cachepaths = []
        anchorcachepaths = []
        operatorpaths = []
        labelpaths = []
        augs = []
        anchor_fs = []
        anchor_fs_labels = []
        stop = False
        assert os.path.isdir(path), '%s is not a valid directory' % path
        # Make cache path
        Path(os.path.join(path, self.opt.cachefolder)).mkdir(exist_ok=True, parents=True)
        Path(os.path.join(path, self.opt.anchorcachefolder)).mkdir(exist_ok=True, parents=True)

        for fname in os.listdir(path):
            if is_mesh_file(fname):
                meshpath = os.path.join(path, fname)
                meshname = os.path.splitext(fname)[0]

                # If interactive, then automatically append
                if self.opt.interactive:
                    if len(self.opt.subset) > 0:
                        if f"{meshname}_0" not in self.opt.subset:
                            continue
                    anchor_fs.append([0])
                    paths.append(meshpath)
                    cachepaths.append(os.path.join(path, self.opt.cachefolder, f"{meshname}.npz"))
                    anchorcachepaths.append(os.path.join(path, self.opt.anchorcachefolder, f"{meshname}_0.npz"))
                    operatorpaths.append(os.path.join(path, self.opt.operatorcachefolder, f"{meshname}"))
                    labelpaths.append(None)
                    augs.append(None)

                    # Default: inf
                    if len(paths) >= max_size:
                        stop = True
                        break

                if not os.path.exists(os.path.join(path, "anchors", f"{meshname}.pkl")):
                    print(f"Warning: no anchors exist for {meshname}.")
                    continue

                with open(os.path.join(path, "anchors", f"{meshname}.pkl"), 'rb') as f:
                    anchors = pickle.load(f)

                for i in range(len(anchors)):
                    if len(self.opt.subset) > 0:
                        if f"{meshname}_{i}" not in self.opt.subset:
                            continue
                    if len(self.opt.exclude) > 0:
                        if f"{meshname}_{i}" in self.opt.exclude:
                            continue

                    anchor = anchors[i] # NOTE: This should be LIST
                    anchor_fs.append(anchor)
                    paths.append(meshpath)
                    cachepaths.append(os.path.join(path, self.opt.cachefolder, f"{meshname}.npz"))
                    anchorcachepaths.append(os.path.join(path, self.opt.anchorcachefolder, f"{meshname}_{i}.npz"))
                    operatorpaths.append(os.path.join(path, self.opt.operatorcachefolder, f"{meshname}"))
                    if self.opt.supervised and os.path.exists(os.path.join(path, "labels", f"{meshname}_{i}.npy")):
                        labelpaths.append(os.path.join(path, "labels", f"{meshname}_{i}.npy"))
                    else:
                        labelpaths.append(None)
                    augs.append(None)

                    # If training + augmentation values: then duplicate and add augmentations
                    # If testing + augmentation: then save new cache values for each augmentation one-off
                    if self.opt.is_train == True or self.opt.testaug == True:
                        for aug in range(self.opt.num_aug):
                            paths.append(meshpath)
                            augs.append(aug)
                            anchor_fs.append(anchor)
                            if self.opt.is_train == False:
                                cachepaths.append(os.path.join(path, self.opt.cachefolder, f"{meshname}_aug{aug}.npz"))
                                anchorcachepaths.append(os.path.join(path, self.opt.anchorcachefolder, f"{meshname}_{i}_aug{aug}.npz"))
                                operatorpaths.append(os.path.join(path, self.opt.operatorcachefolder, f"{meshname}_aug{aug}"))
                            else:
                                cachepaths.append(os.path.join(path, self.opt.cachefolder, f"{meshname}.npz"))
                                anchorcachepaths.append(os.path.join(path, self.opt.anchorcachefolder, f"{meshname}_{i}.npz"))
                                operatorpaths.append(os.path.join(path, self.opt.operatorcachefolder, f"{meshname}"))

                            if self.opt.supervised and os.path.exists(os.path.join(path, "labels", f"{meshname}_{i}.npy")):
                                labelpaths.append(os.path.join(path, "labels", f"{meshname}_{i}.npy"))
                            else:
                                labelpaths.append(None)
                            # Default: inf
                            if len(paths) >= max_size:
                                stop = True
                                break
                    # Default: inf
                    if len(paths) >= max_size:
                        stop = True
                        break
            if stop == True:
                break
        return paths, cachepaths, anchorcachepaths, operatorpaths, labelpaths, anchor_fs, anchor_fs_labels, augs

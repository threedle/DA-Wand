# InteractiveSegmentation: Standardized set of tests for parameterization loss functions
import torch
from util.util import dclamp

# ==================== Energies ===============================
def symmetricdirichlet(vertices, faces, param, face_area, param_area, return_face_energy=True):
    # Inverse equiareal term
    inv_area_distort = 1 + face_area ** 2 / param_area ** 2
    paramtris = param[faces]
    ogtris = vertices[faces]

    dirichlet_left = (torch.norm(paramtris[:, 2, :] - paramtris[:, 0, :], dim=1) ** 2 * torch.norm(
        ogtris[:, 1, :] - ogtris[:, 0, :], dim=1) ** 2 +
                      torch.norm(paramtris[:, 1, :] - paramtris[:, 0, :], dim=1) ** 2 * torch.norm(
                ogtris[:, 2, :] - ogtris[:, 0, :], dim=1) ** 2) / (4 * face_area)
    dirichlet_right = (torch.sum((paramtris[:, 2, :] - paramtris[:, 0, :]) * (paramtris[:, 1, :] - paramtris[:, 0, :]),
                                 dim=-1) *
                       torch.sum((ogtris[:, 2, :] - ogtris[:, 0, :]) * (ogtris[:, 1, :] - ogtris[:, 0, :]), dim=-1)) / (
                                  2 * face_area)
    face_energy = inv_area_distort * (dirichlet_left - dirichlet_right)

    if return_face_energy == False:
        return torch.sum(face_energy)
    return face_energy

# Smoothness term for graphcuts: basically sum pairwise energies as function of label pairs + some kind of geometry weighting
# pairwise: maximum cost of diverging neighbor labels (i.e. 0 next to 1). this value is multiplied by the difference in the soft probs
def gcsmoothness(preds, mesh, feature='dihedral', pairwise=1):
    face_adj = torch.tensor([[edge.halfedge.face.index, edge.halfedge.twin.face.index] for key, edge in sorted(mesh.topology.edges.items())]).long().to(preds.device)

    if feature == 'dihedral':
        if not hasattr(mesh, "dihedrals"):
            from models.layers.meshing.analysis import computeDihedrals
            computeDihedrals(mesh)

        dihedrals = torch.clip(torch.pi - torch.from_numpy(mesh.dihedrals).to(preds.device), 0, torch.pi).squeeze()
        smoothness = -torch.log(dihedrals/torch.pi + 1e-3)
    else:
        raise NotImplementedError(feature)

    adj_preds = preds[face_adj]

    # NOTE: we include the 1 - torch.max() term in order to encourage patch GROWING
    smoothness_cost = torch.mean(smoothness * pairwise * ((torch.abs(adj_preds[:,1] - adj_preds[:,0]))))
<<<<<<< HEAD

    return smoothness_cost
=======
>>>>>>> 75306aed0bd8c8bbd39d40ac1e2f5fe09e151de0

    return smoothness_cost

def arap(vertices, faces, param, return_face_energy=True, paramtris=None, renormalize=False,
         face_weights=None, normalize_filter=0, device=torch.device("cpu"), verbose=False, timeit=False, **kwargs):
    from util.util import get_local_tris
    local_tris = get_local_tris(vertices, faces, device=device)

    if paramtris is None:
        paramtris = param[faces]

    if timeit == True:
        import time
        t0 = time.time()

    # Squared norms of difference in edge vectors multiplied by cotangent of opposite angle
    # NOTE: LSCM applies some constant scaling factor -- can we renormalize to get back original edge lengths?
    try:
        local_tris = local_tris.contiguous()
    except Exception as e:
        print(e)

    e1 = local_tris[:, 2, :] - local_tris[:, 0, :]
    e2 = local_tris[:, 1, :] - local_tris[:, 0, :]
    e3 = local_tris[:, 2, :] - local_tris[:, 1, :]
    e1_p = paramtris[:, 2, :] - paramtris[:, 0, :]
    e2_p = paramtris[:, 1, :] - paramtris[:, 0, :]
    e3_p = paramtris[:, 2, :] - paramtris[:, 1, :]

    # NOTE: sometimes denominator will be 0 i.e. area of triangle is 0 -> cotangent in this case is infty, default to 1e5
    cot1 = torch.abs(torch.sum(e2 * e3, dim=1) / torch.clamp((e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]), min=1e-5))
    cot2 = torch.abs(torch.sum(e1 * e3, dim=1) / torch.clamp((e1[:, 0] * e3[:, 1] - e1[:, 1] * e3[:, 0]), min=1e-5))
    cot3 = torch.abs(torch.sum(e2 * e1, dim=1) / torch.clamp(e2[:, 0] * e1[:, 1] - e2[:, 1] * e1[:, 0], min=1e-5))

    # Debug
    if torch.any(~torch.isfinite(paramtris)):
        print(f"Non-finite parameterization result found.")
        print(f"{torch.sum(~torch.isfinite(param))} non-finite out of {len(param.flatten())} param. elements")
        return None

    # Threshold param tris as well
    e1_p = torch.maximum(torch.minimum(e1_p, torch.tensor(1e5)), torch.tensor(-1e5))
    e2_p = torch.maximum(torch.minimum(e2_p, torch.tensor(1e5)), torch.tensor(-1e5))
    e3_p = torch.maximum(torch.minimum(e3_p, torch.tensor(1e5)), torch.tensor(-1e5))

    # Compute all edge rotations
    cot_full = torch.stack([cot1, cot2, cot3]).reshape(3, len(cot1), 1, 1)
    e_full = torch.stack([e1, e2, e3])
    e_p_full = torch.stack([e1_p, e2_p, e3_p])
    crosscov = torch.sum(cot_full * torch.matmul(e_full.unsqueeze(3), e_p_full.unsqueeze(2)), dim=0)
    crosscov = crosscov.reshape(crosscov.shape[0], 4) # F x 4
<<<<<<< HEAD
=======

    # tdenom = torch.clamp(crosscov[:,0]**2 + crosscov[:,1]**2 - crosscov[:,2]**2 - crosscov[:,3]**2, min=1e-5)
    # pdenom = torch.clamp(crosscov[:,0]**2 - crosscov[:,1]**2 + crosscov[:,2]**2 - crosscov[:,3]**2, min=1e-5)
    # theta = torch.atan2(2 * crosscov[:,0] * crosscov[:,2] + 2 * crosscov[:,1] * crosscov[:,3], tdenom)/2
    # phi = torch.atan2(2 * crosscov[:,0] * crosscov[:,1] + 2 * crosscov[:,2] * crosscov[:,3], pdenom)/2

    # cphi = torch.cos(phi)
    # sphi = torch.sin(phi)
    # ctheta = torch.cos(theta)
    # stheta = torch.sin(theta)
    # s1 = () * + () *
    # s2 = () * + () *

    # U = torch.stack([torch.stack([ctheta, -stheta], dim=1), torch.stack([stheta, ctheta], dim=1)], dim=2)
    # V = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)
>>>>>>> 75306aed0bd8c8bbd39d40ac1e2f5fe09e151de0

    E = (crosscov[:,0] + crosscov[:,3])/2
    F = (crosscov[:,0] - crosscov[:,3])/2
    G = (crosscov[:,2] + crosscov[:,1])/2
    H = (crosscov[:,2] - crosscov[:,1])/2

    Q = torch.sqrt(E ** 2 + H ** 2)
    R = torch.sqrt(F ** 2 + G ** 2)

    S1 = Q + R
    S2 = Q - R
    a1 = torch.atan2(G, F)
    a2 = torch.atan2(H, E)
    theta = (a2 - a1) / 2 # F
    phi = (a2 + a1) / 2 # F

    # F x 2 x 2
    # NOTE: This is U^T
    U = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1), torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=2)

    # F x 2 x 2
    # NOTE: This is V
    V = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1), torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)], dim=2)

    R = torch.matmul(V, U).to(device) # F x 2 x 2
<<<<<<< HEAD
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        U[baddet, 1, :] *= -1
=======

    ## NOTE: Sanity check the SVD
    S = torch.stack([torch.diag(torch.tensor([S1[i], S2[i]])) for i in range(len(S1))]).to(S1.device)
    checkcov = U.transpose(2,1) @ S @ V.transpose(2,1)
    torch.testing.assert_close(crosscov.reshape(-1, 2, 2), checkcov)

    # Sometimes rotation is opposite orientation: just check with determinant and flip
    # NOTE: Can flip sign of det by flipping sign of last column of V
    baddet = torch.where(torch.det(R) <= 0)[0]
    if len(baddet) > 0:
        print(f"ARAP warning: found {len(baddet)} flipped rotations.")
        V[baddet, :, 1] *= -1
>>>>>>> 75306aed0bd8c8bbd39d40ac1e2f5fe09e151de0
        R = torch.matmul(V, U).to(device)
        assert torch.all(torch.det(R) >= 0)

    edge_tmp = torch.stack([e1, e2, e3], dim=2)
    rot_edges = torch.matmul(R, edge_tmp) # F x 2 x 3
    rot_e_full = rot_edges.permute(2, 0, 1) # 3 x F x 2
    cot_full = cot_full.reshape(cot_full.shape[0], cot_full.shape[1]) # 3 x F
    if renormalize == True:
        # ARAP-minimizing scaling of parameterization edge lengths
        if face_weights is not None:
            keepfs = torch.where(face_weights > normalize_filter)[0]
        else:
            keepfs = torch.arange(rot_e_full.shape[1])

        num = torch.sum(cot_full[:,keepfs] * torch.sum(rot_e_full[:,keepfs,:] * e_p_full[:,keepfs,:], dim = 2))
        denom = torch.sum(cot_full[:,keepfs] * torch.sum(e_p_full[:,keepfs,:] * e_p_full[:,keepfs,:], dim = 2))

        ratio = max(num / denom, 1e-5)
        if verbose == True:
            print(f"Scaling param. edges by ARAP-minimizing scalar: {ratio}")

        e_p_full *= ratio

    # If any non-finite values, then return None
    if not torch.all(torch.isfinite(e_p_full)) or not torch.all(torch.isfinite(rot_e_full)):
        print(f"ARAP: non-finite elements found")
        return None

    # Compute face-level distortions
    arap_tris = torch.sum(cot_full * torch.linalg.norm(e_p_full - rot_e_full, dim=2) ** 2, dim=0)
    if timeit == True:
        print(f"ARAP calculation: {time.time()-t0:0.5f}")

<<<<<<< HEAD
=======
    # Debugging: show rotated edges along with parameterization
    # import polyscope as ps
    # ps.init()
    # f1 = faces[0]
    # param_f1 = param[f1]
    # # Normalize the param so first vertex is at 0,0
    # param_f1 = param_f1 - param_f1[0]
    # og_f1 = local_tris[0] # 3 x 2
    # rot_f1 = R[0]
    # new_f1 = torch.matmul(rot_f1, og_f1.transpose(1,0)).transpose(1,0)
    # print(new_f1)
    # og_curve = ps.register_curve_network("og triangle", og_f1.numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[0,1,0])
    # param_curve = ps.register_curve_network("UV", param_f1.numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[0,0,1])
    # rot_curve = ps.register_curve_network("rot triangle", new_f1.numpy(), np.array([[0,1], [1,2], [2,0]]), enabled=True, color=[1,0,0])
    # ps.show()

    # # Compute energies
    # print(e_p_full.shape)
    # print(e_full.shape)
    # print(arap_tris[0])
    # print(torch.sum(cot_full[:,0] * torch.linalg.norm(e_p_full[:,0,:] - e_full[:,0,:], dim=1) ** 2))

    # raise

>>>>>>> 75306aed0bd8c8bbd39d40ac1e2f5fe09e151de0
    if return_face_energy == False:
        return torch.mean(arap_tris)

    return arap_tris

# ==================== Loss Functions ===============================
# Counting Loss
def count_loss(face_errors, fareas, threshold=0.1, delta=5, debug=False,
               return_softloss=True, device=torch.device("cpu"), **kwargs):
    # Normalize face error by parea
    error = face_errors
    fareas = fareas / torch.sum(fareas)
    softloss = fareas * (1 - torch.exp(-(error / threshold) ** delta))
    count_loss = torch.sum(softloss)

    if debug == True:
        print(f"Error quantile: {torch.quantile(error, torch.linspace(0, 1, 5).to(device).float())}")
        print(
            f"Thresh loss quantile: {torch.quantile(torch.exp(-(error / threshold) ** delta), torch.linspace(0, 1, 5).to(device).float())}")

    if return_softloss == True:
        return count_loss, softloss
    return count_loss

# Count Loss v2: sigmoid relaxation of L0 function
def count_loss_v2(face_errors, fareas, threshold=0.1, delta = 1, debug=False,
               return_softloss=True, device=torch.device("cpu"), **kwargs):
    # Normalize face error by parea
    error = face_errors
    fareas = fareas / torch.sum(fareas)

    # NOTE: We use anneal delta => inf s.t. sigmoid => step function
    softloss = fareas * torch.sigmoid(delta * (error - threshold))
    count_loss = torch.sum(softloss)

    if debug == True:
        print(f"Error quantile: {torch.quantile(error, torch.linspace(0, 1, 5).to(device).float())}")
        print(
            f"Thresh loss quantile: {torch.quantile(softloss, torch.linspace(0, 1, 5).to(device).float())}")

    if return_softloss == True:
        return count_loss, softloss
    return count_loss

# Relaxation of L0 function
def count_loss_l0(face_errors, fareas, threshold=0.1, delta = 1, debug=False,
               return_softloss=True, device=torch.device("cpu")):
    # Normalize face error by 3D triangle areas
    error = face_errors
    fareas = fareas / torch.sum(fareas)

    # NOTE: dclamp to 0
    error = dclamp(error - threshold, 0, 1e9)
    softloss = fareas * error ** 2/(error ** 2 + delta)
    count_loss = torch.sum(softloss)

    if debug == True:
        print(f"Error quantile: {torch.quantile(error, torch.linspace(0, 1, 5).to(device).float())}")
        print(
            f"Thresh loss quantile: {torch.quantile(softloss.float(), torch.linspace(0, 1, 5).to(device).float())}")

    if return_softloss == True:
        return count_loss, softloss
    return count_loss

# ================ Classification Metrics =====================
def compute_pr_auc(labels, preds):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    auc_score = auc(recall, precision)
    return precision, recall, thresholds, auc_score

def auc(labels, preds):
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(labels, preds)
    return auc_score

def mAP(labels, preds, multi=False):
    from sklearn.metrics import average_precision_score
    if multi == False:
        return average_precision_score(labels, preds)
    else:
        map = 0.0
        for i in range(len(labels)):
            label = labels[i]
            pred = preds[i]
            map += average_precision_score(label, pred)
        return map/i

def f1(labels, preds, multi=False):
    from sklearn.metrics import f1_score
    if multi == False:
        return f1_score(labels, preds)
    else:
        f1 = 0.0
        for i in range(len(labels)):
            label = labels[i]
            pred = preds[i]
            f1 += f1_score(label, pred)
        return f1 / i
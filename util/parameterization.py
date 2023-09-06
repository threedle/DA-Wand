# InteractiveSegmentation: Parameterization
import numpy as np
from util.util import dclamp
from cholespy import CholeskySolverD, MatrixType

# Parameterization with least squares solve
def weightedlscm(vertices, faces, face_weights=None,
                        fixzero=False, pinned_vertices=None, pinned_vertex_vals=None,
                       return_face_err=False, device=None, weight_error=False, verbose=True,
                       timeit = False):
    """Parameterize a single mesh defined by vertices and faces

    Differentiable parameterization method follows LSCM with p = 2 with the pinned vertices set to the graph diameter, unless passed by the user.
    This is a weighted version where each triangle linear system is elemetwise multiplied by the `face_weights' argument.

    Parameters
    ----------
    vertices: array-like
        Vx3 array of vertex positions
    faces: array-like
        Fx3 array of faces that index into V
    weights: array-like
        List of tensors that are per-face weights applied during parameterization
    pinned_vertices: array-like
        List of two vertex indices to pin during solving
    return_face_err: whether to also return the face-level errors
    device: pytorch device
    weight_error: whether to weight the errors by the face weights

    Returns
    -------
    (torch.Tensor, float)
        Tuple where first element is a list of Tensors with values as 2-D array of UV coordinates.
        The second element is a list of floats of the same length that records the conformal error of the mapping.

    """
    import torch

    if timeit == True:
        import time
        t0 = time.time()

    # Initialize device
    if device is None:
        device = torch.device("cpu")

    # LOOP THROUGH EACH CHART
    # Get 2 pinned vertices (indexed) sorted ascending, unless pinned vertices are specified
    from igl import boundary_loop
    if pinned_vertices is None:
        bdry = boundary_loop(faces.cpu().numpy())

        # If no boundary, then mesh is watertight
        # Choose furthest distance vertex from 0th vertex
        if len(bdry) == 0:
            from igl import edge_lengths, heat_geodesic
            t = np.mean(edge_lengths(vertices.cpu().numpy(), faces.cpu().numpy())) ** 2
            geodesics = heat_geodesic(vertices.cpu().numpy(), faces.cpu().numpy(), t, np.array([0]))
            pverts = torch.tensor([0, np.argmax(geodesics)]).to(device).long()
        else:
            pverts, ppos = torch.sort(torch.tensor([bdry[0], bdry[int(len(bdry) / 2)]]))
            pverts = pverts.to(device).long()
    else:
        pverts = pinned_vertices.to(device)
        # Edge case: need one more pinned vertex if input pinning is length 1
        if len(pverts) == 1:
            bdry = boundary_loop(faces.cpu().numpy())
            # If no boundary, then mesh is watertight
            # Choose two points with furthest distance
            if len(bdry) == 0:
                if pverts[0] == 0 or pverts[0] == len(vertices.cpu().numpy()) - 1:
                    pverts = torch.cat((pverts, torch.tensor([1]).to(device)))
                else:
                    pverts = torch.cat((pverts, torch.tensor([0]).to(device)))
            else:
                if pverts[0] == bdry[0] or pverts[0] == bdry[int(len(bdry) / 2)]:
                    pverts = torch.cat((pverts, torch.tensor([bdry[1]]).to(device)))
                else:
                    pverts = torch.cat((pverts, torch.tensor([bdry[0]]).to(device)))
    pverts, ppos = torch.sort(pverts.long())
    # print(f"Pinned vertices: {pverts}")

    num_faces = faces.shape[0]
    num_verts = vertices.shape[0]
    # Compute the indices of the non-pinned vertices
    free_vertex_idxs = torch.Tensor(list(set(range(num_verts)).difference(set(pverts.tolist())))).to(device).long()

    # Vectorized generation of W matrices
    fverts = vertices[faces].double()

    # Convert to local frame
    e1 = fverts[:, 1, :] - fverts[:, 0, :]
    e2 = fverts[:, 2, :] - fverts[:, 0, :]
    s = torch.linalg.norm(e1, dim=1)
    t = torch.linalg.norm(e2, dim=1)
    angle = torch.acos(torch.sum(e1 / s[:, None] * e2 / t[:, None], dim=1))
    x = torch.column_stack([torch.zeros(len(angle)).to(device), s, t * torch.cos(angle)]).to(device)
    y = torch.column_stack([torch.zeros(len(angle)).to(device), torch.zeros(len(angle)).to(device), t * torch.sin(angle)]).to(device)
    sq_d = torch.sqrt(x[:, 1] * y[:, 2]).reshape(len(x), 1).to(device)

    if face_weights is None:
        fweights = torch.ones((len(x), 1)).to(device)
    else:
        face_weights = face_weights.to(device)

        if len(face_weights.shape) == 1:
            face_weights = face_weights.reshape(len(face_weights), 1)

        if fixzero == True:
            # Add small buffer to zero weights
            if len(torch.nonzero(face_weights.detach() == 0)) > 0 and verbose==True:
                print("Fixzero triggered...")

            # Clamp the weights between 1e-8 and 1
            fweights = dclamp(face_weights, min=1e-8, max=1)
        else:
            fweights = face_weights

    if pinned_vertex_vals is None:
        Up = torch.zeros(len(pverts) * 2, device=device).double()
        Up[[len(pverts) - 1, len(pverts) * 2 - 1]] = 1
        Up = Up.reshape(2 * len(pverts), 1)
    else:
        Up = pinned_vertex_vals.to(device).reshape(len(pinned_vertex_vals), 1)

    # Unweighted version for error
    Wr_unw = torch.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]]) / sq_d
    Wi_unw = torch.column_stack([y[:, 2] - y[:, 1], y[:, 0] - y[:, 2], y[:, 1] - y[:, 0]]) / sq_d

    # Fill out sparse matrices
    M_r_unw = torch.zeros((len(Wr_unw), len(vertices)), dtype=torch.double).to(device)
    M_i_unw = torch.zeros((len(Wr_unw), len(vertices)), dtype=torch.double).to(device)
    for i, face in enumerate(faces):
        M_r_unw[i, face] = Wr_unw[i]
        M_i_unw[i, face] = Wi_unw[i]

    # Separate out vertices into free and pinned
    Mf_1_unw = M_r_unw[:, free_vertex_idxs]
    Mf_2_unw = M_i_unw[:, free_vertex_idxs]
    Mp_1_unw = M_r_unw[:, pverts]
    Mp_2_unw = M_i_unw[:, pverts]

    # Build weighted versions
    Mf_1 = Mf_1_unw * fweights
    Mf_2 = Mf_2_unw * fweights
    Mp_1 = Mp_1_unw * fweights
    Mp_2 = Mp_2_unw * fweights

    # Construct the matrices A and b
    # A => Jacobian => want to be as close as possible to similarity matrix transformation, ergo ((a, -b), (b, a))
    A = torch.cat((torch.cat((Mf_1, -Mf_2), dim=1), torch.cat((Mf_2, Mf_1), dim=1)), dim=0)
    b = -torch.matmul(torch.cat((torch.cat((Mp_1, -Mp_2), dim=1), torch.cat((Mp_2, Mp_1), dim=1)), dim=0), Up)

    A_unw = torch.cat((torch.cat((Mf_1_unw, -Mf_2_unw), dim=1), torch.cat((Mf_2_unw, Mf_1_unw), dim=1)), dim=0)
    b_unw = -torch.matmul(
        torch.cat((torch.cat((Mp_1_unw, -Mp_2_unw), dim=1), torch.cat((Mp_2_unw, Mp_1_unw), dim=1)), dim=0),
        Up)

    if timeit == True:
        print(f"Param. matrix setup: {time.time() - t0:0.5f} seconds")
        t0 = time.time()

    # TODO: Replace with sparse solver
    # NOTE: 'A' matrix goes into cholesky solver d
    #          solve(b matrix, x for holding solutions)
    # choleskysolver = CholeskySolverD(self.n, self.inds[0,:], self.inds[1,:], self.vals, MatrixType.COO)
    # b = b.double().contiguous()
    # c = b.permute(1,2,0).contiguous()
    # c = c.view(c.shape[0], -1)
    # x = torch.zeros_like(c)
    # choleskysolver.solve(c, x)
    # x = x.view(b.shape[1], b.shape[2], b.shape[0])
    # x = x.permute(2,0,1).contiguous()

    x = torch.linalg.lstsq(A, b).solution

    if timeit == True:
        print(f"Param. lstsq problem solve: {time.time() - t0:0.5f} seconds")
        t0 = time.time()
    nonf_count = torch.sum(~torch.isfinite(x))/len(x)
    if not torch.all(torch.isfinite(x)):
        print(f"LSTSQ solution resulted in non-finite values: {nonf_count}")
        print(f"Trying on CPU with condition-robust driver...")
        x = torch.linalg.lstsq(A.cpu(), b.cpu(), driver="gelsd").solution
        x = x.to(device)

    if not torch.all(torch.isfinite(x)):
        print(f"CPU solve failed. Trying CVXPY...")
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        x_var = cp.Variable((A.shape[1], 1))
        A_var = cp.Parameter(A.shape)
        b_var = cp.Parameter(b.shape)
        objective = cp.Minimize(cp.pnorm(A_var @ x_var - b_var, p=2)**2)
        problem = cp.Problem(objective)
        # assert problem.is_dpp()
        layer = CvxpyLayer(problem, parameters=[A_var, b_var], variables=[x_var])
        x, = layer(A, b)

    doub_nverts = x.shape[0]
    assert doub_nverts % 2 == 0
    param = torch.column_stack([x[:int(doub_nverts / 2)], x[int(doub_nverts / 2):]])
    # NOTE: Below assumes pverts is SORTED
    for i in range(len(pverts)):
        param = torch.cat((param[:pverts[i]], Up[[i, len(pverts) + i]].transpose(0, 1), param[pverts[i]:]), dim=0)

    # Save face-level error if set
    if return_face_err == True:
        face_err_tmp = torch.matmul(A_unw, x) - b_unw
        face_err_tmp = face_err_tmp ** 2
        ferr_ret = torch.sum(torch.column_stack([face_err_tmp[:int(face_err_tmp.shape[0] / 2)],
                                                 face_err_tmp[int(face_err_tmp.shape[0] / 2):]]), dim=1)
        assert ferr_ret.shape[0] == faces.shape[0]

        return param, ferr_ret

        if weight_error == True:
            w_face_err_tmp = torch.matmul(A, x) - b
            w_face_err_tmp = w_face_err_tmp ** 2
            wferr_ret = torch.sum(torch.column_stack([w_face_err_tmp[:int(w_face_err_tmp.shape[0] / 2)],
                                                      w_face_err_tmp[int(w_face_err_tmp.shape[0] / 2):]]), dim=1)
            assert wferr_ret.shape[0] == faces.shape[0]

            return param, wferr_ret

    if timeit == True:
        print(f"Param. solution postprocessing: {time.time() - t0:0.5f} seconds")
        t0 = time.time()

    # Return the parameterization coordinates and error values
    return param
import torch
import utils
import datagen

# DEBUG = False


from config import DEBUG, FP_DTYPE, PREC_EPS


def network_anomalography_obj(scenario, P, Q, A, lam, mu, batch_mean=True):
    Y, R, Omega = datagen.nw_scenario_observation(scenario)
    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)
    # batch_size = scenario["batch_size"]
    obj = _network_anomalography_obj_primitive(Y, R, Omega, P, Q, A, lam, mu, batch_mean=batch_mean)
    return obj


def _network_anomalography_obj_primitive(Y, R, Omega, P, Q, A, lam, mu, batch_mean=True):
    obj = utils.frob_norm_sq(Omega * (Y - P @ Q.mT - R @ A), dim=(-2, -1)) / 2
    obj += lam * (utils.frob_norm_sq(P, dim=(-2, -1)) + utils.frob_norm_sq(Q, dim=(-2, -1))) / 2
    obj += utils.l1_norm(mu * A, dim=(-2, -1))

    if batch_mean:
        obj = obj.mean(dim=-1)  # only batch size, iteration dimension is kept

    return obj


def bsca_incomplete_meas(scenario_dict, lam, mu, rank, num_iter=10, return_im_steps=True, init="randsc", order="PQA", rconv=None):
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)

    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    # Init
    # if init == "default":
    #     P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)
    # elif init == "detuneq":
    #     P, Q, A = _bsca_incomplete_meas_init_deterministic_uneq(Y, R, Omega, rank, alpha=1.0)
    if init == "randsc":
        # print("sigma=0.1")
        P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1)
    # elif init == "randn":
    #     P, Q, A = _bsca_incomplete_meas_init_randn(Y, R, Omega, rank)
    else:
        raise ValueError

    if rconv is not None:
        obj = _network_anomalography_obj_primitive(Y, R, Omega, P, Q, A, lam, mu, batch_mean=True)

    if return_im_steps:
        P_list = [P]
        Q_list = [Q]
        A_list = [A]

    for i in range(num_iter):
        if DEBUG:
            print("Iteration {}".format(i+1))
        P_new, Q_new, A_new = _bsca_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order=order)

        if rconv is not None:
            obj_new = _network_anomalography_obj_primitive(Y, R, Omega, P_new, Q_new, A_new, lam, mu, batch_mean=True)
            rcrit = (obj - obj_new).abs() / obj
            if DEBUG:
                print("obj={}, relcrit={}".format(obj_new, rcrit))
            if rcrit < rconv:
                print("Converged after {} iterations.".format(i+1))
                break

            obj = obj_new

        if return_im_steps:
            # P_list.append(P)
            # Q_list.append(Q)
            # A_list.append(A_new)

            # P_list.append(P_new)
            # Q_list.append(Q)
            # A_list.append(A_new)

            P_list.append(P_new)
            Q_list.append(Q_new)
            A_list.append(A_new)

        P, Q, A = P_new, Q_new, A_new

    if return_im_steps:
        P_list = torch.stack(P_list)
        Q_list = torch.stack(Q_list)
        A_list = torch.stack(A_list)
        return P_list, Q_list, A_list
    else:
        return P, Q, A


def bsca_incomplete_meas_minerr(scenario_dict, lam, mu, rank, num_iter=10, return_im_steps=True, init="default", rconv=None):
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)

    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    # Init
    if init == "default":
        P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)
    elif init == "detuneq":
        P, Q, A = _bsca_incomplete_meas_init_deterministic_uneq(Y, R, Omega, rank, alpha=1.0)
    elif init == "randsc":
        # print("sigma=0.1")
        P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1)
    elif init == "randn":
        P, Q, A = _bsca_incomplete_meas_init_randn(Y, R, Omega, rank)
    else:
        raise ValueError

    if rconv is not None:
        obj = _network_anomalography_obj_primitive(Y, R, Omega, P, Q, A, lam, mu, batch_mean=True)

    if return_im_steps:
        P_list = [P]
        Q_list = [Q]
        A_list = [A]

    for i in range(num_iter):
        if DEBUG:
            print("Iteration {}".format(i+1))
        P_new, Q_new, A_new = _bsca_incomplete_meas_iteration_minerr(Y, R, Omega, P, Q, A, lam, mu)

        if rconv is not None:
            obj_new = _network_anomalography_obj_primitive(Y, R, Omega, P_new, Q_new, A_new, lam, mu, batch_mean=True)
            rcrit = (obj - obj_new).abs() / obj
            if DEBUG:
                print("obj={}, relcrit={}".format(obj_new, rcrit))
            if rcrit < rconv:
                print("Converged after {} iterations.".format(i+1))
                break

            obj = obj_new

        if return_im_steps:
            # P_list.append(P)
            # Q_list.append(Q)
            # A_list.append(A_new)

            # P_list.append(P_new)
            # Q_list.append(Q)
            # A_list.append(A_new)

            P_list.append(P_new)
            Q_list.append(Q_new)
            A_list.append(A_new)

        P, Q, A = P_new, Q_new, A_new

    if return_im_steps:
        P_list = torch.stack(P_list)
        Q_list = torch.stack(Q_list)
        A_list = torch.stack(A_list)
        return P_list, Q_list, A_list
    else:
        return P, Q, A


def _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank):
    # Init
    batch_size = Y.shape[:-2]
    num_flows = R.shape[-1]
    num_edges = Y.shape[-2]
    num_time_steps = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ratio = torch.sqrt(torch.tensor(num_edges / num_time_steps))
    ymean = torch.sum(torch.abs(Y), dim=(-2, -1)) / torch.sum(Omega, dim=(-2, -1))  # abs of Y is dirty fix against negative values

    pval = torch.sqrt(ymean / rank / ratio)
    qval = torch.sqrt(ymean * ratio / rank)
    P = torch.ones(*batch_size, num_edges, rank, dtype=FP_DTYPE, device=Y.device) * pval.unsqueeze(-1).unsqueeze(-1)
    Q = torch.ones(*batch_size, num_time_steps, rank, dtype=FP_DTYPE, device=Y.device) * qval.unsqueeze(-1).unsqueeze(-1)
    A = torch.zeros(*batch_size, num_flows, num_time_steps, dtype=FP_DTYPE, device=Y.device)

    return P, Q, A


def _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1):
    # Init
    P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)

    num_edges = Y.shape[-2]
    num_time_steps = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ratio = torch.sqrt(torch.tensor(num_edges / num_time_steps))
    ymean = torch.sum(torch.abs(Y), dim=(-2, -1)) / torch.sum(Omega, dim=(-2, -1))
    pval = torch.sqrt(ymean / rank / ratio)
    qval = torch.sqrt(ymean * ratio / rank)

    P = P + pval.unsqueeze(-1).unsqueeze(-1)*sigma*torch.randn_like(P)
    Q = Q + qval.unsqueeze(-1).unsqueeze(-1)*sigma*torch.randn_like(Q)

    return P, Q, A


def _bsca_incomplete_meas_init_randn(Y, R, Omega, rank):
    # Init
    batch_size = Y.shape[:-2]
    num_flows = R.shape[-1]
    num_edges = Y.shape[-2]
    num_time_steps = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ratio = torch.sqrt(torch.tensor(num_edges / num_time_steps))
    ymean = torch.sum(torch.abs(Y), dim=(-2, -1)) / torch.sum(Omega, dim=(-2, -1))  # abs of Y is dirty fix against negative values

    # This pretends that P and Q were correlated, leadig to the elements of P@Q.T being chi-squared(rank) with mean rank*pval*qval
    pval = torch.sqrt(ymean / rank / ratio)
    qval = torch.sqrt(ymean * ratio / rank)
    P = torch.randn(*batch_size, num_edges, rank, dtype=FP_DTYPE) * pval.unsqueeze(-1).unsqueeze(-1)
    Q = torch.randn(*batch_size, num_time_steps, rank, dtype=FP_DTYPE) * qval.unsqueeze(-1).unsqueeze(-1)
    A = torch.zeros(*batch_size, num_flows, num_time_steps, dtype=FP_DTYPE)

    return P, Q, A


def _bsca_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order="PQA"):
    """mu - must broadcast with A"""
    if order == "PQA":
        err = Omega * (Y - (R @ A))
        P_new = _bsca_update_P(Y, R, Omega, Q, A, lam, err)
        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam, err)
        # Q_new = Q
        A_new = _bsca_update_A(Y, R, Omega, P_new, Q_new, A, mu, err)
        # A_new = A
    elif order == "APQ":
        A_new = _bsca_update_A(Y, R, Omega, P, Q, A, mu)
        err = Omega * (Y - (R @ A_new))
        P_new = _bsca_update_P(Y, R, Omega, Q, A_new, lam, err)
        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A_new, lam, err)
    else:
        raise ValueError

    return P_new, Q_new, A_new


def _bsca_incomplete_meas_iteration_minerr(Y, R, Omega, P, Q, A, lam, mu):
    """mu - must broadcast with A"""
    err = Omega * (Y - (R @ A))
    P_new = _bsca_update_P(Y, R, Omega, Q, A, lam, err)
    Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam, err)
    A_new = _bsca_update_A_rlx_minerr(Y, R, Omega, P_new @ Q_new.mT, A, mu, err=err)

    return P_new, Q_new, A_new


def _bsca_update_P(Y, R, Omega, Q, A, lam, err=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = Q.shape[-1]

    if err is None:
        rhs = ((Omega**2) * (Y - (R @ A))) @ Q
    else:
        rhs = (Omega * err) @ Q
    rhs = rhs.unsqueeze(-1)  # (*batch, links, rank, 1)

    lhs = Q.mT.unsqueeze(-3) @ ((Omega.unsqueeze(-1)**2) * Q.unsqueeze(-3))
    if FP_DTYPE == torch.float:
        regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), lam.unsqueeze(-1))
    else:
        regularizer = lam.unsqueeze(-1)
    lhs = lhs + torch.eye(rank, dtype=FP_DTYPE, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)
    # try:
    P_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)
    # except:
    #     print("This shouldn't occur anymore")

    return P_new


def _bsca_update_Q(Y, R, Omega, P, A, lam, err=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = P.shape[-1]

    if err is None:
        rhs = ((Omega**2) * (Y - R @ A)).mT @ P
    else:
        rhs = (Omega * err).mT @ P
    rhs = rhs.unsqueeze(-1)  # (*batch, time, rank, 1)

    lhs = P.mT.unsqueeze(-3) @ ((Omega.mT.unsqueeze(-1)**2) * P.unsqueeze(-3))
    if FP_DTYPE == torch.float:
        regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), lam.unsqueeze(-1))
    else:
        regularizer = lam.unsqueeze(-1)
    lhs = lhs + torch.eye(rank, dtype=FP_DTYPE, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)

    # try:
    Q_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)  # (*batch, time, rank, 1)
    # except:
        # print("wtf")

    return Q_new


def _bsca_update_A(Y, R, Omega, P, Q, A, mu, err=None, soft_thresh_cont_grad=False):
    """All matrices have leading batch dimension.
        mu must broadcast with A.
        err = Omega * (Y - (R @ A))
        """
    X = P @ Q.mT
    return _bsca_update_A_rlx(Y, R, Omega, X, A, mu, err=err, soft_thresh_cont_grad=soft_thresh_cont_grad, return_gamma=False)


def _bsca_update_A_rlx(Y, R, Omega, X, A, mu, err=None, soft_thresh_cont_grad=False, return_gamma=False):
    """All matrices have leading batch dimension.
        mu must broadcast with A.
        err = Omega * (Y - (R @ A))
        X = P @ Q.mT
        """

    if err is None:
        full_err = Omega * (Y - (R @ A) - X)
    else:
        full_err = (err - Omega * X)

    # Direction
    A_scale = (R**2).mT @ (Omega**2)
    A_scale_zero = A_scale == 0
    soft_thresh_args = (A_scale * A + R.mT @ (full_err*Omega), mu)
    if soft_thresh_cont_grad:
        BA_temp = utils.soft_thresholding_contgrad(*soft_thresh_args)
    else:
        BA_temp = utils.soft_thresholding(*soft_thresh_args)

    A_scale_safezero = A_scale + A_scale_zero * 1
    BA = BA_temp / A_scale_safezero
    BA[A_scale_zero] = 0  # set direction to 0 where A does not receive information (no connected link measurements for particular a)

    # Step size
    proj_step = (R @ (BA - A)) * Omega
    # denom = Omega * proj_step
    denom = torch.sum(proj_step.square(), dim=(-2, -1))
    nom1 = - full_err * proj_step
    nom1 = torch.sum(nom1, dim=(-2, -1))
    nom2 = utils.l1_norm(mu * BA, dim=(-2, -1)) - utils.l1_norm(mu * A, dim=(-2, -1))
    nom = - nom1 - nom2

    denom_zero = denom == 0
    denom[denom_zero] = 1  # avoiding division by 0
    gamma = nom / denom
    gamma[denom_zero] = 0
    gamma = torch.clamp(gamma, min=0, max=1)
    if torch.any(torch.isnan(gamma)) or torch.any(torch.isnan(BA)):
        raise RuntimeError("Gamma or BA was nan")

    # print("Gamma", gamma.mean())

    # Step
    A_new = gamma.unsqueeze(-1).unsqueeze(-1) * BA + (1 - gamma.unsqueeze(-1).unsqueeze(-1)) * A

    if return_gamma:
        return A_new, gamma
    else:
        return A_new


def _bsca_update_A_rlx_minerr(Y, R, Omega, X, A, mu, err=None, soft_thresh_cont_grad=False, return_gamma=False):
    """All matrices have leading batch dimension.
        mu must broadcast with A.
        err = Omega * (Y - (R @ A))
        X = P @ Q.mT
        """
    """QUICKFIX"""
    # mu = 0

    if err is None:
        full_err = Omega * (Y - (R @ A) - X)
    else:
        full_err = (err - Omega * X)

    HIGH = 1e12
    EPS = 1e-12

    edge_corresp = R.mT.unsqueeze(-1)  # dim (F, E, 1)
    no_edge_corresp = edge_corresp == 0
    # unobserved = edge_corresp == 0
    # reliable = torch.amin((R.mT.unsqueeze(-1) * Omega.unsqueeze(-3)) + no_edge_corresp * HIGH, dim=-2) != 0

    # err_corresp_to_ano = (Omega*(Y - X)).unsqueeze(-3) / (edge_corresp + EPS)
    err_corresp_to_ano = full_err.unsqueeze(-3).expand(Y.shape[-3], R.shape[-1], *Y.shape[-2:])
    # min_corresp_err_ind = torch.min(err_corresp_to_ano.abs() + unobserved * HIGH, dim=-2, keepdim=True)[1]  # get only indices
    min_corresp_err_ind = torch.min(err_corresp_to_ano.abs() + no_edge_corresp * HIGH, dim=-2, keepdim=True)[1]
    min_corresp_err = torch.gather(err_corresp_to_ano, -2, min_corresp_err_ind).squeeze(-2)  # collapse edge dim


    # Direction
    # A_scale = (R**2).mT @ (Omega**2)
    # A_scale_zero = A_scale == 0
    soft_thresh_args = (A + min_corresp_err, mu)
    BA = utils.soft_thresholding_contgrad(*soft_thresh_args)

    # Step size
    proj_step = (R @ (BA - A)) * Omega
    # denom = Omega * proj_step
    denom = torch.sum(proj_step.square(), dim=(-2, -1))
    nom1 = - full_err * proj_step
    nom1 = torch.sum(nom1, dim=(-2, -1))
    nom2 = utils.l1_norm(mu * BA, dim=(-2, -1)) - utils.l1_norm(mu * A, dim=(-2, -1))
    nom = - nom1 - nom2

    denom_zero = denom == 0
    denom[denom_zero] = 1  # avoiding division by 0
    gamma = nom / denom
    gamma[denom_zero] = 0
    gamma = torch.clamp(gamma, min=0, max=1)

    if torch.any(torch.isnan(gamma)) or torch.any(torch.isnan(BA)):
        raise RuntimeError("Gamma or BA was nan")

    # print("Gamma_minerr", gamma.mean())

    # Step
    A_new = gamma.unsqueeze(-1).unsqueeze(-1) * BA + (1 - gamma.unsqueeze(-1).unsqueeze(-1)) * A

    if return_gamma:
        return A_new, gamma
    else:
        return A_new


def batch_bcd_incomplete_meas(scenario_dict, lam, mu, rank, num_iter=10, init="default", return_im_steps=True, order="APQ"):
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)

    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    # Init
    if init == "default":
        P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, rank)
    elif init == "randsc":
        # print("sigma=0.1")
        P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, rank, sigma=0.1)
    elif init == "randn":
        P, Q, A = _bsca_incomplete_meas_init_randn(Y, R, Omega, rank)
    else:
        raise ValueError
    if return_im_steps:
        P_list = [P]
        Q_list = [Q]
        A_list = [A]

    for _ in range(num_iter):
        P, Q, A = _batch_bcd_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order=order)
        if return_im_steps:
            P_list.append(P)
            Q_list.append(Q)
            A_list.append(A)

    if return_im_steps:
        P_list = torch.stack(P_list)
        Q_list = torch.stack(Q_list)
        A_list = torch.stack(A_list)
        return P_list, Q_list, A_list
    else:
        return P, Q, A


def _batch_bcd_incomplete_meas_iteration(Y, R, Omega, P, Q, A, lam, mu, order="APQ"):
    if order == "APQ":
        err = Omega * (Y - (R @ A))
        A_new = _batch_bcd_update_A(Y, R, Omega, P, Q, A, mu, err)
        err = Omega * (Y - (R @ A_new))
        P_new = _batch_bcd_update_P(Y, R, Omega, Q, A_new, lam, err)
        Q_new = _batch_bcd_update_Q(Y, R, Omega, P_new, A_new, lam, err)

    if order == "APQold":
        err = Omega * (Y - (R @ A))
        A_new = _batch_bcd_update_A(Y, R, Omega, P, Q, A, mu, err)
        P_new = _batch_bcd_update_P(Y, R, Omega, Q, A_new, lam, err)
        Q_new = _batch_bcd_update_Q(Y, R, Omega, P_new, A_new, lam, err)

    elif order == "PQA":
        err = Omega * (Y - (R @ A))
        P_new = _batch_bcd_update_P(Y, R, Omega, Q, A, lam, err)
        Q_new = _batch_bcd_update_Q(Y, R, Omega, P_new, A, lam, err)
        A_new = _batch_bcd_update_A(Y, R, Omega, P_new, Q_new, A, mu, err)

    return P_new, Q_new, A_new


def _batch_bcd_update_P(Y, R, Omega, Q, A, lam, err=None):
    return _bsca_update_P(Y, R, Omega, Q, A, lam, err=None)


def _batch_bcd_update_Q(Y, R, Omega, P, A, lam, err=None):
    return _bsca_update_Q(Y, R, Omega, P, A, lam, err=None)


def _batch_bcd_update_A(Y, R, Omega, P, Q, A, mu, err=None):
    """All matrices have leading batch dimension.
        err = Omega * (Y - (R @ A))"""
    if err is None:
        full_err = Omega * (Y - (R @ A) - (P @ Q.mT))
    else:
        full_err = Omega * (err - (P @ Q.mT))

    num_flows = R.shape[-1]
    A_new = torch.zeros_like(A)
    for f in range(num_flows):
        Y_f = Omega * (Y - P @ Q.mT - R[..., :, :f] @ A_new[..., :f, :] - R[..., :, f:] @ A[..., f:, :])
        A_new[..., f, :] = utils.soft_thresholding(R[..., :, f].unsqueeze(-2) @ Y_f, mu).squeeze(-2) \
                           / utils.frob_norm_sq(R[..., :, f], dim=-1).unsqueeze(-1)

    return A_new

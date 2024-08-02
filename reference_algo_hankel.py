"""
Author: 
Lukas Schynol
lschynol@nt.tu-darmstadt.de
"""

import time
import torch
import utils
import tensor_utils as tutils
import datagen

from config import DEBUG, FP_DTYPE, PREC_EPS

"""ADMM parameters (adopted from reference)"""
ADMM_MAX_ITER = 10000
ADMM_EPS_ABS = 1e-5
ADMM_EPS_REL = 1e-3

"""STELA parameters"""
STELA_MAX_ITER = 1000
# STELA_GAMMA_MIN = 1e-12
STELA_MIN_DIFF = 1e-6


# ADMM_EPS_ABS = 1e-12
# ADMM_EPS_REL = 1e-9


def online_obj(scenario_dict, window_length, U_list, V_list, W, A, lam, mu_r, mu_h, mu_s, batch_mean=True, t=None):
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    winlen = window_length  # alias

    # if not winlen:
    #     winlen = scenario_dict["num_time_seg"]
    batch_size = Y.shape[:-2]
    num_links, num_time_steps = Y.shape[-2:]
    num_flows = R.shape[-1]

    # Hankelization
    fill = torch.zeros_like(Y)[...,
           :(winlen - 1)]  # dummies such that Hankel has same length, not described in reference
    Yh = tutils.hankel_transform(torch.cat([fill, Y], dim=-1), winlen)
    Omegah = tutils.hankel_transform(torch.cat([fill, Omega], dim=-1), winlen)

    Aprep = [torch.zeros_like(A[..., 0]).unsqueeze(-1)] * (window_length - 1)
    RAh = tutils.hankel_transform(R @ torch.cat([*Aprep, A], dim=-1), winlen)

    assert (len(U_list) == len(V_list))
    Xh_list = []
    for i in range(len(U_list)):
        Xh_list.append(tutils.cpd(U_list[i], V_list[i], W))

    obj_t = []
    if t is None:
        t_list = list(range(Y.shape[-1]))
    else:
        t_list = t

    for t in t_list:
        lam_stack = (lam ** torch.arange(0, t + 1)).flip(0)

        df = Omegah[..., :(t + 1)] * (Yh[..., :(t + 1)] - Xh_list[t][..., :(t + 1)] - RAh[..., :(t + 1)])
        df = torch.linalg.matrix_norm(df, dim=(-3, -2)) ** 2  # * lam_stack.view(t, 1, 1)
        df = df / 2

        hkl = Omegah[..., :-1, 1:] * (Xh_list[t][..., 1:, :-1] - Xh_list[t][..., :-1, 1:])
        hkl = torch.linalg.matrix_norm(hkl, dim=(-3, -2)) ** 2
        hkl = torch.cat([torch.zeros_like(hkl[..., 0].unsqueeze(-1)), hkl], dim=-1)[...,
              :(t + 1)]  # for first sum term, there is no hankel reg
        hkl = mu_h * hkl / 2

        # mu_r_bar = mu_r / torch.cumsum(lam_stack.flip(0), 0)[-1]
        mu_r_bar = mu_r / torch.sum(lam_stack)  # also wrong in paper, upper bound of sum is t not tau
        rr = torch.linalg.matrix_norm(U_list[t], dim=(-2, -1)) ** 2 + torch.linalg.matrix_norm(V_list[t],
                                                                                               dim=(-2, -1)) ** 2
        rr = rr.unsqueeze(-1) * mu_r_bar + mu_r * torch.linalg.vector_norm(W[..., :(t + 1), :], dim=-1) ** 2
        rr = rr / 2

        sp = mu_s * A[..., :(t + 1)].abs().sum(dim=-2)  # 2x likely forgotten in paper

        obj_temp = (df + hkl + rr + sp) * lam_stack
        obj_t.append(obj_temp.sum(dim=-1))

    obj_t = torch.stack(obj_t, dim=0)
    if batch_mean:
        obj_t = obj_t.mean(dim=-1)

    return obj_t


def subspace_tracking_anomography(scenario_dict, rank, lam, mu_r, mu_h, mu_s, window_length=None,
                                  device=torch.device("cpu")):
    """Compared to reference paper, (A, C, B) -> (U, V, W) and V -> A.
    CPD-size (num_links, window_len, num_time_steps)"""
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
    device = Y.device
    # lam = 0.173
    # mu_h = 0.973
    # window_length = 67
    # mu_r = 0.004
    # mu_s = 0.015

    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    if not window_length:
        window_length = scenario_dict["num_time_seg"]
    batch_size = Y.shape[:-2]
    num_links, num_time_steps = Y.shape[-2:]
    num_flows = R.shape[-1]

    # Hankelization
    fill0 = torch.zeros_like(Y)[...,
            :(window_length - 1)]  # dummies such that Hankel has same length, not described in reference
    Yh = tutils.hankel_transform(torch.cat([fill0, Y], dim=-1), window_length)
    # fill1 = torch.zeros_like(Omega)[..., :(window_length - 1)]
    Omegah = tutils.hankel_transform(torch.cat([fill0, Omega], dim=-1), window_length)

    # Yh = tutils.hankel_transform(Y, window_length)
    # Omegah = tutils.hankel_transform(Omega, window_length)
    T_max = Yh.shape[-1]

    """Initialization"""
    # X = []
    W = []
    A = [torch.zeros(*batch_size, num_flows, device=device)] * (window_length - 1)

    # Does not work, initialization not clear from reference
    # U = torch.zeros(*batch_size, num_links, rank, dtype=FP_DTYPE)
    # V = torch.zeros(*batch_size, window_length, rank, dtype=FP_DTYPE)
    # w_t = torch.zeros(*batch_size, rank, dtype=FP_DTYPE)

    # Not described in paper, therefore just a reasonable choice here
    stdapprx = (torch.var(Yh[..., 0], dim=(-1, -2)) / rank) ** (1 / 6)
    U = torch.randn(*batch_size, num_links, rank, dtype=FP_DTYPE, device=device) * stdapprx.unsqueeze(-1).unsqueeze(-1)
    V = torch.randn(*batch_size, window_length, rank, dtype=FP_DTYPE, device=device) * stdapprx.unsqueeze(-1).unsqueeze(
        -1)
    w_t = torch.randn(*batch_size, rank, dtype=FP_DTYPE, device=device) * stdapprx.unsqueeze(-1)

    ### DEBUGGING
    # U = scenario_dict.data["P"]
    # V = torch.ones(*batch_size, window_length, rank, dtype=FP_DTYPE)
    # V = torch.zeros(*batch_size, window_length, rank, dtype=FP_DTYPE)
    # V[..., -1, :] = 1.0

    RU_t = mu_r * torch.eye(rank, dtype=FP_DTYPE, device=device)  # assumed, not described in paper
    RV_t = mu_r * torch.eye(rank, dtype=FP_DTYPE, device=device).expand(*batch_size, window_length, rank,
                                                                        rank)  # assumed, not described in paper

    admm_init = None

    # DEBUGGING
    # U_list = []
    # V_list = []

    """Iteration"""
    for t in range(T_max):
        if False:
            print("Iteration {}".format(t))
        # Simplification here (as in paper)
        if t == 0:  # not described in paper, but makes sense
            lam_t = 0
        else:
            lam_t = lam
        mu_r_prev = mu_r

        # Do iteration
        Yh_t = Yh[..., t]
        Omegah_t = Omegah[..., t]

        U, RU_t, V, RV_t, w_t, a_t, admm_init = _subspace_tracking_anomography_iteration(
            Yh_t, R, Omegah_t, U, RU_t, V, RV_t, w_t, A, lam_t, mu_r, mu_r_prev, mu_h, mu_s,
            admm_init=admm_init)  # A will be modified in place

        # U_list.append(U)
        # V_list.append(V)

        # x_t = tutils.cpd(U, V, w_t.unsqueeze(-2))
        # X.append(x_t)
        W.append(w_t)
        A.append(a_t)
        # print(V.abs().sum(dim=(-2, -1)).mean(dim=0), U.abs().sum(dim=(-2, -1)).mean(dim=0), w_t.abs().sum(dim=(-2)).mean(dim=0))
        if U.abs().sum(dim=(-2, -1)).mean(dim=0) > 1e6 or V.abs().sum(dim=(-2, -1)).mean(dim=0) > 1e6:
            print("Algorithm diverged due to mu_h being too large.")
            for _ in range(t + 1, T_max):
                W.append(torch.zeros_like(W[0]))
                A.append(torch.zeros_like(A[0]))
            break

    W = torch.stack(W, dim=-2)
    A = A[(window_length - 1):]  # remove first window_length-1 entries again
    A = torch.stack(A, dim=-1)

    """
    # # DEBUGGING
    import matplotlib.pyplot as plt
    # Xest_list = []
    # Xesth_list = []
    # Xdiff = []
    # Xdiffh = []
    # for i in range(len(U_list)):
    #     # Xest_list.append(tutils.inverse_hankel_transform(tutils.cpd(U_list[i], V_list[i], W), window_length)[..., :-(window_length-1)])
    #     Xesth_list.append(tutils.cpd(U_list[i], V_list[i], W))
    #     Xest_list.append(tutils.inverse_hankel_transform(Xesth_list[-1], window_length)[..., (window_length-1):])
    #     Xdiff.append(torch.mean((Omega * (Xest_list[i] - Y)).square()))
    #     Xdiffh.append(torch.mean((Omegah * (Xesth_list[i] - Yh)).square()))
    #
    # Xdiff2 = Omega * (Xest_list[-1] - Y)
    # Xdiff2 = Xdiff2.square().mean(dim=(0, -2))
    # Xdiff2h = Omegah * (Xesth_list[i] - Yh)
    # Xdiff2h = Xdiff2h.square().mean(dim=(0, -3, -2))
    #
    # plt.figure()
    # plt.plot(torch.tensor(Xdiff))
    # plt.plot(torch.tensor(Xdiffh))
    # plt.title("Global Xdiff over iterations")
    #
    # plt.figure()
    # plt.plot(torch.tensor(Xdiff2))
    # plt.plot(torch.tensor(Xdiff2h))
    # plt.title("Temporal Xdiff over iterations")
    obj_t = online_obj(scenario_dict, window_length, U_list, V_list, W, A, lam, mu_r, mu_h, mu_s)
    plt.figure()
    plt.plot(obj_t, "b")

    torch.manual_seed(0)
    EPS = 1e-4
    obj_t_n = []
    for i in range(25):
        ### Probe W Opt
        # obj_t_n.append(torch.zeros(W.shape[-2], dtype=FP_DTYPE))
        # for t in range(W.shape[-2]):
        #     mask = torch.zeros(W.shape[-2]).unsqueeze(-1)
        #     mask[t] = 1
        #     pow_t = W[..., t, :].square().mean()
        #     Wnoise = EPS * torch.randn_like(W[..., [t], :]) * pow_t.sqrt()
        #     obj_t_n[i][t] = online_obj(scenario_dict, window_length, U_list, V_list, W + mask * Wnoise, A, lam, mu_r, mu_h, mu_s, t=[t])

        ## Probe V Opt
        V_list_n = []
        for Vt in V_list:
            pow = Vt.square().sum().mean()
            # win_mask = torch.tensor([1, 1, 1]).unsqueeze(-1)
            V_list_n.append(Vt + EPS * torch.randn_like(Vt) * pow.sqrt())
        obj_t_n.append(online_obj(scenario_dict, window_length, U_list, V_list_n, W, A, lam, mu_r, mu_h, mu_s))

        ### Probe U Opt
        # U_list_n = []
        # for Ut in U_list:
        #     pow = Ut.square().sum().mean()
        #     U_list_n.append(Ut + EPS * torch.randn_like(Ut) * pow.sqrt())
        # obj_t_n.append(online_obj(scenario_dict, window_length, U_list_n, V_list, W, A, lam, mu_r, mu_h, mu_s))

        ### Probe A Opt
        # obj_t_n.append(torch.zeros(A.shape[-1], dtype=FP_DTYPE))
        # for t in range(A.shape[-1]):
        #     mask = torch.zeros(A.shape[-1])
        #     mask[t] = 1
        #     An = EPS * torch.randn_like(A[..., [t]])
        #     obj_t_n[i][t] = online_obj(scenario_dict, window_length, U_list, V_list, W, A + mask * An, lam, mu_r, mu_h, mu_s, t=[t])

        plt.plot(obj_t_n[i], "r")

    obj_t_n = torch.stack(obj_t_n)
    print("worst suboptimality: (positive is bad)", torch.max(obj_t - obj_t_n))

    plt.show()
    V_list2 = [torch.ones(*batch_size, window_length, rank, dtype=FP_DTYPE), *V_list[:-1]]
    obj_t_2 = online_obj(scenario_dict, window_length, U_list, V_list2, W, A, lam, mu_r, mu_h, mu_s)
    plt.plot(obj_t_2, "g")
    """

    return U, V, W, A


def _subspace_tracking_anomography_iteration(Yh_t, R, Omegah_t, U, RU_t, V, RV_t, w_t, A_list, lam, mu_r, mu_r_prev,
                                             mu_h, mu_s,
                                             xi=0.01, admm_init=None):
    window_length = Yh_t.shape[-1]

    # w_t_new = w_t
    # tp = time.process_time()
    w_t_new = _update_w(Yh_t, Omegah_t, U, V, w_t, mu_r, mu_h)
    # print("W", time.process_time() - tp)
    # a_t_new, zK, yK = _update_a_admm(Yh_t, R, Omegah_t, U, V, w_t_new, mu_s, xi, admm_init=admm_init)
    # admm_init_new = (None, zK, yK)
    # tp = time.process_time()
    a_t_new = _update_a_stela(Yh_t, R, Omegah_t, U, V, w_t_new, mu_s)
    # print("A", time.process_time() - tp)
    admm_init_new = None
    # a_t_new = torch.zeros_like(a_t_new)

    Rah_t_new = tutils.hankel_transform(R @ torch.stack(A_list[-(window_length - 1):] + [a_t_new], dim=-1),
                                        window_length).squeeze(-1)

    # V_new, RV_t_new = V, RV_t
    # tp = time.process_time()
    V_new, RV_t_new = _update_V(Yh_t, Omegah_t, U, V, RV_t, w_t_new, w_t, Rah_t_new, lam, mu_h, mu_r, mu_r_prev)
    # print("V", time.process_time() - tp)

    # U_new, RU_t_new = U, RU_t
    # tp = time.process_time()
    U_new, RU_t_new = _update_U(Yh_t, Omegah_t, U, RU_t, V_new, w_t_new, w_t, Rah_t_new, lam, mu_h, mu_r, mu_r_prev)
    # print("U", time.process_time() - tp)

    # print("old", U.abs().sum(), V.abs().sum(), w_t.abs().sum())
    # print("new", U_new.abs().sum(), V_new.abs().sum(), w_t_new.abs().sum(), Rah_t_new.abs().sum())

    return U_new, RU_t_new, V_new, RV_t_new, w_t_new, a_t_new, admm_init_new


def _update_w(Yh_t, Omegah_t, U, V, wt, mu_r, mu_h):
    device = Yh_t.device
    rank = U.shape[-1]
    g = U.unsqueeze(-2) * V.unsqueeze(-3)  # (g_lw = g[l, w])
    ggT = g.unsqueeze(-1) * g.unsqueeze(-2)  # g[l,w]g[l,w]^T

    Omegah_t_sum = (1 + mu_h) * Omegah_t
    Omegah_t_sum[..., :, -1] -= mu_h * Omegah_t[..., :, -1]  # compute sum over LxW, but mu_h part does not act on w=W
    lhs = (Omegah_t_sum.unsqueeze(-1).unsqueeze(-1) * ggT).sum(dim=(-4, -3)) + mu_r * torch.eye(rank, device=device)
    # lhs = (Omegah_t.unsqueeze(-1).unsqueeze(-1) * ggT).sum(dim=(-4, -3)) + mu_h * (Omegah_t.unsqueeze(-1).unsqueeze(-1) * ggT)[..., :-1, :, :].sum(dim=(-4, -3)) + mu_r * torch.eye(rank)

    gb = (g[..., 1:, :] * wt.unsqueeze(-2).unsqueeze(-2)).sum(dim=(-1), keepdims=True) * g[..., :-1,
                                                                                         :]  # accounts for w+1 index shift of first g in (7)
    # gb = (g[..., :-1, :].unsqueeze(-1) * wt.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)) @ g[..., 1:, :].unsqueeze(-1)  # does not work

    rhs = ((Omegah_t * Yh_t).unsqueeze(-1) * g).sum(dim=(-3, -2)) + mu_h * (Omegah_t[..., :-1].unsqueeze(-1) * gb).sum(
        dim=(-3, -2))

    wt_new = torch.linalg.solve(lhs, rhs)

    return wt_new


"""DEPRECATED, replaced by STELA which is much faster"""
def _update_a_admm(Yh_t, R, Omegah_t, U, V, wt, mu_s, xi, admm_init=None):
    """Only supports single batch dim."""
    device = Yh_t.device

    if admm_init is not None:
        a0, z0, y0 = admm_init
    else:
        a0, z0, y0 = None, None, None

    num_flows = R.shape[-1]
    # batch_dim = Yh_t.shape[:-3]
    batch_dim = Yh_t.shape[0]
    Robs = R * Omegah_t[..., -1].unsqueeze(-1)  # (*, links, num_flows)
    q_t = Omegah_t[..., -1] * (Yh_t[..., -1] - ((U * wt.unsqueeze(-2)) @ V[..., [-1], :].mT).squeeze(-1))

    def _admm_iter(Robs, q_t, ak, zk, yk):
        """Update ak"""
        lhs = Robs.mT @ Robs + xi * torch.eye(num_flows, device=device)
        rhs = Robs.mT @ q_t.unsqueeze(-1) + xi * (zk - yk).unsqueeze(-1)
        ak = torch.linalg.solve(lhs, rhs).squeeze(-1)

        # print(ak.abs().sum(), zk.abs().sum(), yk.abs().sum())
        """update zk"""
        zk_old = zk
        zk = utils.soft_thresholding(ak + yk, mu_s / xi)

        """update yk (uk in paper)"""
        yk = yk + ak - zk

        resi_prim = torch.linalg.vector_norm(ak - zk, ord=2, dim=-1)
        resi_dual = xi * torch.linalg.vector_norm(zk - zk_old, ord=2, dim=-1)
        eps_prim = ADMM_EPS_ABS * (num_flows ** 0.5) \
                   + ADMM_EPS_REL * torch.max(torch.linalg.vector_norm(ak, ord=2, dim=-1),
                                              torch.linalg.vector_norm(zk, ord=2, dim=-1))
        eps_dual = ADMM_EPS_ABS * (num_flows ** 0.5) + ADMM_EPS_REL * torch.linalg.vector_norm(yk, ord=2, dim=-1)

        # print(resi_prim / eps_prim, resi_dual / eps_dual)
        # print(ak.abs().sum())
        finished_flag = torch.logical_and(resi_prim < eps_prim, resi_dual < eps_dual)

        return ak, zk, yk, finished_flag

    ak = a0 if a0 is not None else torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE)
    zk = z0 if z0 is not None else torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE)
    yk = y0 if y0 is not None else torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE)
    iter_args = [Robs, q_t, ak, zk, yk]

    finished = torch.zeros(batch_dim, dtype=torch.bool)
    aK = torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE)
    zK = torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE)
    yK = torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE)

    for it_idx in range(ADMM_MAX_ITER):
        # print(it_idx)
        ak_new, zk_new, yk_new, fin_flag = _admm_iter(*iter_args)
        iter_args[2:] = [ak_new, zk_new, yk_new]

        aK_temp, zK_temp, yK_temp = utils.mask_vars(fin_flag, *iter_args[2:])
        iter_args = utils.mask_vars(~fin_flag, *iter_args)
        write_res_mask = ~finished
        write_res_mask[~finished] *= fin_flag
        # aK[~finished][fin_flag], zK[~finished][fin_flag], yK[~finished][fin_flag] = aK_temp, zK_temp, yK_temp
        aK[write_res_mask], zK[write_res_mask], yK[write_res_mask] = aK_temp, zK_temp, yK_temp
        finished[~finished] = fin_flag  # updated finished vector

        # print((finished * 1.0).sum())

        if torch.all(finished):
            if DEBUG:
                print("Required {} ADMM iterations".format(it_idx + 1))
            break
    else:
        aK[~finished], zK[~finished], yK[~finished] = iter_args[2:]
        print("ADMM: Not all batches converged within {} steps.".format(ADMM_MAX_ITER))

    return aK, zK, yK


def _update_a_stela(Yh_t, R, Omegah_t, U, V, wt, mu_s, a0=None):
    """Solving LASSO using STELA because original ADMM was very slow."""
    device = Yh_t.device
    num_flows = R.shape[-1]
    # batch_dim = Yh_t.shape[:-3]
    batch_dim = Yh_t.shape[0]
    Robs = R * Omegah_t[..., -1].unsqueeze(-1)  # (*, links, num_flows)
    q_t = Omegah_t[..., -1] * (Yh_t[..., -1] - ((U * wt.unsqueeze(-2)) @ V[..., [-1], :].mT).squeeze(-1))

    def _stela_iter(Robs, q_t, ak):
        # Direction
        err = q_t.unsqueeze(-1) - Robs @ ak.unsqueeze(-1)
        ak_scale = Robs.sum(dim=-2)  # (R ** 2).mT @ (Omega ** 2)
        ak_scale_zero = ak_scale == 0
        soft_thresh_args = (ak_scale * ak + (Robs.mT @ err).squeeze(-1), mu_s)
        Bak_temp = utils.soft_thresholding(*soft_thresh_args)

        ak_scale_safezero = ak_scale + ak_scale_zero * 1
        Bak = Bak_temp / ak_scale_safezero
        Bak[ak_scale_zero] = 0

        # Step size
        proj_step = (Robs @ (Bak - ak).unsqueeze(-1)).squeeze(-1)
        # denom = Omega * proj_step
        denom = torch.sum(proj_step.square(), dim=(-1))
        nom1 = - err.squeeze(-1) * proj_step
        nom1 = torch.sum(nom1, dim=(-1))
        nom2 = utils.l1_norm(mu_s * Bak, dim=(-1)) - utils.l1_norm(mu_s * Bak, dim=(-1))
        nom = - nom1 - nom2

        denom_zero = denom == 0
        denom[denom_zero] = 1  # avoiding division by 0
        gamma = nom / denom
        gamma[denom_zero] = 0
        gamma = torch.clamp(gamma, min=0, max=1)
        if torch.any(torch.isnan(gamma)) or torch.any(torch.isnan(Bak)):
            raise RuntimeError("Gamma or BA was nan")

        # print("Gamma", gamma.mean())

        # Step
        ak_old = ak
        ak = gamma.unsqueeze(-1) * Bak + (1 - gamma.unsqueeze(-1)) * ak

        diff = torch.linalg.vector_norm(ak - ak_old, ord=2, dim=-1) / ak.shape[-1]
        finished_flag = diff < STELA_MIN_DIFF
        # print(gamma.max())

        return ak, finished_flag

    ak = a0 if a0 is not None else torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE, device=device)
    iter_args = [Robs, q_t, ak]

    finished = torch.zeros(batch_dim, dtype=torch.bool, device=device)
    aK = torch.zeros(batch_dim, num_flows, dtype=FP_DTYPE, device=device)

    for it_idx in range(STELA_MAX_ITER):
        # print(it_idx)
        ak_new, fin_flag = _stela_iter(*iter_args)
        iter_args[2] = ak_new

        aK_temp, = utils.mask_vars(fin_flag, iter_args[2])
        iter_args = utils.mask_vars(~fin_flag, *iter_args)
        write_res_mask = ~finished
        write_res_mask[~finished] *= fin_flag
        # aK[~finished][fin_flag], zK[~finished][fin_flag], yK[~finished][fin_flag] = aK_temp, zK_temp, yK_temp
        aK[write_res_mask] = aK_temp
        finished[~finished] = fin_flag  # updated finished vector

        # print((finished * 1.0).sum())

        if torch.all(finished):
            if DEBUG:
                print("Required {} STELA iterations".format(it_idx + 1))
            break
    else:
        aK[~finished] = iter_args[2]
        print("STELA: Not all batches converged within {} steps.".format(ADMM_MAX_ITER))

    return aK


def _update_U(Yh_t, Omegah_t, U, RU_t, V, wt, wt_prev, Rah_t, lam, mu_h, mu_r, mu_r_prev):
    """Implements equation (11) and (12)"""
    """Update RU somewhat unclear in paper"""
    """RU (batch, L, R, R)"""
    device = Yh_t.device
    rank = U.shape[-1]
    alpha = V * wt.unsqueeze(-2)  # (batch, W, R)
    beta = V[..., 1:, :] * wt_prev.unsqueeze(-2) - V[..., :-1, :] * wt.unsqueeze(-2)

    RU_t_new = lam * RU_t + torch.sum(
        Omegah_t.unsqueeze(-1).unsqueeze(-1) * (alpha.unsqueeze(-1) * alpha.unsqueeze(-2)).unsqueeze(-4), dim=-3) \
               + mu_h * torch.sum(
        Omegah_t[..., :-1].unsqueeze(-1).unsqueeze(-1) * (beta.unsqueeze(-1) * beta.unsqueeze(-2)).unsqueeze(-4),
        dim=-3) \
               + (mu_r - lam * mu_r_prev) * torch.eye(rank, device=device)

    rhs1 = mu_h * torch.sum(
        Omegah_t[..., :-1].unsqueeze(-1).unsqueeze(-1) * (beta.unsqueeze(-1) * beta.unsqueeze(-2)).unsqueeze(-4),
        dim=-3) \
           + (mu_r - lam * mu_r_prev) * torch.eye(rank, device=device)
    rhs1 = (rhs1 @ U.unsqueeze(-1)).squeeze(-1)

    rhs2 = Omegah_t * (Yh_t - Rah_t - torch.sum(U.unsqueeze(-2) * alpha.unsqueeze(-3), dim=-1))
    rhs2 = torch.sum(rhs2.unsqueeze(-1) * alpha.unsqueeze(-3), dim=-2)  # sum over w
    U_new = U + torch.linalg.solve(RU_t_new, (-rhs1 + rhs2))
    return U_new, RU_t_new


def _update_V(Yh_t, Omegah_t, U, V, RV_t, wt, wt_prev, Rah_t, lam, mu_h, mu_r, mu_r_prev):
    device = Yh_t.device
    win_len = V.shape[-2]
    rank = U.shape[-1]
    gamma = U * wt.unsqueeze(-2)  # (batch, L, R)
    eta = U * wt_prev.unsqueeze(-2)  # (batch, L, R)

    rhs1 = (mu_r - lam * mu_r_prev) * V

    # Vshifted = torch.zeros_like(V)
    # Vshifted[..., :-1, :] = V[..., 1:,
    #                         :]  # (batch, W, R), for index w+1, assume 0 if w > W

    # LU, pv = torch.linalg.lu_factor(RV_t_new)
    V_new = torch.zeros_like(V)
    # RV_t_new = torch.zeros_like(RV_t)
    diff_Yh_RAh = Yh_t - Rah_t
    RV_t_new = lam * RV_t + torch.sum(
        Omegah_t.unsqueeze(-1).unsqueeze(-1) * (1 + mu_h) * (gamma.unsqueeze(-1) * gamma.unsqueeze(-2)).unsqueeze(-3),
        dim=-4) \
               + (mu_r - lam * mu_r_prev) * torch.eye(rank, device=device)

    RV_t_new_LU, RV_t_new_pv = torch.linalg.lu_factor(RV_t_new)

    for w in range(win_len)[::-1]:
        if w == win_len - 1:
            mu_h_temp = 0
            Vshifted = torch.zeros_like(V)[..., w, :]
        else:
            mu_h_temp = mu_h
            Vshifted = V_new[..., w + 1, :]

        # RV_t_new[..., w, :, :] = lam * RV_t[..., w, :, :] + torch.sum(Omegah_t[..., w].unsqueeze(-1).unsqueeze(-1) * (1 + mu_h_temp) * (gamma.unsqueeze(-1) * gamma.unsqueeze(-2)), dim=-3) \
        #            + (mu_r - lam * mu_r_prev) * torch.eye(rank, device=device)

        # rhs2_w = Yh_t[..., w] - Rah_t[..., w] + mu_h_temp * torch.sum(Vshifted.unsqueeze(-2) * eta, dim=-1)  # (batch, L)
        rhs2_w = diff_Yh_RAh[..., w] + mu_h_temp * torch.sum(Vshifted.unsqueeze(-2) * eta, dim=-1)  # (batch, L)

        # rhs2_w = rhs2_w.unsqueeze(-1).unsqueeze(-1) * torch.eye(rank) - (1 + mu_h_temp) * V[..., w, :].unsqueeze(-2).unsqueeze(-1) * gamma.unsqueeze(-2)  # (batch, L, R, R)
        # rhs2_w = rhs2_w * Omegah_t[..., w].unsqueeze(-1).unsqueeze(-1)
        # rhs2_w = torch.sum(rhs2_w @ gamma.unsqueeze(-1), dim=-3)  # (batch, R, 1)
        rhs2_w = rhs2_w - (1 + mu_h_temp) * torch.sum(V[..., w, :].unsqueeze(-2) * gamma,
                                                      dim=-1)  # (batch, L)  # order of V and gamma reversed since wrong in paper...
        rhs2_w = (rhs2_w * Omegah_t[..., w]).unsqueeze(-1) * gamma
        rhs2_w = torch.sum(rhs2_w, dim=-2)  # (batch, R)

        rhs_w = - rhs1[..., w, :] + rhs2_w
        # rhs_w = rhs2_w
        # print(rhs_w.abs().sum(), rhs1[..., w, :, :].abs().sum(), rhs2_w.abs().sum(), RV_t_new[..., w, :, :].abs().sum())
        # V_new[..., w, :] = V[..., w, :] + torch.linalg.lu_solve(LU, pv, rhs_w).squeeze(-1)

        # V_new[..., w, :] = V[..., w, :] + torch.linalg.solve(RV_t_new[..., w, :, :], rhs_w)
        V_new[..., w, :] = V[..., w, :] + torch.linalg.lu_solve(RV_t_new_LU[..., w, :, :], RV_t_new_pv[..., w, :],
                                                                rhs_w.unsqueeze(-1)).squeeze(-1)

        # V_new[..., w, :] = torch.linalg.solve(RV_t_new[..., w, :, :], rhs_w).squeeze(-1)
        # print(w, V_new[..., w, :].abs().sum())

    return V_new, RV_t_new

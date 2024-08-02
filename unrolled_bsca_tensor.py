"""
Author: 
Lukas Schynol
lschynol@nt.tu-darmstadt.de
"""

from copy import deepcopy
import warnings
import time

import torch
import torch.nn as nn

import datagen
import utils as utils
from utils import RegParamMLP, _masked_mean_var
import tensor_utils as tutl

from config import DEBUG, FP_DTYPE, PREC_EPS


def obj_cpd_relaxed(scenario, X, U, V, W, A, lam, mu, nu, batch_mean=True, balanced=False):
    Y, R, Omega = datagen.nw_scenario_observation(scenario)
    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    num_time_seg = W.shape[-2]
    Y = timeseg_mat2tensor(Y, num_time_seg)
    Omega = timeseg_mat2tensor(Omega, num_time_seg)
    X = timeseg_mat2tensor(X, num_time_seg)
    # batch_size = scenario["batch_size"]
    obj = _obj_cpd_relaxed_primitive(Y, R, Omega, X, U, V, W, A, lam, mu, nu, batch_mean=batch_mean, balanced=balanced)
    return obj


def _obj_cpd_relaxed_primitive(Y, R, Omega, X, U, V, W, A, lam, mu, nu, batch_mean=True, balanced=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
    num_time_seg = W.shape[-2]
    obj = utils.frob_norm_sq(Omega * (Y - X - timeseg_mat2tensor(R @ A, num_time_seg)), dim=(-3, -2, -1)) / 2
    obj += nu * utils.frob_norm_sq(X - tutl.cpd(U, V, W), dim=(-3, -2, -1)) / 2
    obj += lam * (su * utils.frob_norm_sq(U, dim=(-2, -1)) + sv * utils.frob_norm_sq(V, dim=(
    -2, -1)) + sw * utils.frob_norm_sq(W, dim=(-2, -1))) / 2
    obj += mu * utils.l1_norm(A, dim=(-2, -1))

    if batch_mean:
        obj = obj.mean(dim=-1)  # only batch size, iteration dimension is kept

    return obj


def nw_scenario_timeseg_mat2tensor(Y, Omega, num_seg):
    # Last dimension stacks segments of the sequence in time.
    num_time_steps = Y.shape[-1]  # hopefully identical for Omega and A
    num_time_steps_short = num_time_steps // num_seg

    Ysiz0 = Y.shape[:-1]
    Ytens = Y.view(*Ysiz0, num_seg, num_time_steps_short).swapaxes(-2, -1)
    Omegatens = Omega.view(*Ysiz0, num_seg, num_time_steps_short).swapaxes(-2, -1)  # A should have the same dimension

    # Asiz0 = A.shape[:-1]
    # Atens = A.view(*Asiz0, num_seg, num_time_steps_short).swapaxes(-2, -1)  # A should have the same dimension
    return Ytens, Omegatens  # , Atens


def timeseg_mat2tensor(T, num_seg):
    num_time_steps_short = T.shape[-1] // num_seg
    Tsiz0 = T.shape[:-1]
    Ttens = T.view(*Tsiz0, num_seg, num_time_steps_short).swapaxes(-2, -1)
    return Ttens


def timeseg_tensor2mat(T):
    Tsiz0 = T.shape[:-2]
    return T.swapaxes(-2, -1).reshape(*Tsiz0, -1)


def _bsca_tensor_init_deterministic(Y, R, Omega, rank, balanced=False):
    # Init
    batch_size = Y.shape[:-3]
    num_flows = R.shape[-1]
    num_edges = Y.shape[-3]
    num_time_steps1 = Y.shape[-2]
    num_time_steps2 = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ymean = torch.sum(torch.abs(Y), dim=(-3, -2, -1)) / torch.sum(Omega, dim=(
    -3, -2, -1))  # abs of Y is dirty fix against negative values

    if not balanced:
        temp = (ymean ** 2 * rank * num_time_steps2 * num_time_steps1 * num_edges) ** (1 / 6)
        uval = temp / torch.tensor(rank * num_edges, device=Y.device).sqrt()
        vval = temp / torch.tensor(rank * num_time_steps1, device=Y.device).sqrt()
        wval = temp / torch.tensor(rank * num_time_steps2, device=Y.device).sqrt()
    else:
        uval = (ymean / rank) ** (1 / 3)
        vval = (ymean / rank) ** (1 / 3)
        wval = (ymean / rank) ** (1 / 3)
    U = torch.ones(*batch_size, num_edges, rank, dtype=torch.float, device=Y.device) * uval.unsqueeze(-1).unsqueeze(-1)
    V = torch.ones(*batch_size, num_time_steps1, rank, dtype=torch.float, device=Y.device) * vval.unsqueeze(
        -1).unsqueeze(-1)
    W = torch.ones(*batch_size, num_time_steps2, rank, dtype=torch.float, device=Y.device) * wval.unsqueeze(
        -1).unsqueeze(-1)
    A = torch.zeros(*batch_size, num_flows, num_time_steps1 * num_time_steps2, dtype=FP_DTYPE, device=Y.device)
    X = torch.zeros_like(Y)
    # X = Y.clone()

    return X, U, V, W, A


def _bsca_tensor_init_scaled_randn(Y, R, Omega, rank, sigma=0.1, balanced=False, no_tensor=False):
    # Init
    X, U, V, W, A = _bsca_tensor_init_deterministic(Y, R, Omega, rank, balanced=balanced)

    # Init
    batch_size = Y.shape[:-3]
    num_flows = R.shape[-1]
    num_edges = Y.shape[-3]
    num_time_steps1 = Y.shape[-2]
    num_time_steps2 = Y.shape[-1]

    # We expect Y to be all-positive, thus we initialize P and Q all-positive,
    # with similar Frobenius norm, and deterministic.
    ymean = torch.sum(torch.abs(Y), dim=(-3, -2, -1)) / torch.sum(Omega, dim=(
    -3, -2, -1))  # abs of Y is dirty fix against negative values

    if not balanced:
        temp = (ymean ** 2 * rank * num_time_steps2 * num_time_steps1 * num_edges) ** (1 / 6)
        uval = temp / torch.tensor(rank * num_edges, device=Y.device).sqrt()
        vval = temp / torch.tensor(rank * num_time_steps1, device=Y.device).sqrt()
        wval = temp / torch.tensor(rank * num_time_steps2, device=Y.device).sqrt()
    else:
        uval = (ymean / rank) ** (1 / 3)
        vval = (ymean / rank) ** (1 / 3)
        wval = (ymean / rank) ** (1 / 3)

    U = U + uval.unsqueeze(-1).unsqueeze(-1) * sigma * torch.randn_like(U)
    V = V + vval.unsqueeze(-1).unsqueeze(-1) * sigma * torch.randn_like(V)
    if not no_tensor:
        W = W + wval.unsqueeze(-1).unsqueeze(-1) * sigma * torch.randn_like(W)

    return X, U, V, W, A


def bsca_tensor(scenario_dict, lam, mu, nu, rank, num_time_seg=None, num_iter=10, return_im_steps=True, balanced=False,
                option=None, it1_option=None, nnf=False, normalize=False, prox_param=None, rconv=None, no_tensor=False):
    """Omega can only consist of 0 and 1 for this implementation."""
    Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
    if num_time_seg is None:
        if scenario_dict["num_time_seg"] is None or no_tensor:
            num_time_seg = 1
        else:
            num_time_seg = scenario_dict["num_time_seg"]

    # To tensor
    Y, Omega = nw_scenario_timeseg_mat2tensor(Y, Omega, num_time_seg)

    if Y.dtype != FP_DTYPE:
        Y = Y.to(FP_DTYPE)
        Omega = Omega.to(FP_DTYPE)
        R = R.to(FP_DTYPE)

    if normalize:
        _, link_var = utils._masked_mean_var(Y, mask=Omega, dim=(-2, -1))
        Y = Y / link_var.sqrt().unsqueeze(-1).unsqueeze(-1)
        R = R / link_var.sqrt().unsqueeze(-1)

    # Init
    X, U, V, W, A = _bsca_tensor_init_scaled_randn(Y, R, Omega, rank, sigma=0.1, balanced=balanced, no_tensor=no_tensor)
    # X, U, V, W, A = _bsca_tensor_iteration(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=balanced)
    # A initialized as matrix, not tensor, careful

    if nu is None:
        nu = 0

    if rconv is not None:
        obj = _obj_cpd_relaxed_primitive(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                         batch_mean=True, balanced=balanced)

    if return_im_steps:
        X_list = [X]
        U_list = [U]
        V_list = [V]
        W_list = [W]
        A_list = [A]

    # print(U.square().mean(dim=(-2, -1)), V.square().mean(dim=(-2, -1)), W.square().mean(dim=(-2, -1)))
    for i in range(num_iter):
        if i == 0 and (it1_option is not None):
            curr_option = it1_option
        else:
            curr_option = option
        if DEBUG:
            print("Iteration {}".format(i + 1))
        if curr_option == "rlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                           balanced=balanced, nnf=nnf, no_W_update=no_tensor)
        elif curr_option == "nrlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration(Y, R, Omega, X, U, V, W, A, lam, mu,
                                                                       prox_param=prox_param, balanced=balanced, no_W_update=no_tensor)
        elif curr_option == "nrlxapx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_apx(Y, R, Omega, X, U, V, W, A, lam, mu,
                                                                           balanced=balanced, no_W_update=no_tensor)
        elif curr_option == "grprlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_grprlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                              balanced=balanced, no_W_update=no_tensor)
        elif curr_option == "grprlx2":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_grprlx2(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                               balanced=balanced, no_W_update=no_tensor)
        elif curr_option == "rlx4x":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx4x(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, no_W_update=no_tensor)
        elif curr_option == "rlx2x":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx2x(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, no_W_update=no_tensor)
        else:
            raise ValueError
        # print(U_new.square().mean(dim=(-2, -1)), V_new.square().mean(dim=(-2, -1)), W_new.square().mean(dim=(-2, -1)))

        if return_im_steps:
            X_list.append(X_new)
            U_list.append(U_new)
            V_list.append(V_new)
            W_list.append(W_new)
            A_list.append(A_new)

        if rconv is not None:
            obj_new = _obj_cpd_relaxed_primitive(Y, R, Omega, X_new, U_new, V_new, W_new, A_new, lam, mu, nu,
                                                 batch_mean=True, balanced=balanced)
            rcrit = (obj - obj_new).abs() / obj
            if DEBUG:
                print("obj={}, relcrit={}".format(obj_new, rcrit))
            if rcrit < rconv:
                print("Converged after {} iterations.".format(i + 1))
                break

            obj = obj_new

        X, U, V, W, A = X_new, U_new, V_new, W_new, A_new

    if return_im_steps:
        X_list = timeseg_tensor2mat(torch.stack(X_list))
        U_list = torch.stack(U_list)
        V_list = torch.stack(V_list)
        W_list = torch.stack(W_list)
        A_list = torch.stack(A_list)
        return X_list, U_list, V_list, W_list, A_list
    else:
        return X, U, V, W, A


def _bsca_tensor_iteration_rlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=False, nnf=False, no_W_update=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
        # print(su, sw, sn)

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    # print(_obj_relaxed_primitive(Y, R, Omega, X, U, V, W, A, lam, mu, nu))
    X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err, nnf=nnf)
    # print(_obj_relaxed_primitive(Y, R, Omega, X_new, U, V, W, A, lam, mu, nu))
    # if torch.all(X == 0):
    #     X_new = _bsca_tensor_update_X(Y, R, Omega, U, V, W, A, nu, err)
    # else:
    #     X_new = X
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U, V, W), dim=(-3, -2, -1)).mean())
    U_new = _bsca_tensor_update_U_rlx(X_new, V, W, su * lam / nu)
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V, W), dim=(-3, -2, -1)).mean())
    # U_new = U
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V, W), dim=(-3, -2, -1)).mean())
    V_new = _bsca_tensor_update_V_rlx(X_new, U_new, W, sv * lam / nu)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W), dim=(-3, -2, -1)).mean())
    # V_new = V
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W), dim=(-3, -2, -1)).mean())
    if not no_W_update:
        W_new = _bsca_tensor_update_W_rlx(X_new, U_new, V_new, sw * lam / nu)
    else:
        W_new = W
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W_new), dim=(-3, -2, -1)).mean())
    # W_new = W
    # A_new = A
    A_new = _bsca_tensor_update_A(Y, R, Omega, X_new, A, mu, err)

    return X_new, U_new, V_new, W_new, A_new


def _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err, nnf=False):
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    nu_usq = nu.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    divisor = Omega ** 2 + torch.ones_like(Omega) * nu_usq
    X_new = (Omega * err + nu_usq * tutl.cpd(U, V, W)) / divisor

    if nnf:
        X_new = torch.clamp(X_new, min=0)
    return X_new


def _bsca_tensor_update_U_rlx(X, V, W, regpar, returnQ=False, nnf=False):
    rank = V.shape[-1]

    lhs = (W.mT @ W) * (V.mT @ V)
    # print("U", torch.linalg.cond(lhs))
    regularizer = regpar
    # if torch.any((PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer) > 0):
    #     print("!!!U!!!", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0])).max(), regularizer)
    # print("U", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer).squeeze() < 0)
    # print(lhs, lhs.abs().max(dim=-1)[0].max(dim=-1)[0])
    regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]),
                                regularizer)
    lhs = lhs + torch.eye(rank, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)

    X1 = tutl.tensor2mat_unfold(X, 1)
    Q1 = tutl.khatri_rao(W, V)
    rhs = X1 @ Q1
    # try:
    U_new = torch.linalg.solve(lhs, rhs, left=False)
    # except:
    #     print("This shouldn't occur anymore")

    if nnf:
        # U_new = torch.clamp(U_new, min=0)
        raise NotImplementedError
    if returnQ:
        return U_new, Q1
    else:
        return U_new


def _bsca_tensor_update_V_rlx(X, U, W, regpar, returnQ=False, nnf=False):
    rank = U.shape[-1]

    lhs = (W.mT @ W) * (U.mT @ U)
    # print("V", torch.linalg.cond(lhs))
    regularizer = regpar
    # if torch.any((PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer) > 0):
    #     print("!!!V!!!", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0])).max(), regularizer)
    regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]),
                                regularizer)
    lhs = lhs + torch.eye(rank, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)

    X2 = tutl.tensor2mat_unfold(X, 2)
    Q2 = tutl.khatri_rao(W, U)
    rhs = X2 @ Q2
    # try:
    V_new = torch.linalg.solve(lhs, rhs, left=False)
    # except:
    #     print("This shouldn't occur anymore")
    if nnf:
        V_new = torch.clamp(V_new, min=0)
    if returnQ:
        return V_new, Q2
    else:
        return V_new


def _bsca_tensor_update_W_rlx(X, U, V, regpar, returnQ=False, nnf=False):
    rank = V.shape[-1]

    lhs = (V.mT @ V) * (U.mT @ U)
    # print("W", torch.linalg.cond(lhs))
    regularizer = regpar
    # if torch.any((PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer) > 0):
    #     print("!!!W!!!", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0])).max(), regularizer)
    regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]),
                                regularizer)
    lhs = lhs + torch.eye(rank, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)

    X3 = tutl.tensor2mat_unfold(X, 3)
    Q3 = tutl.khatri_rao(V, U)
    rhs = X3 @ Q3
    # try:
    W_new = torch.linalg.solve(lhs, rhs, left=False)
    # except:
    #     print("This shouldn't occur anymore")
    if nnf:
        W_new = torch.clamp(W_new, min=0)
    if returnQ:
        return W_new, Q3
    else:
        return W_new


def _bsca_tensor_update_A(Y, R, Omega, X, A, mu, err=None, return_gamma=False):
    """All matrices have leading batch dimension.
        err = Omega * (Y - (R @ A))"""
    if err is None:
        num_time_seg = Y.shape[-1]
        full_err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg) - X)
    else:
        full_err = err - Omega * X

    full_err = tutl.tensor2mat_unfold(full_err, 1)
    Omega = tutl.tensor2mat_unfold(Omega, 1)
    # Direction
    A_scale = (R * R).transpose(-2, -1) @ (Omega ** 2)
    A_scale_zero = A_scale == 0
    soft_thresh_args = (A_scale * A + R.mT @ (full_err * Omega), mu)
    BA_temp = utils.soft_thresholding(*soft_thresh_args)

    A_scale_safezero = A_scale + A_scale_zero * 1
    BA = BA_temp / A_scale_safezero
    BA[
        A_scale_zero] = 0  # set direction to 0 where A does not receive information (no connected link measurements for particular a)

    # Step size
    proj_step = R @ (BA - A) * Omega
    # denom = Omega * proj_step
    denom = torch.sum(proj_step.square(), dim=(-2, -1))
    nom1 = - full_err * proj_step
    nom1 = torch.sum(nom1, dim=(-2, -1))
    nom2 = (utils.l1_norm(mu * BA, dim=(-2, -1)) - utils.l1_norm(mu * A, dim=(-2, -1)))
    nom = - nom1 - nom2

    denom_zero = denom == 0
    denom[denom_zero] = 1  # avoiding division by 0
    gamma = nom / denom
    gamma[denom_zero] = 0
    gamma = torch.clamp(gamma, min=0, max=1)
    if torch.any(torch.isnan(gamma)) or torch.any(torch.isnan(BA)):
        raise RuntimeError("Gamma or BA was nan")  # debugging

    # print("Gamma", gamma.mean())

    # Step
    A_new = gamma.unsqueeze(-1).unsqueeze(-1) * BA + (1 - gamma.unsqueeze(-1).unsqueeze(-1)) * A

    if return_gamma:
        return A_new, gamma
    else:
        return A_new


class BSCATensorUnrolled(nn.Module):
    lam_val = []
    mu_val = []
    nu_val = []

    def __init__(self,
                 num_layers,
                 rank,
                 param_nw=None,  # "modewise"
                 shared_weights=False,
                 it1_option=None,
                 balanced=False,
                 normalize=False,
                 prox=False,
                 skipA=0,
                 no_tensor=False,  # reduces algorithm to basically matrix factorization version
                 init=None,  # placeholder does nothing right now
                 layer_param=None):
        super().__init__()
        if layer_param is None:
            layer_param = {}

        self.num_layers = num_layers
        self.same_weights_across_layers = shared_weights
        self.rank = rank
        self.param_nw = param_nw
        self.it1_option = it1_option
        self.balanced = balanced
        self.normalize = normalize
        self.skipA = skipA
        self.prox = prox
        self.no_tensor = no_tensor

        if self.prox:
            self.prox_param = nn.Parameter(torch.tensor([-1, 0], dtype=FP_DTYPE))
        else:
            self.prox_param = None

        if layer_param:
            self.layer_param = layer_param
        else:
            self.layer_param = {}
        self.layers = []

        # if self.param_nw:
        #     iter_class = BSCAUnrolledIteration_ParamNW
        # else:
        #     iter_class = BSCATensorUnrolledIteration
        assert(not (self.param_nw == "modewise_sw" and shared_weights))
        if self.param_nw:
            if self.param_nw == "modewise" or self.param_nw == "modewise_sw":
                iter_class = BSCATensorUnrolledIteration_ModeWise
            else:
                raise ValueError
        else:
            iter_class = BSCATensorUnrolledIteration

        if not shared_weights:
            for i_layer in range(self.num_layers):
                layer_param_temp = deepcopy(self.layer_param)
                if self.it1_option and i_layer == 0:
                    layer_param_temp["option"] = self.it1_option
                # if i_layer < self.skipA:
                #     layer_param_temp["init_val"] = [0, 1e2, 2]
                self.layers.append(iter_class(**layer_param_temp))
        else:
            if self.it1_option:
                layer_param_temp = deepcopy(self.layer_param)
                layer_param_temp["option"] = self.it1_option
                self.layers.append(iter_class(**layer_param_temp))
                self.layers.extend([iter_class(**self.layer_param)] * (num_layers - 1))
            else:
                self.layers = [iter_class(**self.layer_param)] * num_layers

        ### Weights of adaptive DF and MU are shared
        if self.param_nw == "modewise_sw":
            for ly in range(1, self.num_layers):
                self.layers[ly].datafit_nw = self.layers[ly].datafit_nw
                self.layers[ly].mu_nw = self.layers[ly].mu_nw

        self.layers = nn.ModuleList(self.layers)

    def forward(self, scenario_dict, num_time_seg=None, return_im_steps=True, normalize=False):
        Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)
        if num_time_seg is None:
            if scenario_dict["num_time_seg"] is None or self.no_tensor:
                num_time_seg = 1
            else:
                num_time_seg = scenario_dict["num_time_seg"]

        # To tensor
        Y, Omega = nw_scenario_timeseg_mat2tensor(Y, Omega, num_time_seg)
        if Y.dtype != FP_DTYPE:
            Y = Y.to(FP_DTYPE)
            Omega = Omega.to(FP_DTYPE)
            R = R.to(FP_DTYPE)

        if normalize or self.normalize:
            _, link_var = _masked_mean_var(Y, mask=Omega, dim=(-2, -1))
            Y = Y / link_var.sqrt().unsqueeze(-1).unsqueeze(-1)
            R = R / link_var.sqrt().unsqueeze(-1)

        # Init
        X, U, V, W, A = _bsca_tensor_init_scaled_randn(Y, R, Omega, self.rank, sigma=0.1, balanced=self.balanced, no_tensor=self.no_tensor)
        # print(U.square().mean(dim=(-2, -1)), V.square().mean(dim=(-2, -1)), W.square().mean(dim=(-2, -1)))

        if return_im_steps:
            X_list = [X]
            U_list = [U]
            V_list = [V]
            W_list = [W]
            A_list = [A]

        self.lam_val = []
        self.mu_val = []
        self.nu_val = []

        if self.prox_param is not None:
            prox_param = torch.exp(self.prox_param)
        else:
            prox_param = None

        if return_im_steps:
            X_list = [X]
            U_list = [U]
            V_list = [V]
            W_list = [W]
            A_list = [A]

        if self.param_nw:
            nw_out = {"datafit_weight_log": torch.zeros_like(Y), "mu_log": torch.zeros_like(A)}
        else:
            pass

        for l in range(self.num_layers):
            if DEBUG:
                print("layer", l)
            if self.param_nw:
                X_new, U_new, V_new, W_new, A_new, nw_out = self.layers[l](Y, R, Omega, X, U, V, W, A,
                                                                           prev_nw_out=nw_out,
                                                                           balanced=self.balanced,
                                                                           A_update=(l >= self.skipA),
                                                                           no_tensor=self.no_tensor)
            else:
                X_new, U_new, V_new, W_new, A_new = self.layers[l](Y, R, Omega, X, U, V, W, A,
                                                                   balanced=self.balanced,
                                                                   prox_param=prox_param, A_update=(l >= self.skipA),
                                                                   no_tensor=self.no_tensor)
            # print(U_new.square().mean(dim=(-2, -1)), V_new.square().mean(dim=(-2, -1)), W_new.square().mean(dim=(-2, -1)))

            if return_im_steps:
                X_list.append(X_new)
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
                A_list.append(A_new)

            X, U, V, W, A = X_new, U_new, V_new, W_new, A_new

            self.lam_val.append(self.layers[l].lam_val)
            self.mu_val.append(self.layers[l].mu_val)
            self.nu_val.append(self.layers[l].nu_val)

        self.lam_val = torch.stack(self.lam_val)
        self.mu_val = torch.stack(self.mu_val)
        self.nu_val = torch.stack(self.nu_val)

        if return_im_steps:
            X_list = timeseg_tensor2mat(torch.stack(X_list))
            U_list = torch.stack(U_list)
            V_list = torch.stack(V_list)
            W_list = torch.stack(W_list)
            A_list = torch.stack(A_list)
            return X_list, U_list, V_list, W_list, A_list
        else:
            return X, U, V, W, A


class BSCATensorUnrolledIteration(nn.Module):
    lam_val = None
    mu_val = None
    nu_val = None

    def __init__(self, init_val=None, skip_connections=False, option="nrlx", weightings=None):
        super().__init__()
        # self.relaxed = relaxed
        if init_val is None:
            init_val = [0, -3, 2]
        self.option = option
        # self.balanced = balanced
        self.lam_log = nn.Parameter(torch.tensor(init_val[0], dtype=torch.float))
        self.mu_log = nn.Parameter(torch.tensor(init_val[1], dtype=torch.float))
        self.nu_log = nn.Parameter(torch.tensor(init_val[2], dtype=torch.float))

    def forward(self, Y, R, Omega, X, U, V, W, A, balanced=False, prox_param=None, A_update=True, no_tensor=False):
        lam = torch.exp(self.lam_log)
        mu = torch.exp(self.mu_log)
        nu = torch.exp(self.nu_log)

        self.lam_val = lam.detach().clone()
        self.mu_val = mu.detach().clone()
        self.nu_val = nu.detach().clone()

        # if self.relaxed:
        #     X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=self.balanced)
        # else:
        #     X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=self.balanced)
        # print(Y.device, R.device, X.device, U.device, lam.device)
        # exit()

        if self.option == "nrlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration(Y, R, Omega, X, U, V, W, A, lam, mu,
                                                                       prox_param=prox_param, balanced=balanced,
                                                                       A_update=A_update, no_W_update=no_tensor)
        elif self.option == "nrlxapx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_apx(Y, R, Omega, X, U, V, W, A, lam, mu,
                                                                           balanced=balanced, no_W_update=no_tensor)
        elif self.option == "rlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                           balanced=balanced, no_W_update=no_tensor)
        elif self.option == "grprlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_grprlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                              balanced=balanced, no_W_update=no_tensor)
        elif self.option == "rlx4x":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx4x(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, no_W_update=no_tensor)
        elif self.option == "rlx2x":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx2x(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, no_W_update=no_tensor)
        elif self.option == "rlx2x+nnf":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx2x(Y, R, Omega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, no_W_update=no_tensor, nnf=True)
        else:
            raise ValueError
        return X_new, U_new, V_new, W_new, A_new

    def get_regularization_parameters(self, clone=False):
        param_dict = {}
        # if self.two_lambda:
        #     lam = (torch.exp(self.lam_log), torch.exp(self.lam2_log))
        # else:
        #     lam = torch.exp(self.lam_log)
        #
        # mu = torch.exp(self.mu_log)
        if clone:
            param_dict["lam"] = torch.exp(self.lam_log).detach().clone()
            param_dict["mu"] = torch.exp(self.mu_log).detach().clone()
            param_dict["nu"] = torch.exp(self.mu_log).detach().clone()
        else:
            param_dict["lam"] = torch.exp(self.lam_log)
            param_dict["mu"] = torch.exp(self.mu_log)
            param_dict["nu"] = torch.exp(self.mu_log)

        return param_dict


class BSCATensorUnrolledIteration_ModeWise(nn.Module):
    lam_val = None
    mu_val = None
    nu_val = None

    DF_OPT_1 = "1dyn1ly"

    MU_OPT_0 = "d1ly"
    # MU_OPT_1 = "1dyn1ly"
    # MU_OPT_2 = "1dyn1lyi"

    DF_OPT_F1 = "f1ly"
    DF_OPT_F2 = "f2ly"
    MU_OPT_F1 = "f1ly"
    MU_OPT_F2 = "f2ly"

    # MAX_DFIT = 6
    # MAX_MU_LOG = 6

    def __init__(self,
                 init_val=None,
                 skip_connections=False,
                 option="nrlx",
                 datafit_options=None,
                 mu_options=None,
                 batch_norm=False,
                 two_nu=False,
                 soft_clip_max=6,
                 ):
        super().__init__()
        # self.relaxed = relaxed
        if init_val is None:
            init_val = [-2, -5, 0]
        self.option = option
        self.batch_norm = batch_norm
        self.two_nu = two_nu

        self.MAX_DFIT = soft_clip_max
        self.MAX_MU_LOG = soft_clip_max
        # self.balanced = balanced
        self.lam_log = nn.Parameter(torch.tensor(init_val[0], dtype=torch.float))
        # self.mu_log = nn.Parameter(torch.tensor(init_val[1], dtype=torch.float))
        if two_nu:
            self.nu_log = nn.Parameter(torch.tensor([init_val[2], init_val[2]], dtype=torch.float))
        else:
            self.nu_log = nn.Parameter(torch.tensor(init_val[2], dtype=torch.float))

        if datafit_options is None:
            self.datafit_options = []
        else:
            self.datafit_options = datafit_options

        if mu_options is None:
            self.mu_options = []
        else:
            self.mu_options = mu_options

        self.option = option

        if self.DF_OPT_1 in self.datafit_options:
            datafit_ffun = _datafit_features_dynamic
            self.datafit_nw = RegParamMLP([8, 1], batch_norm_input=self.batch_norm)
        elif self.DF_OPT_F1 in self.datafit_options:
            # Bias = False gets rid of ambiguity of objective function (if no soft clipping were used)
            self.datafit_nw = RegParamMLP([7, 1], batch_norm_input=self.batch_norm, init="uniform_small", bias=False)
            datafit_ffun = _datafit_embedding_dyn_final
        elif self.DF_OPT_F2 in self.datafit_options:
            # Bias = False gets rid of ambiguity of objective function (if no soft clipping were used)
            self.datafit_nw = RegParamMLP([7, 7, 1], batch_norm_input=self.batch_norm, init="uniform_small", bias=False)
            datafit_ffun = _datafit_embedding_dyn_final

        if "fmask" in self.datafit_options:
            self.datafit_ffun = lambda *args: (datafit_ffun(*args) * self.datafit_options["fmask"].to(args[0].device))
        elif len(self.datafit_options) > 0:
            self.datafit_ffun = datafit_ffun
        else:
            self.datafit_nw = None

        if self.MU_OPT_0 in self.mu_options:
            self.mu_nw = RegParamMLP([32, 1], batch_norm_input=self.batch_norm)
            mu_ffun = _mu_features_dyn
        # elif self.MU_OPT_1 in self.mu_options:
        #     self.mu_nw = RegParamMLP([32, 1], batch_norm_input=self.batch_norm)
        #     mu_ffun = _mu_features_dynamic
        # elif self.MU_OPT_2 in self.mu_options:
        #     self.mu_nw = RegParamMLP([32, 1], batch_norm_input=self.batch_norm)
        #     mu_ffun = _mu_features_dynamic_i
        elif self.MU_OPT_F1 in self.mu_options:
            self.mu_nw = RegParamMLP([13, 1], batch_norm_input=self.batch_norm, init="uniform_small")
            mu_ffun = _mu_embedding_dyn_final
        elif self.MU_OPT_F2 in self.mu_options:
            self.mu_nw = RegParamMLP([13, 13, 1], batch_norm_input=self.batch_norm, init="uniform_small")
            mu_ffun = _mu_embedding_dyn_final
        else:
            self.mu_log = nn.Parameter(torch.tensor(init_val[1], dtype=torch.float))
            self.mu_nw = None

        if "fmask" in self.mu_options:
            self.mu_ffun = lambda *args: (mu_ffun(*args) * self.mu_options["fmask"].to(args[0].device))
        elif len(self.mu_options) > 0:
            self.mu_ffun = mu_ffun

    def forward(self, Y, R, Omega, X, U, V, W, A, prev_nw_out=None, balanced=False, A_update=True, no_tensor=False):
        nw_out = {}

        prox_param = None  # does not help
        lam = torch.exp(self.lam_log)
        self.lam_val = lam.detach().clone()
        nu = torch.exp(self.nu_log)
        self.nu_val = nu.detach().clone()
        if self.two_nu:
            self.nu_val = self.nu_val.mean()

        if self.DF_OPT_1 in self.datafit_options \
                or self.DF_OPT_F1 in self.datafit_options or self.DF_OPT_F2 in self.datafit_options:
            prev_datafit_weight_log = prev_nw_out["datafit_weight_log"]
            max_fit = self.MAX_DFIT
            datafit_fvec = self.datafit_ffun(Y, R, Omega, X, A, prev_datafit_weight_log)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            datafit_weight_log = max_fit * torch.tanh(datafit_weight_log / max_fit)
            datafit_weight = torch.exp(datafit_weight_log)  # soft clipping
            WOmega = Omega * datafit_weight
            nw_out["datafit_weight_log"] = datafit_weight_log
        else:
            WOmega = Omega

        if self.MU_OPT_1 in self.mu_options or self.MU_OPT_2 in self.mu_options or self.MU_OPT_0 in self.mu_options \
                or self.MU_OPT_F1 in self.mu_options or self.MU_OPT_F2 in self.mu_options:
            max_mulog = self.MAX_MU_LOG
            mu_log_prev = prev_nw_out["mu_log"]
            mu_fvec = self.mu_ffun(Y, R, WOmega, X, A, mu_log_prev)
            mu_log = self.mu_nw(mu_fvec)[..., 0] - max_mulog  # initial bias = -5 to prevent all-zero outputs
            mu_log = max_mulog * torch.tanh(mu_log / max_mulog)  # soft clipping
            nw_out["mu_log"] = mu_log
            mu = torch.exp(mu_log)  # -5 to start with small values, otherwise might get stuck
            self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
        else:
            mu = torch.exp(self.mu_log)
            self.mu_val = mu.detach().clone()

        # print(Y.abs().sum(), Omega.abs().sum(), WOmega.abs().sum(), R.abs().sum(), torch.any(R.sum(dim=-1) == 0), lam)

        if self.option == "nrlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration(Y, R, WOmega, X, U, V, W, A, lam, mu,
                                                                       prox_param=prox_param, balanced=balanced,
                                                                       A_update=A_update, no_W_update=no_tensor)
        elif self.option == "rlx":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx(Y, R, WOmega, X, U, V, W, A, lam, mu, nu,
                                                                           balanced=balanced, no_W_update=no_tensor)
        elif self.option == "rlx+nnf":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx(Y, R, WOmega, X, U, V, W, A, lam, mu, nu,
                                                                           balanced=balanced, nnf=True, no_W_update=no_tensor)
        elif self.option == "rlx2x":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx2x(Y, R, WOmega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, no_W_update=no_tensor)
        elif self.option == "rlx2x+nnf":
            X_new, U_new, V_new, W_new, A_new = _bsca_tensor_iteration_rlx2x(Y, R, WOmega, X, U, V, W, A, lam, mu, nu,
                                                                             balanced=balanced, nnf=True, no_W_update=no_tensor)
        else:
            raise ValueError
        return X_new, U_new, V_new, W_new, A_new, nw_out

    def get_regularization_parameters(self, clone=False):
        param_dict = {}
        # if self.two_lambda:
        #     lam = (torch.exp(self.lam_log), torch.exp(self.lam2_log))
        # else:
        #     lam = torch.exp(self.lam_log)
        #
        # mu = torch.exp(self.mu_log)
        if clone:
            param_dict["lam"] = torch.exp(self.lam_log).detach().clone()
            param_dict["mu"] = torch.exp(self.mu_log).detach().clone()
            param_dict["nu"] = torch.exp(self.mu_log).detach().clone()
        else:
            param_dict["lam"] = torch.exp(self.lam_log)
            param_dict["mu"] = torch.exp(self.mu_log)
            param_dict["nu"] = torch.exp(self.mu_log)

        return param_dict


def _bsca_tensor_iteration(Y, R, Omega, X, U, V, W, A, lam, mu, prox_param=None, balanced=False, A_update=True, no_W_update=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
        # print(sw, lam)

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))

    prox = _compute_prox_regularizer(prox_param, err, Omega, U, V, W)
    # exit()
    U_new = _bsca_tensor_update_U(Y, R, Omega, U, V, W, A, su * lam, err, prox=prox)
    # exit()
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    # utils._fm_corr(U_new[0])
    prox = _compute_prox_regularizer(prox_param, err, Omega, U_new, V, W)
    V_new = _bsca_tensor_update_V(Y, R, Omega, U_new, V, W, A, sv * lam, err, prox=prox)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    # utils._fm_corr(V_new[0])
    prox = _compute_prox_regularizer(prox_param, err, Omega, U_new, V_new, W)
    if not no_W_update:
        W_new = _bsca_tensor_update_W(Y, R, Omega, U_new, V_new, W, A, sw * lam, err, prox=prox)
    else:
        W_new = W
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)
    # utils._fm_corr(W_new[0])
    X_new = tutl.cpd(U_new, V_new, W_new)
    if A_update:
        A_new = _bsca_tensor_update_A(Y, R, Omega, X_new, A, mu, err)
    else:
        A_new = A

    return X_new, U_new, V_new, W_new, A_new


def _compute_prox_regularizer(prox_param, err, Omega, U, V, W):
    if prox_param is None:
        prox_reg = None
    else:
        prox_reg = prox_param[0] + prox_param[1] * utils.frob_norm_sq(Omega * (err - tutl.cpd(U, V, W)),
                                                                      dim=(-3, -2, -1)) ** 2 \
                   / utils.frob_norm_sq(err, dim=(-3, -2, -1)) ** 2
    return prox_reg


def _bsca_tensor_update_U(Y, R, Omega, U, V, W, A, lam, err=None, prox=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = V.shape[-1]
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    Q1 = tutl.khatri_rao(W, V)
    Omega1 = tutl.tensor2mat_unfold(Omega, 1)
    regularizer = lam
    rhs = tutl.tensor2mat_unfold(Omega * err, 1) @ Q1

    if prox is not None:
        rhs = rhs + prox.unsqueeze(-1).unsqueeze(-1) * U
        regularizer = regularizer + prox

    rhs = rhs.unsqueeze(-1)  # (*batch, rowsU, rank, 1)
    lhs = Q1.mT.unsqueeze(-3) @ ((Omega1.unsqueeze(-1) ** 2) * Q1.unsqueeze(-3))
    regularizer = regularizer.unsqueeze(-1)
    # if torch.any((PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer) > 0):
    #     print("!!!U!!!", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0])).max(), regularizer)
    # regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), regularizer)
    lhs = lhs + torch.eye(rank, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)
    # b = 1
    # r = 150
    # l = 2
    # print("r {}".format(r))
    # print("b {}".format(b))
    # print("l {}".format(l))
    # time.sleep(0.5)
    # # lhs1 = lhs[..., :r, :r]
    # # rhs1 = rhs[..., :r, [0]]
    # # c = 1.0*2.5 + 1
    # lhs1 = torch.randn(l, r, r, dtype=torch.float32)
    # rhs1 = torch.randn(l, r, 1, dtype=torch.float32)
    # # C = lhs1 @ rhs1
    # # print(rhs1.shape)
    U_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)
    # exit()
    # U_new = utils.lstsq(lhs, rhs).squeeze(-1)

    return U_new


def _bsca_tensor_update_V(Y, R, Omega, U, V, W, A, lam, err=None, prox=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = W.shape[-1]
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    Q2 = tutl.khatri_rao(W, U)
    Omega2 = tutl.tensor2mat_unfold(Omega, 2)
    regularizer = lam
    rhs = tutl.tensor2mat_unfold(Omega * err, 2) @ Q2

    if prox is not None:
        rhs = rhs + prox.unsqueeze(-1).unsqueeze(-1) * V
        regularizer = regularizer + prox

    rhs = rhs.unsqueeze(-1)  # (*batch, rowsU, rank, 1)
    lhs = Q2.mT.unsqueeze(-3) @ ((Omega2.unsqueeze(-1) ** 2) * Q2.unsqueeze(-3))
    regularizer = regularizer.unsqueeze(-1)
    # if torch.any((PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer) > 0):
    #     print("!!!V!!!", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0])).max(), regularizer)
    # regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), regularizer)
    lhs = lhs + torch.eye(rank, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)
    V_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)
    # V_new = utils.lstsq(lhs, rhs).squeeze(-1)

    return V_new


def _bsca_tensor_update_W(Y, R, Omega, U, V, W, A, lam, err=None, prox=None):
    """All matrices have leading batch dimension.
    err = Omega * (Y - (R @ A))"""
    rank = U.shape[-1]
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    Q3 = tutl.khatri_rao(V, U)
    Omega3 = tutl.tensor2mat_unfold(Omega, 3)
    regularizer = lam
    rhs = tutl.tensor2mat_unfold(Omega * err, 3) @ Q3

    if prox is not None:
        rhs = rhs + prox.unsqueeze(-1).unsqueeze(-1) * W
        regularizer = regularizer + prox

    rhs = rhs.unsqueeze(-1)  # (*batch, rowsU, rank, 1)
    lhs = Q3.mT.unsqueeze(-3) @ ((Omega3.unsqueeze(-1) ** 2) * Q3.unsqueeze(-3))
    regularizer = regularizer.unsqueeze(-1)
    # if torch.any((PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]) - regularizer) > 0):
    #     print("!!!W!!!", (PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0])).max(), regularizer)
    # regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), regularizer)
    lhs = lhs + torch.eye(rank, device=lhs.device) * regularizer.unsqueeze(-1).unsqueeze(-1)
    W_new = torch.linalg.solve(lhs, rhs, left=True).squeeze(-1)
    # W_new = utils.lstsq(lhs, rhs).squeeze(-1)

    return W_new


def _bsca_tensor_iteration_apx(Y, R, Omega, X, U, V, W, A, lam, mu, balanced=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
        # print(sw, lam)

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))

    U_new = _bsca_tensor_update_U_apx(Y, R, Omega, U, V, W, A, su * lam, err)
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    V_new = _bsca_tensor_update_V_apx(Y, R, Omega, U_new, V, W, A, sv * lam, err)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    W_new = _bsca_tensor_update_W_apx(Y, R, Omega, U_new, V_new, W, A, sw * lam, err)
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)
    X_new = tutl.cpd(U_new, V_new, W_new)
    A_new = _bsca_tensor_update_A(Y, R, Omega, X_new, A, mu, err)

    return X_new, U_new, V_new, W_new, A_new


def _bsca_tensor_update_U_apx(Y, R, Omega, U, V, W, A, lam, err=None, nnf=False):
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    Omega = tutl.tensor2mat_unfold(Omega, 1)
    Q1 = tutl.khatri_rao(W, V)
    err_fit = Omega * (tutl.tensor2mat_unfold(err, 1) - U @ Q1.mT)

    temp = err_fit @ Q1
    C = (Omega.unsqueeze(-1) * (Q1 ** 2).unsqueeze(-3)).sum(dim=-2)  # sum over size of other modes
    U_m = (temp + C * U) / (C + lam.unsqueeze(-1).unsqueeze(-1))

    if nnf:
        U_m = torch.clamp(U_m, min=0)

    # Step Size
    U_dir = U_m - U
    M_dir = U_dir @ Q1.mT
    a1 = (err_fit * M_dir).sum(dim=(-2, -1))
    a2 = - lam * (U * U_dir).sum(dim=(-2, -1))
    b1 = utils.frob_norm_sq(Omega * M_dir, dim=(-2, -1))
    b2 = lam * utils.frob_norm_sq(U_dir, dim=(-2, -1))
    b2[b2 == 0] = 1  # fixes if Udir = 0
    gamma = torch.clamp((a1 + a2) / (b1 + b2), min=0, max=1)
    # gamma = (a1 + a2) / (b1 + b2)

    U_new = U + gamma.unsqueeze(-1).unsqueeze(-1) * U_dir

    return U_new


def _bsca_tensor_update_V_apx(Y, R, Omega, U, V, W, A, lam, err=None, nnf=False):
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    Omega = tutl.tensor2mat_unfold(Omega, 2)
    Q2 = tutl.khatri_rao(W, U)
    err_fit = Omega * (tutl.tensor2mat_unfold(err, 2) - V @ Q2.mT)

    temp = err_fit @ Q2
    C = (Omega.unsqueeze(-1) * (Q2 ** 2).unsqueeze(-3)).sum(dim=-2)  # sum over size of other modes
    V_m = (temp + C * V) / (C + lam.unsqueeze(-1).unsqueeze(-1))

    if nnf:
        V_m = torch.clamp(V_m, min=0)

    # Step Size
    V_dir = V_m - V
    M_dir = V_dir @ Q2.mT
    a1 = (err_fit * M_dir).sum(dim=(-2, -1))
    a2 = - lam * (V * V_dir).sum(dim=(-2, -1))
    b1 = utils.frob_norm_sq(Omega * M_dir, dim=(-2, -1))
    b2 = lam * utils.frob_norm_sq(V_dir, dim=(-2, -1))
    b2[b2 == 0] = 1  # fixes if Vdir = 0
    gamma = torch.clamp((a1 + a2) / (b1 + b2), min=0, max=1)
    # gamma = (a1 + a2) / (b1 + b2)

    V_new = V + gamma.unsqueeze(-1).unsqueeze(-1) * V_dir

    return V_new


def _bsca_tensor_update_W_apx(Y, R, Omega, U, V, W, A, lam, err=None, nnf=False):
    if err is None:
        num_time_seg = Y.shape[-1]
        err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))
    else:
        err = err

    Omega = tutl.tensor2mat_unfold(Omega, 3)
    Q3 = tutl.khatri_rao(V, U)
    err_fit = Omega * (tutl.tensor2mat_unfold(err, 3) - W @ Q3.mT)

    temp = err_fit @ Q3
    C = (Omega.unsqueeze(-1) * (Q3 ** 2).unsqueeze(-3)).sum(dim=-2)  # sum over size of other modes
    W_m = (temp + C * W) / (C + lam.unsqueeze(-1).unsqueeze(-1))

    if nnf:
        W_m = torch.clamp(W_m, min=0)

    # Step Size
    W_dir = W_m - W
    M_dir = W_dir @ Q3.mT
    a1 = (err_fit * M_dir).sum(dim=(-2, -1))
    a2 = - lam * (W * W_dir).sum(dim=(-2, -1))
    b1 = utils.frob_norm_sq(Omega * M_dir, dim=(-2, -1))
    b2 = lam * utils.frob_norm_sq(W_dir, dim=(-2, -1))
    b2[b2 == 0] = 1  # fixes if Wdir = 0
    gamma = torch.clamp((a1 + a2) / (b1 + b2), min=0, max=1)
    # gamma = (a1 + a2) / (b1 + b2)

    W_new = W + gamma.unsqueeze(-1).unsqueeze(-1) * W_dir

    return W_new


def _bsca_tensor_iteration_grprlx(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
    # print(U.square().mean(), V.square().mean(), W.square().mean())

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))

    U_new, X_new = _bsca_tensor_update_UX_grprlx(Y, R, Omega, X, U, V, W, A, su * lam, nu, err)
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    # X_new = X
    # U_new = U
    # V_new = V
    # W_new = W
    V_new, X_new = _bsca_tensor_update_VX_grprlx(Y, R, Omega, X_new, U_new, V, W, A, sv * lam, nu, err)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    W_new, X_new = _bsca_tensor_update_WX_grprlx(Y, R, Omega, X_new, U_new, V_new, W, A, sw * lam, nu, err)
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)
    A_new = _bsca_tensor_update_A(Y, R, Omega, X_new, A, mu, err)

    return X_new, U_new, V_new, W_new, A_new


def _bsca_tensor_iteration_grprlx2(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
    # print(U.square().mean(), V.square().mean(), W.square().mean())

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))

    U_new, X_new = _bsca_tensor_update_UX_grprlx(Y, R, Omega, X, U, V, W, A, su * lam, nu, err)
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    # X_new = X
    # U_new = U
    # V_new = V
    # W_new = W
    V_new, X_new = _bsca_tensor_update_VX_grprlx(Y, R, Omega, X_new, U_new, V, W, A, sv * lam, nu, err)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    W_new, X_new = _bsca_tensor_update_WX_grprlx(Y, R, Omega, X_new, U_new, V_new, W, A, sw * lam, nu, err)
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)
    A_new, X_new = _bsca_tensor_update_AX_grprlx(Y, R, Omega, X_new, U_new, V_new, W_new, A, mu, nu, err)

    return X_new, U_new, V_new, W_new, A_new



def _bsca_tensor_update_UX_grprlx(Y, R, Omega, X, U, V, W, A, lam, nu, err=None):
    # num_time_seg = Y.shape[-1]

    X_m = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err)
    U_m, Q1 = _bsca_tensor_update_U_rlx(X, V, W, lam / nu, returnQ=True)

    X_dir = (X_m - X)
    U_dir = (U_m - U)

    a1 = (err * X_dir).sum(dim=(-3, -2, -1))
    b1 = utils.frob_norm_sq(X_dir * Omega, dim=(-3, -2, -1))

    X_dir_mat = tutl.tensor2mat_unfold(X_dir, 1)
    Btemp = (X_dir_mat - U_dir @ Q1.mT)
    a2 = -nu * ((tutl.tensor2mat_unfold(X, 1) - U @ Q1.mT) * Btemp).sum(dim=(-2, -1))
    b2 = nu * utils.frob_norm_sq(Btemp, dim=(-2, -1))

    a3 = - lam * (U * U_dir).sum(dim=(-2, -1))
    b3 = lam * utils.frob_norm_sq(U_dir, dim=(-2, -1))
    gamma = torch.clamp((a1 + a2 + a3) / (b1 + b2 + b3), 0, 1)
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)

    X_new = X + gamma.unsqueeze(-1) * X_dir
    U_new = U + gamma * U_dir
    # print("gamU", gamma, (a1 + a2 + a3) / (b1 + b2 + b3), (a1, b1), (a2, b2), (a3, b3))

    return U_new, X_new


def _bsca_tensor_update_VX_grprlx(Y, R, Omega, X, U, V, W, A, lam, nu, err=None):
    # num_time_seg = Y.shape[-1]

    X_m = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err)
    V_m, Q2 = _bsca_tensor_update_V_rlx(X, U, W, lam / nu, returnQ=True)

    X_dir = (X_m - X)
    V_dir = (V_m - V)

    a1 = (err * X_dir).sum(dim=(-3, -2, -1))
    b1 = utils.frob_norm_sq(X_dir * Omega, dim=(-3, -2, -1))

    X_dir_mat = tutl.tensor2mat_unfold(X_dir, 2)
    Btemp = (X_dir_mat - V_dir @ Q2.mT)
    a2 = -nu * ((tutl.tensor2mat_unfold(X, 2) - V @ Q2.mT) * Btemp).sum(dim=(-2, -1))
    b2 = nu * utils.frob_norm_sq(Btemp, dim=(-2, -1))

    a3 = - lam * (V * V_dir).sum(dim=(-2, -1))
    b3 = lam * utils.frob_norm_sq(V_dir, dim=(-2, -1))
    gamma = torch.clamp((a1 + a2 + a3) / (b1 + b2 + b3), 0, 1)
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)

    X_new = X + gamma.unsqueeze(-1) * X_dir
    V_new = V + gamma * V_dir
    # print("gamV", gamma, (a1 + a2 + a3) / (b1 + b2 + b3))

    return V_new, X_new


def _bsca_tensor_update_WX_grprlx(Y, R, Omega, X, U, V, W, A, lam, nu, err=None):
    # num_time_seg = Y.shape[-1]

    X_m = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err)
    W_m, Q3 = _bsca_tensor_update_W_rlx(X, U, V, lam / nu, returnQ=True)

    X_dir = (X_m - X)
    W_dir = (W_m - W)

    a1 = (err * X_dir).sum(dim=(-3, -2, -1))
    b1 = utils.frob_norm_sq(X_dir * Omega, dim=(-3, -2, -1))

    X_dir_mat = tutl.tensor2mat_unfold(X_dir, 3)
    Btemp = (X_dir_mat - W_dir @ Q3.mT)
    a2 = -nu * ((tutl.tensor2mat_unfold(X, 3) - W @ Q3.mT) * Btemp).sum(dim=(-2, -1))
    b2 = nu * utils.frob_norm_sq(Btemp, dim=(-2, -1))

    a3 = - lam * (W * W_dir).sum(dim=(-2, -1))
    b3 = lam * utils.frob_norm_sq(W_dir, dim=(-2, -1))
    gamma = torch.clamp((a1 + a2 + a3) / (b1 + b2 + b3), 0, 1)
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)

    X_new = X + gamma.unsqueeze(-1) * X_dir
    W_new = W + gamma * W_dir
    # print("gamW", gamma, (a1 + a2 + a3) / (b1 + b2 + b3))

    return W_new, X_new


def _bsca_tensor_update_AX_grprlx(Y, R, Omega, X, U, V, W, A, mu, nu, err=None):
    """All matrices have leading batch dimension.
        err = Omega * (Y - (R @ A))"""
    num_time_seg = Y.shape[-1]
    if err is None:
        full_err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg) - X)
    else:
        full_err = Omega * (err - X)

    X_m = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err)

    full_err_mat = tutl.tensor2mat_unfold(full_err, 1)
    Omega = tutl.tensor2mat_unfold(Omega, 1)
    # Direction
    A_scale = (R * R).transpose(-2, -1) @ Omega.type(torch.float)
    A_scale_zero = A_scale == 0
    soft_thresh_args = (A_scale * A + R.mT @ full_err_mat, mu.unsqueeze(-1).unsqueeze(-1))
    A_m_temp = utils.soft_thresholding(*soft_thresh_args)

    A_scale_safezero = A_scale + A_scale_zero * 1
    A_m = A_m_temp / A_scale_safezero
    A_m[A_scale_zero] = 0

    # Step size
    Xdir = X_m - X
    Adir = A_m - A

    B = Xdir + timeseg_mat2tensor(R @ Adir, num_time_seg)

    a1 = (full_err * B).sum(dim=(-3, -2, -1))
    b1 = utils.frob_norm_sq(B, dim=(-3, -2, -1))
    a2 = - nu * ((X - tutl.cpd(U, V, W)) * Xdir).sum(dim=(-3, -2, -1))
    b2 = nu * utils.frob_norm_sq(Xdir, dim=(-3, -2, -1))
    a3 = - mu * (utils.l1_norm(A_m, dim=(-2, -1)) - utils.l1_norm(A, dim=(-2, -1)))

    gamma = torch.clamp((a1 + a2 + a3) / (b1 + b2), min=0, max=1)
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)

    A_new = A + gamma * Adir
    X_new = X + gamma.unsqueeze(-1) * Xdir

    return A_new, X_new


def _bsca_tensor_iteration_rlx4x(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=False, no_W_update=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
    # print(U.square().mean(), V.square().mean(), W.square().mean())

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))

    X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu, err)
    U_new = _bsca_tensor_update_U_rlx(X_new, V, W, su * lam / nu)
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V, W), dim=(-3, -2, -1)).mean())
    # U_new = U
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V, W), dim=(-3, -2, -1)).mean())
    X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U_new, V, W, A, nu, err)
    V_new = _bsca_tensor_update_V_rlx(X_new, U_new, W, sv * lam / nu)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W), dim=(-3, -2, -1)).mean())
    # V_new = V
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W), dim=(-3, -2, -1)).mean())
    if not no_W_update:
        X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U_new, V_new, W, A, nu, err)
        W_new = _bsca_tensor_update_W_rlx(X_new, U_new, V_new, sw * lam / nu)
    else:
        W_new = W
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)

    X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U_new, V_new, W_new, A, nu, err)
    A_new = _bsca_tensor_update_A(Y, R, Omega, X_new, A, mu, err)

    return X_new, U_new, V_new, W_new, A_new


def _bsca_tensor_iteration_rlx2x(Y, R, Omega, X, U, V, W, A, lam, mu, nu, balanced=False, nnf=False, no_W_update=False):
    if not balanced:
        su, sv, sw = 1, 1, 1
    else:
        sn = (Y.shape[-1] * Y.shape[-2] * Y.shape[-3]) ** (1 / 3)
        su = sn / (Y.shape[-3])
        sv = sn / (Y.shape[-2])
        sw = sn / (Y.shape[-1])
    # print(U.square().mean(), V.square().mean(), W.square().mean())

    if nu.ndim > 0 and len(nu) == 2:
        nu0 = nu[0]
        nu1 = nu[1]
    else:
        nu0 = nu
        nu1 = nu

    num_time_seg = Y.shape[-1]
    err = Omega * (Y - timeseg_mat2tensor(R @ A, num_time_seg))

    X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U, V, W, A, nu0, err, nnf=nnf)
    U_new = _bsca_tensor_update_U_rlx(X_new, V, W, su * lam / nu0)
    if balanced:
        U_new, V, W = balance_cpd(U_new, V, W)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V, W), dim=(-3, -2, -1)).mean())
    # U_new = U
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V, W), dim=(-3, -2, -1)).mean())
    V_new = _bsca_tensor_update_V_rlx(X_new, U_new, W, sv * lam / nu0)
    if balanced:
        U_new, V_new, W = balance_cpd(U_new, V_new, W)
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W), dim=(-3, -2, -1)).mean())
    # V_new = V
    # print(utils.frob_norm_sq(X_new - tutl.cpd(U_new, V_new, W), dim=(-3, -2, -1)).mean())
    if not no_W_update:
        W_new = _bsca_tensor_update_W_rlx(X_new, U_new, V_new, sw * lam / nu0)
    else:
        W_new = W
    if balanced:
        U_new, V_new, W_new = balance_cpd(U_new, V_new, W_new)

    X_new = _bsca_tensor_update_X_rlx(Y, R, Omega, U_new, V_new, W_new, A, nu1, err, nnf=nnf)
    A_new = _bsca_tensor_update_A(Y, R, Omega, X_new, A, mu, err)

    return X_new, U_new, V_new, W_new, A_new


def balance_cpd(U, V, W):
    # avoids numerical problems
    usc = U.square().mean(dim=(-2, -1))
    vsc = V.square().mean(dim=(-2, -1))
    wsc = W.square().mean(dim=(-2, -1))

    sc = (usc * vsc * wsc) ** (1 / 6)

    U_new = U * (sc / usc.sqrt()).unsqueeze(-1).unsqueeze(-1)
    V_new = V * (sc / vsc.sqrt()).unsqueeze(-1).unsqueeze(-1)
    W_new = W * (sc / wsc.sqrt()).unsqueeze(-1).unsqueeze(-1)

    return U_new, V_new, W_new


def _datafit_features_dynamic(Y, R, Omega, X, A, datafit_weight_log):
    """Returns (batch_size, E, T, num_features) feature tensor.
    Features: Y var per link, err var per link, Y var per t, err var per t, num_routes per link, prev_dataweight
    Omega - expected to have all elements in {0, 1}"""
    EPS = 1e-4
    shape = Y.shape
    Omega = Omega > 0

    err = Y - X - timeseg_mat2tensor(R @ A, shape[-1])

    # mode1
    mode1_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-2, -1])[1] + EPS).log()
    mode1_var2 = (_masked_mean_var(err, mask=Omega, dim=[-2, -1])[1] + EPS).log()
    mode1_feat = torch.stack([mode1_var1, mode1_var2], dim=0).unsqueeze(-1).unsqueeze(-1).expand(2, *shape)

    # mode2
    mode2_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-3, -1])[1] + EPS).log()
    mode2_var2 = (_masked_mean_var(err, mask=Omega, dim=[-3, -1])[1] + EPS).log()
    mode2_feat = torch.stack([mode2_var1, mode2_var2], dim=0).unsqueeze(-2).unsqueeze(-1).expand(2, *shape)

    # mode3
    mode3_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-3, -2])[1] + EPS).log()
    mode3_var2 = (_masked_mean_var(err, mask=Omega, dim=[-3, -2])[1] + EPS).log()
    mode3_feat = torch.stack([mode3_var1, mode3_var2], dim=0).unsqueeze(-2).unsqueeze(-2).expand(2, *shape)

    routing_feat = (R.sum(dim=-1) + EPS).log().unsqueeze(-1).unsqueeze(-1).expand(*shape).unsqueeze(0)

    # lam_feat = lam_log.expand(*shape).unsqueeze(0)
    # omega_feat = Omega.expand(*shape).unsqueeze(0)
    datafit_weight_log = datafit_weight_log.unsqueeze(0)
    datafit_feat = torch.movedim(torch.cat([mode1_feat, mode2_feat, mode3_feat, routing_feat, datafit_weight_log]), 0,
                                 -1)  # feature dim to last

    # features are nonnegative, but may be 0

    return datafit_feat


def _datafit_embedding_dyn_final(Y, R, Omega, X, A, datafit_weight_log):
    """Returns (batch_size, E, T, num_features) feature tensor.
    Features: Y var per link, err var per link, Y var per t, err var per t, num_routes per link, prev_dataweight
    Omega - expected to have all elements in {0, 1}"""
    """7 features"""

    EPS = 1e-4
    shape = Y.shape
    Omega = Omega > 0

    err = Y - X - timeseg_mat2tensor(R @ A, shape[-1])

    # mode1
    mode1_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-2, -1])[1] + EPS).log()
    mode1_var2 = (_masked_mean_var(err, mask=Omega, dim=[-2, -1])[1] + EPS).log()
    mode1_feat = torch.stack([mode1_var1, mode1_var2], dim=0).unsqueeze(-1).unsqueeze(-1).expand(2, *shape)

    # mode2
    mode2_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-3, -1])[1] + EPS).log()
    mode2_var2 = (_masked_mean_var(err, mask=Omega, dim=[-3, -1])[1] + EPS).log()
    mode2_feat = torch.stack([mode2_var1, mode2_var2], dim=0).unsqueeze(-2).unsqueeze(-1).expand(2, *shape)

    # mode3
    mode3_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-3, -2])[1] + EPS).log()
    mode3_var2 = (_masked_mean_var(err, mask=Omega, dim=[-3, -2])[1] + EPS).log()
    mode3_feat = torch.stack([mode3_var1, mode3_var2], dim=0).unsqueeze(-2).unsqueeze(-2).expand(2, *shape)

    routing_feat = (R.sum(dim=-1) + EPS).log().unsqueeze(-1).unsqueeze(-1).expand(*shape).unsqueeze(0)

    # lam_feat = lam_log.expand(*shape).unsqueeze(0)
    # omega_feat = Omega.expand(*shape).unsqueeze(0)
    # datafit_weight_log = datafit_weight_log.unsqueeze(0)
    datafit_feat = torch.movedim(torch.cat([mode1_feat, mode2_feat, mode3_feat, routing_feat]), 0,
                                 -1)  # feature dim to last

    # features are nonnegative, but may be 0

    return datafit_feat


# def _mu_features_dynamic(Y, R, Omega, X, A, mu_log):
#     """Returns (batch_size, F, T, num_features) feature tensor.
#     Omega - expected to have all elements in {0, 1}"""
#     """Set 22 features for now"""
#     """
#     Features
#     0-2: VAR PER FLOW projflow, err1, err2
#     3-5: MAX PER FLOW projflow, err1, err2
#     6-8: VAR PER TIME projflow, err1, err2
#     9-11: MAX PER TIME projflow, err1, err2
#     12: num_observations
#     13-14: err1 pos and neg
#     15-16: err2 pos and neg
#     17-18: A pos and neg
#     19: mulogprev
#     """
#     EPS = 1e-4
#     HIGH = 1e18
#
#     num_seg = Omega.shape[-1]
#
#     A_scale = timeseg_mat2tensor((R ** 2).mT @ timeseg_tensor2mat(Omega), num_seg)
#     A_noobs = A_scale == 0
#     A_scale_safezero = A_scale + A_noobs * 1
#     proj_flow = timeseg_mat2tensor(R.mT @ timeseg_tensor2mat(Y * Omega), num_seg) / A_scale_safezero
#     proj_err = timeseg_mat2tensor(R.mT @ timeseg_tensor2mat((Y - X) * Omega), num_seg) / A_scale_safezero
#     # proj_err2 = R.mT @ ((Y - X - R @ A) * Omega) / A_scale_safezero
#     observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension
#
#     # Ytime_var = _masked_mean_var(Y, mask=Omega, dim=[-2])[1]
#
#     # time_var = proj_flow.var(dim=-2)
#     # Ytime_var_log = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()
#
#     Aabs = timeseg_mat2tensor(A.abs(), num_seg)
#     A = timeseg_mat2tensor(A, num_seg)
#     shape = A.shape
#
#     proj_err_abs = proj_err.abs()
#
#     proj_flow_mode1_var = proj_flow.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     proj_flow_mode2_var = proj_flow.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     proj_flow_mode3_var = proj_flow.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#
#     proj_err_mode1_var = proj_err.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     proj_err_mode2_var = proj_err.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     proj_err_mode3_var = proj_err.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#
#     A_mode1_var = A.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     A_mode2_var = A.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     A_mode3_var = A.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#
#     mode1_proj_flow_var = proj_flow_mode1_var.log() / 2
#     mode1_proj_err_var = proj_flow_mode1_var.log() / 2
#     mode1_Avar = A_mode1_var.log() / 2
#     mode1_scpf_Amax = (torch.amax(Aabs / (proj_flow_mode2_var * proj_flow_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
#         -1).unsqueeze(-1) + EPS).log()
#     mode1_scpe_Amax = (torch.amax(Aabs / (proj_err_mode2_var * proj_err_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
#         -1).unsqueeze(-1) + EPS).log()
#     mode1_scpf_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_flow_mode2_var * proj_flow_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
#                     -1).unsqueeze(-1) + EPS).log()
#     mode1_scpe_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_err_mode2_var * proj_err_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
#                     -1).unsqueeze(-1) + EPS).log()
#
#     mode1_feat = torch.stack([mode1_Avar, mode1_proj_flow_var, mode1_proj_err_var,
#                               mode1_scpf_Amax, mode1_scpe_Amax,
#                               mode1_scpf_proj_err_max, mode1_scpe_proj_err_max], dim=0).expand(7, *shape)
#
#     mode2_proj_flow_var = proj_flow_mode2_var.log() / 2
#     mode2_proj_err_var = proj_flow_mode2_var.log() / 2
#     mode2_Avar = A_mode2_var.log() / 2
#     mode2_scpf_Amax = (torch.amax(Aabs / (proj_flow_mode1_var * proj_flow_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
#         -2).unsqueeze(-1) + EPS).log()
#     mode2_scpe_Amax = (torch.amax(Aabs / (proj_err_mode1_var * proj_err_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
#         -2).unsqueeze(-1) + EPS).log()
#     mode2_scpf_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_flow_mode1_var * proj_flow_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
#                     -2).unsqueeze(-1) + EPS).log()
#     mode2_scpe_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
#                     -2).unsqueeze(-1) + EPS).log()
#
#     mode2_feat = torch.stack([mode2_Avar, mode2_proj_flow_var, mode2_proj_err_var,
#                               mode2_scpf_Amax, mode2_scpe_Amax,
#                               mode2_scpf_proj_err_max, mode2_scpe_proj_err_max], dim=0).expand(7, *shape)
#
#     mode3_proj_flow_var = proj_flow_mode3_var.log() / 2
#     mode3_proj_err_var = proj_flow_mode3_var.log() / 2
#     mode3_Avar = A_mode3_var.log() / 2
#     mode3_scpf_Amax = (torch.amax(Aabs / (proj_flow_mode1_var * proj_flow_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
#         -2).unsqueeze(-2) + EPS).log()
#     mode3_scpe_Amax = (torch.amax(Aabs / (proj_err_mode1_var * proj_err_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
#         -2).unsqueeze(-2) + EPS).log()
#     mode3_scpf_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_flow_mode1_var * proj_flow_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
#                     -2).unsqueeze(-2) + EPS).log()
#     mode3_scpe_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
#                     -2).unsqueeze(-2) + EPS).log()
#
#     mode3_feat = torch.stack([mode3_Avar, mode3_proj_flow_var, mode3_proj_err_var,
#                               mode3_scpf_Amax, mode3_scpe_Amax,
#                               mode3_scpf_proj_err_max, mode3_scpe_proj_err_max], dim=0).expand(7, *shape)
#
#     padding = torch.zeros_like(mu_log).unsqueeze(0)
#     # Unfolding back to matrix since A is usually stored as matrix
#     observations = timeseg_tensor2mat(observations)
#     mode1_feat = timeseg_tensor2mat(mode1_feat)
#     mode2_feat = timeseg_tensor2mat(mode2_feat)
#     mode3_feat = timeseg_tensor2mat(mode3_feat)
#
#     mu_log = mu_log.unsqueeze(0)
#     mu_feat = torch.movedim(torch.cat([mode1_feat, *([padding] * 3),
#                                        mode2_feat, *([padding] * 3),
#                                        mode3_feat, *([padding] * 3),
#                                        observations, mu_log]), 0, -1)  # feature dim to last, 20 feats
#
#     return mu_feat


def _mu_features_dyn(Y, R, Omega, X, A, mu_log):
    """Returns (batch_size, F, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
    """Set 23 features for now"""
    """
    Features
    0-2: VAR PER FLOW projflow, err1, err2
    3-5: MAX PER FLOW projflow, err1, err2
    6-8: VAR PER TIME projflow, err1, err2
    9-11: MAX PER TIME projflow, err1, err2
    12: num_observations
    13-14: err1 pos and neg
    15-16: err2 pos and neg
    17-18: A pos and neg
    19: mulogprev
    """
    EPS = 1e-4
    HIGH = 1e18

    num_seg = Omega.shape[-1]

    A_scale = timeseg_mat2tensor((R ** 2).mT @ timeseg_tensor2mat(Omega), num_seg)
    A_noobs = A_scale == 0
    A_scale_safezero = A_scale + A_noobs * 1
    proj_flow = timeseg_mat2tensor(R.mT @ timeseg_tensor2mat(Y * Omega), num_seg) / A_scale_safezero
    proj_err = timeseg_mat2tensor(R.mT @ timeseg_tensor2mat((Y - X) * Omega), num_seg) / A_scale_safezero
    observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension

    # Ytime_var = _masked_mean_var(Y, mask=Omega, dim=[-2])[1]

    # time_var = proj_flow.var(dim=-2)
    # Ytime_var_log = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()

    Aabs = timeseg_mat2tensor(A.abs(), num_seg)
    A = timeseg_mat2tensor(A, num_seg)
    shape = A.shape

    proj_err_abs = proj_err.abs()

    proj_flow_mode1_var = proj_flow.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
    proj_flow_mode2_var = proj_flow.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
    proj_flow_mode3_var = proj_flow.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2

    proj_err_mode1_var = proj_err.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
    proj_err_mode2_var = proj_err.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
    proj_err_mode3_var = proj_err.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2

    A_mode1_var = A.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
    A_mode2_var = A.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
    A_mode3_var = A.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2

    mode1_proj_flow_var = proj_flow_mode1_var.log() / 2
    mode1_proj_err_var = proj_flow_mode1_var.log() / 2
    mode1_Avar = A_mode1_var.log() / 2

    mode1_Amax = (torch.amax(Aabs, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS).log()
    mode1_sca_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(-1).unsqueeze(
        -1) + EPS).log()
    mode1_proj_err_max = (torch.amax(proj_err_abs, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS).log()
    mode1_scpe_proj_err_max = (
                torch.amax(proj_err_abs / (proj_err_mode2_var * proj_err_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
                    -1).unsqueeze(-1) + EPS).log()

    mode1_feat = torch.stack([mode1_proj_flow_var, mode1_proj_err_var, mode1_Avar,
                              mode1_Amax, mode1_sca_Amax,
                              mode1_proj_err_max, mode1_scpe_proj_err_max], dim=0).expand(7, *shape)

    mode2_proj_flow_var = proj_flow_mode2_var.log() / 2
    mode2_proj_err_var = proj_flow_mode2_var.log() / 2
    mode2_Avar = A_mode2_var.log() / 2

    mode2_Amax = (torch.amax(Aabs, dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS).log()
    mode2_sca_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(-2).unsqueeze(
        -1) + EPS).log()
    mode2_proj_err_max = (torch.amax(proj_err_abs, dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS).log()
    mode2_scpe_proj_err_max = (
                torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
                    -2).unsqueeze(-1) + EPS).log()

    mode2_feat = torch.stack([mode2_proj_flow_var, mode2_proj_err_var, mode2_Avar,
                              mode2_Amax, mode2_sca_Amax,
                              mode2_proj_err_max, mode2_scpe_proj_err_max], dim=0).expand(7, *shape)

    mode3_proj_flow_var = proj_flow_mode3_var.log() / 2
    mode3_proj_err_var = proj_flow_mode3_var.log() / 2
    mode3_Avar = A_mode3_var.log() / 2

    mode3_Amax = (torch.amax(Aabs, dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS).log()
    mode3_sca_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-3, -2)).unsqueeze(-2).unsqueeze(
        -2) + EPS).log()
    mode3_proj_err_max = (torch.amax(proj_err_abs, dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS).log()
    mode3_scpe_proj_err_max = (
                torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
                    -2).unsqueeze(-2) + EPS).log()

    mode3_feat = torch.stack([mode3_proj_flow_var, mode3_proj_err_var, mode3_Avar,
                              mode3_Amax, mode3_sca_Amax,
                              mode3_proj_err_max, mode3_scpe_proj_err_max], dim=0).expand(7, *shape)

    padding = torch.zeros_like(mu_log).unsqueeze(0)
    # Unfolding back to matrix since A is usually stored as matrix
    observations = timeseg_tensor2mat(observations)
    mode1_feat = timeseg_tensor2mat(mode1_feat)
    mode2_feat = timeseg_tensor2mat(mode2_feat)
    mode3_feat = timeseg_tensor2mat(mode3_feat)

    mu_log = mu_log.unsqueeze(0)
    mu_feat = torch.movedim(torch.cat([mode1_feat, *([padding] * 3),
                                       mode2_feat, *([padding] * 3),
                                       mode3_feat, *([padding] * 3),
                                       observations, mu_log]), 0, -1)  # feature dim to last, 20 feats

    return mu_feat


def _mu_embedding_dyn_final(Y, R, Omega, X, A, mu_log):
    """Returns (batch_size, F, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
    """13 features"""
    EPS = 1e-4

    num_seg = Omega.shape[-1]

    Omega2 = Omega * Omega
    A_scale = timeseg_mat2tensor((R ** 2).mT @ timeseg_tensor2mat(Omega2), num_seg)
    A_noobs = A_scale == 0
    A_scale_safezero = A_scale + A_noobs * 1
    # proj_flow = timeseg_mat2tensor(R.mT @ timeseg_tensor2mat(Y * Omega), num_seg) / A_scale_safezero
    proj_err = timeseg_mat2tensor(R.mT @ timeseg_tensor2mat((Y - X) * Omega2), num_seg) / A_scale_safezero
    observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension

    # Ytime_var = _masked_mean_var(Y, mask=Omega, dim=[-2])[1]

    # time_var = proj_flow.var(dim=-2)
    # Ytime_var_log = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()

    Aabs = timeseg_mat2tensor(A.abs(), num_seg)
    A = timeseg_mat2tensor(A, num_seg)
    shape = A.shape

    proj_err_abs = proj_err.abs()

    # proj_flow_mode1_var = proj_flow.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
    # proj_flow_mode2_var = proj_flow.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
    # proj_flow_mode3_var = proj_flow.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2

    proj_err_mode1_var = proj_err.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
    proj_err_mode2_var = proj_err.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
    proj_err_mode3_var = proj_err.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2

    A_mode1_var = A.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
    A_mode2_var = A.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
    A_mode3_var = A.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2

    # mode1_proj_flow_var = proj_flow_mode1_var.log() / 2
    # mode1_proj_err_var = proj_flow_mode1_var.log() / 2
    mode1_Avar = A_mode1_var.log() / 2

    # mode1_Amax = (torch.amax(Aabs, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS).log()
    mode1_sca_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(-1).unsqueeze(
        -1) + EPS).log()
    mode1_proj_err_max = (torch.amax(proj_err_abs, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS).log()
    mode1_scpe_proj_err_max = (
                torch.amax(proj_err_abs / (proj_err_mode2_var * proj_err_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
                    -1).unsqueeze(-1) + EPS).log()

    mode1_feat = torch.stack([mode1_Avar, mode1_sca_Amax,
                              mode1_proj_err_max, mode1_scpe_proj_err_max], dim=0).expand(4, *shape)

    # mode2_proj_flow_var = proj_flow_mode2_var.log() / 2
    # mode2_proj_err_var = proj_flow_mode2_var.log() / 2
    mode2_Avar = A_mode2_var.log() / 2

    # mode2_Amax = (torch.amax(Aabs, dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS).log()
    mode2_sca_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(-2).unsqueeze(
        -1) + EPS).log()
    mode2_proj_err_max = (torch.amax(proj_err_abs, dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS).log()
    mode2_scpe_proj_err_max = (
                torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
                    -2).unsqueeze(-1) + EPS).log()

    mode2_feat = torch.stack([mode2_Avar, mode2_sca_Amax,
                              mode2_proj_err_max, mode2_scpe_proj_err_max], dim=0).expand(4, *shape)

    # mode3_proj_flow_var = proj_flow_mode3_var.log() / 2
    # mode3_proj_err_var = proj_flow_mode3_var.log() / 2
    mode3_Avar = A_mode3_var.log() / 2

    # mode3_Amax = (torch.amax(Aabs, dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS).log()
    mode3_sca_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-3, -2)).unsqueeze(-2).unsqueeze(
        -2) + EPS).log()
    mode3_proj_err_max = (torch.amax(proj_err_abs, dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS).log()
    mode3_scpe_proj_err_max = (
                torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
                    -2).unsqueeze(-2) + EPS).log()

    mode3_feat = torch.stack([mode3_Avar, mode3_sca_Amax,
                              mode3_proj_err_max, mode3_scpe_proj_err_max], dim=0).expand(4, *shape)

    # padding = torch.zeros_like(mu_log).unsqueeze(0)
    # Unfolding back to matrix since A is usually stored as matrix
    observations = timeseg_tensor2mat(observations)
    mode1_feat = timeseg_tensor2mat(mode1_feat)
    mode2_feat = timeseg_tensor2mat(mode2_feat)
    mode3_feat = timeseg_tensor2mat(mode3_feat)

    # mu_log = mu_log.unsqueeze(0)
    mu_feat = torch.movedim(torch.cat([mode1_feat,
                                       mode2_feat,
                                       mode3_feat,
                                       observations]), 0, -1)  # feature dim to last, 20 feats

    return mu_feat


# def _mu_features_dynamic_i(Y, R, Omega, X, A, mu_log):
#     """Returns (batch_size, F, T, num_features) feature tensor.
#     Omega - expected to have all elements in {0, 1}"""
#     """Set 32 features for now"""
#
#     EPS = 1e-4
#     HIGH = 1e15
#
#     num_seg = Omega.shape[-1]
#     # shape = Omega.shape
#
#     Y = timeseg_tensor2mat(Y)
#     Omega = timeseg_tensor2mat(Omega)
#     X = timeseg_tensor2mat(X)
#
#     A_scale = (R ** 2).mT @ Omega
#     A_noobs = A_scale == 0
#     A_scale_safezero = A_scale + A_noobs * 1
#     proj_err = R.mT @ ((Y - X) * Omega) / A_scale_safezero
#
#     # proj_err2 = R.mT @ ((Y - X - R @ A) * Omega) / A_scale_safezero
#     # observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension
#
#     # Ytime_var = _masked_mean_var(Y, mask=Omega, dim=[-2])[1]
#
#     # time_var = proj_flow.var(dim=-2)
#     # Ytime_var_log = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()
#
#     # Aabs = timeseg_mat2tensor(A.abs(), num_seg)
#     # A = timeseg_mat2tensor(A, num_seg)
#     # shape = A.shape
#     # proj_err_abs = proj_err.abs()
#
#     observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension
#     # edge_corresp = R.mT.unsqueeze(-1) * Omega.unsqueeze(-3)  # dim (F, E, T)
#     # unobserved = edge_corresp == 0
#     edge_corresp = R.mT.unsqueeze(-1)  # dim (F, E, 1)
#     no_edge_corresp = edge_corresp == 0
#     reliable = torch.amin((R.mT.unsqueeze(-1) * Omega.unsqueeze(-3)) + no_edge_corresp * HIGH, dim=-2) != 0
#
#     flows_corresp_to_ano = (Y * Omega).unsqueeze(-3) / (
#                 edge_corresp + EPS)  # eps to avoid div by 0, division if routing matrix in [0, 1] instead of {0, 1}
#     # min_corresp_flow_ind = torch.min(flows_corresp_to_ano.abs() + unobserved * HIGH, dim=-2, keepdim=True)[1]  # get only indices
#     min_corresp_flow_ind = torch.min(flows_corresp_to_ano.abs() + no_edge_corresp * HIGH, dim=-2, keepdim=True)[
#         1]  # get only indices
#     min_corresp_flow = torch.gather(flows_corresp_to_ano, -2, min_corresp_flow_ind).squeeze(-2)  # collapse edge dim
#
#     err_corresp_to_ano = ((Y - X) * Omega).unsqueeze(-3) / (edge_corresp + EPS)
#     # min_corresp_err_ind = torch.min(err_corresp_to_ano.abs() + unobserved * HIGH, dim=-2, keepdim=True)[1]  # get only indices
#     min_corresp_err_ind = torch.min(err_corresp_to_ano.abs() + no_edge_corresp * HIGH, dim=-2, keepdim=True)[
#         1]  # get only indices
#     min_corresp_err = torch.gather(err_corresp_to_ano, -2, min_corresp_err_ind).squeeze(-2)  # collapse edge dim
#
#     min_corresp_flow = timeseg_mat2tensor(min_corresp_flow, num_seg)
#     min_corresp_err = timeseg_mat2tensor(min_corresp_err, num_seg)
#     proj_err = timeseg_mat2tensor(proj_err, num_seg)
#     A = timeseg_mat2tensor(A, num_seg)
#     reliable = timeseg_mat2tensor(reliable, num_seg)
#
#     shape = A.shape
#
#     Aabs = A.abs()
#     proj_err_abs = proj_err.abs()
#
#     # min_corresp_flow_mode1_var = min_corresp_flow.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     min_corresp_flow_mode1_var = _masked_mean_var(min_corresp_flow, mask=reliable, dim=(-2, -1))[1].unsqueeze(
#         -1).unsqueeze(-1) + EPS ** 2
#     # min_corresp_flow_mode2_var = min_corresp_flow.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     min_corresp_flow_mode2_var = _masked_mean_var(min_corresp_flow, mask=reliable, dim=(-3, -1))[1].unsqueeze(
#         -2).unsqueeze(-1) + EPS ** 2
#     # min_corresp_flow_mode3_var = min_corresp_flow.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#     min_corresp_flow_mode3_var = _masked_mean_var(min_corresp_flow, mask=reliable, dim=(-3, -2))[1].unsqueeze(
#         -2).unsqueeze(-2) + EPS ** 2
#
#     # min_corresp_err_mode1_var = min_corresp_err.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     min_corresp_err_mode1_var = _masked_mean_var(min_corresp_err, mask=reliable, dim=(-2, -1))[1].unsqueeze(
#         -1).unsqueeze(-1) + EPS ** 2
#     # min_corresp_err_mode2_var = min_corresp_err.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     min_corresp_err_mode2_var = _masked_mean_var(min_corresp_err, mask=reliable, dim=(-3, -1))[1].unsqueeze(
#         -2).unsqueeze(-1) + EPS ** 2
#     # min_corresp_err_mode3_var = min_corresp_err.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#     min_corresp_err_mode3_var = _masked_mean_var(min_corresp_err, mask=reliable, dim=(-3, -2))[1].unsqueeze(
#         -2).unsqueeze(-2) + EPS ** 2
#
#     proj_err_mode1_var = proj_err.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     proj_err_mode2_var = proj_err.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     proj_err_mode3_var = proj_err.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#
#     A_mode1_var = A.var(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS ** 2
#     A_mode2_var = A.var(dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS ** 2
#     A_mode3_var = A.var(dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS ** 2
#
#     mode1_min_corresp_flow_var = min_corresp_flow_mode1_var.log() / 2
#     mode1_min_corresp_err_var = min_corresp_err_mode1_var.log() / 2
#     mode1_proj_err_var = proj_err_mode1_var.log() / 2
#     mode1_Avar = A_mode1_var.log() / 2
#
#     mode1_min_corresp_flow_max = (
#                 torch.amax(min_corresp_flow.abs() / (min_corresp_flow_mode2_var * min_corresp_flow_mode3_var).sqrt(),
#                            dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS).log()
#     mode1_min_corresp_err_max = (
#                 torch.amax(min_corresp_err.abs() / (min_corresp_err_mode2_var * min_corresp_err_mode3_var).sqrt(),
#                            dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + EPS).log()
#     mode1_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_err_mode2_var * proj_err_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(
#                     -1).unsqueeze(-1) + EPS).log()
#     mode1_Amax = (torch.amax(Aabs / (A_mode2_var * A_mode3_var).sqrt(), dim=(-2, -1)).unsqueeze(-1).unsqueeze(
#         -1) + EPS).log()
#
#     mode2_min_corresp_flow_var = min_corresp_flow_mode2_var.log() / 2
#     mode2_min_corresp_err_var = min_corresp_err_mode2_var.log() / 2
#     mode2_proj_err_var = proj_err_mode2_var.log() / 2
#     mode2_Avar = A_mode2_var.log() / 2
#
#     mode2_min_corresp_flow_max = (
#                 torch.amax(min_corresp_flow.abs() / (min_corresp_flow_mode1_var * min_corresp_flow_mode3_var).sqrt(),
#                            dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS).log()
#     mode2_min_corresp_err_max = (
#                 torch.amax(min_corresp_err.abs() / (min_corresp_err_mode1_var * min_corresp_err_mode3_var).sqrt(),
#                            dim=(-3, -1)).unsqueeze(-2).unsqueeze(-1) + EPS).log()
#     mode2_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(
#                     -2).unsqueeze(-1) + EPS).log()
#     mode2_Amax = (torch.amax(Aabs / (A_mode1_var * A_mode3_var).sqrt(), dim=(-3, -1)).unsqueeze(-2).unsqueeze(
#         -1) + EPS).log()
#
#     mode3_min_corresp_flow_var = min_corresp_flow_mode3_var.log() / 2
#     mode3_min_corresp_err_var = min_corresp_err_mode3_var.log() / 2
#     mode3_proj_err_var = proj_err_mode3_var.log() / 2
#     mode3_Avar = A_mode3_var.log() / 2
#
#     mode3_min_corresp_flow_max = (
#                 torch.amax(min_corresp_flow.abs() / (min_corresp_flow_mode1_var * min_corresp_flow_mode2_var).sqrt(),
#                            dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS).log()
#     mode3_min_corresp_err_max = (
#                 torch.amax(min_corresp_err.abs() / (min_corresp_err_mode1_var * min_corresp_err_mode2_var).sqrt(),
#                            dim=(-3, -2)).unsqueeze(-2).unsqueeze(-2) + EPS).log()
#     mode3_proj_err_max = (
#                 torch.amax(proj_err_abs / (proj_err_mode1_var * proj_err_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(
#                     -2).unsqueeze(-2) + EPS).log()
#     mode3_Amax = (torch.amax(Aabs / (A_mode1_var * A_mode2_var).sqrt(), dim=(-3, -2)).unsqueeze(-2).unsqueeze(
#         -2) + EPS).log()
#
#     mode1_feat = torch.stack([mode1_min_corresp_flow_var, mode1_min_corresp_err_var,
#                               mode1_proj_err_var, mode1_Avar,
#                               mode1_min_corresp_flow_max, mode1_min_corresp_err_max,
#                               mode1_proj_err_max, mode1_Amax], dim=0).expand(8, *shape)
#
#     mode2_feat = torch.stack([mode2_min_corresp_flow_var, mode2_min_corresp_err_var,
#                               mode2_proj_err_var, mode2_Avar,
#                               mode2_min_corresp_flow_max, mode2_min_corresp_err_max,
#                               mode2_proj_err_max, mode2_Amax], dim=0).expand(8, *shape)
#
#     mode3_feat = torch.stack([mode3_min_corresp_flow_var, mode3_min_corresp_err_var,
#                               mode3_proj_err_var, mode3_Avar,
#                               mode3_min_corresp_flow_max, mode3_min_corresp_err_max,
#                               mode3_proj_err_max, mode3_Amax], dim=0).expand(8, *shape)
#
#     padding = torch.zeros_like(mu_log).unsqueeze(0)
#     # Unfolding back to matrix since A is usually stored as matrix
#     # observations = timeseg_tensor2mat(observations)
#     mode1_feat = timeseg_tensor2mat(mode1_feat)
#     mode2_feat = timeseg_tensor2mat(mode2_feat)
#     mode3_feat = timeseg_tensor2mat(mode3_feat)
#
#     mu_log = mu_log.unsqueeze(0)
#     mu_feat = torch.movedim(torch.cat([mode1_feat, *([padding] * 2),
#                                        mode2_feat, *([padding] * 2),
#                                        mode3_feat, *([padding] * 2),
#                                        observations, mu_log]), 0, -1)  # feature dim to last, 20 feats
#
#     return mu_feat

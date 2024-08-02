from copy import deepcopy

import torch
import torch.nn as nn

import datagen
import utils
from reference_algo import _bsca_update_P, _bsca_update_Q, _bsca_update_A, _bsca_update_A_rlx
from reference_algo import _bsca_incomplete_meas_init_deterministic
from reference_algo import _bsca_incomplete_meas_init_scaled_randn
from utils import RegParamMLP, _masked_mean_var

from config import DEBUG, FP_DTYPE, PREC_EPS, DEVICE

DEBUG_GLOBAL_FLOW = False  # to check whether there are advantages when knowing the true flow variance


class BSCAUnrolled(nn.Module):
    lam_val = []
    mu_val = []

    def __init__(self,
                 num_layers,
                 rank,
                 param_nw=False,  # networks for parameters
                 shared_weights=False,
                 layer_param=None,
                 last_layer_unmodified=False,
                 init="randsc",
                 balanced=None,  # placeholder
                 it1_option=None,  # placeholder
                 it1_not_relaxed=None,  # placeholder
                 skipA=0,
                 A_holdoff=0):
        super().__init__()
        if layer_param is None:
            layer_param = {}

        self.num_layers = num_layers
        self.same_weights_across_layers = shared_weights
        self.rank = rank
        self.param_nw = param_nw
        self.it1_option = it1_option
        self.layer_param = layer_param
        self.last_layer_unmodified = last_layer_unmodified
        self.A_holdoff = max(A_holdoff, skipA)
        self.init = init
        self.layers = []

        if self.param_nw == "bayes":
            iter_class = BSCAUnrolledIteration_Bayes
            print("Bayes NW")
        elif self.param_nw:
            iter_class = BSCAUnrolledIteration_ParamNW
        else:
            iter_class = BSCAUnrolledIteration

        if not shared_weights:
            for i_layer in range(self.num_layers):
                if i_layer == num_layers - 1 and self.last_layer_unmodified:
                    self.layers.append(iter_class())
                else:
                    layer_param_temp = deepcopy(self.layer_param)
                    if self.it1_option and i_layer == 0:
                        layer_param_temp["option"] = self.it1_option
                    self.layers.append(iter_class(**layer_param_temp))
        else:
            self.layers = [iter_class(**layer_param)] * num_layers

        self.layers = nn.ModuleList(self.layers)

    def forward(self, scenario_dict, return_im_steps=True):
        Y, R, Omega = datagen.nw_scenario_observation(scenario_dict)

        # if DEBUG_GLOBAL_FLOW:
        #     global scenFlow, scenA
        #     scenFlow = scenario_dict.data["Z"].to(FP_DTYPE) + scenario_dict.data["A"].to(FP_DTYPE)
        #     scenA = scenario_dict.data["A"]

        if Y.dtype != FP_DTYPE:
            Y = Y.to(FP_DTYPE)
            Omega = Omega.to(FP_DTYPE)
            R = R.to(FP_DTYPE)

        # Init
        if self.init == "default":
            P, Q, A = _bsca_incomplete_meas_init_deterministic(Y, R, Omega, self.rank)
        elif self.init == "randsc":
            # alpha is chosen heuristically
            P, Q, A = _bsca_incomplete_meas_init_scaled_randn(Y, R, Omega, self.rank, sigma=0.1)
        else:
            raise ValueError

        if return_im_steps:
            P_list = [P]
            Q_list = [Q]
            A_list = [A]

        self.lam_val = []
        self.mu_val = []

        if self.param_nw == "bayes":
            nw_out = {"datafit_weight_log": torch.zeros_like(Y), "mu_log": torch.zeros_like(A), "sa_log": torch.zeros_like(A)}
        elif self.param_nw == True:
            nw_out = torch.zeros(scenario_dict["batch_size"], self.layers[0].num_out, device=DEVICE)

        for l in range(self.num_layers):
            # print(l)
            if self.param_nw:
                P, Q, A, nw_out = self.layers[l](Y, R, Omega, P, Q, A, nw_out, last_layer=(l == (self.num_layers-1)))
            else:
                P, Q, A = self.layers[l](Y, R, Omega, P, Q, A, A_update=(l >= self.A_holdoff))

            if return_im_steps:
                P_list.append(P)
                Q_list.append(Q)
                A_list.append(A)

            self.lam_val.append(self.layers[l].lam_val)
            self.mu_val.append(self.layers[l].mu_val)

        self.lam_val = torch.stack(self.lam_val)
        self.mu_val = torch.stack(self.mu_val)

        if return_im_steps:
            P_list = torch.stack(P_list)
            Q_list = torch.stack(Q_list)
            A_list = torch.stack(A_list)
            return P_list, Q_list, A_list
        else:
            return P, Q, A


class BSCAUnrolledIteration(nn.Module):
    lam_val = None
    mu_val = None

    def __init__(self,
                 two_lambda=False,
                 soft_thresh_cont_grad=False,
                 A_postproc=None,
                 skip_connections=False,
                 option=None,  # placeholder
                 relaxed=None,  # placeholder
                 init_val=-5):
        super().__init__()
        self.two_lambda = two_lambda
        self.soft_thresh_cont_grad = soft_thresh_cont_grad
        self.A_postproc = A_postproc
        self.skip_connections = skip_connections

        self.lam_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))
        self.mu_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))

        if self.two_lambda:
            self.lam2_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))

        if self.skip_connections:
            self.skip_P = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
            self.skip_Q = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
            self.skip_A = nn.Parameter(torch.tensor(0.5, dtype=torch.float))

    def forward(self, Y, R, Omega, P, Q, A, A_update=True):
        lam1 = torch.exp(self.lam_log)
        if self.two_lambda:
            lam2 = torch.exp(self.lam2_log)
        else:
            lam2 = lam1
        mu = torch.exp(self.mu_log)

        self.lam_val = lam1.detach().clone()
        self.mu_val = mu.detach().clone()

        err = Omega * (Y - (R @ A))

        P_new = _bsca_update_P(Y, R, Omega, Q, A, lam1, err)
        if self.skip_connections:
            P_new = P + torch.sigmoid(self.skip_P) * (P_new - P)

        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam2, err)
        if self.skip_connections:
            Q_new = Q + torch.sigmoid(self.skip_Q) * (Q_new - Q)

        if A_update:
            A_new = _bsca_update_A(Y, R, Omega, P_new, Q_new, A, mu, err,
                                   soft_thresh_cont_grad=self.soft_thresh_cont_grad)
        else:
            A_new = A

        if self.skip_connections:
            A_new = A + torch.sigmoid(self.skip_A) * (A_new - A)

        return P_new, Q_new, A_new

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
            if self.two_lambda:
                param_dict["lam2"] = torch.exp(self.lam2_log).detach().clone()
        else:
            param_dict["lam"] = torch.exp(self.lam_log)
            param_dict["mu"] = torch.exp(self.mu_log)
            if self.two_lambda:
                param_dict["lam2"] = torch.exp(self.lam2_log)

        return param_dict


class BSCAUnrolledIteration_ParamNW(nn.Module):
    lam_val = None
    mu_val = None
    MU_LOG_OFFSET = -5.0

    def __init__(self,
                 two_lambda=False,  # DEPRECATED
                 skip_connections=False,  # this is a placeholder for compatibility

                 ):
        super().__init__()
        self.two_lambda = two_lambda
        self.skip_connections = False
        self.num_in_features = 8

        self.nw_out = ["lam", "mu"]

        # if self.two_lambda:
        #     self.nw_out.append("lam2")

        if self.skip_connections:
            self.nw_out.extend(["skip_P", "skip_Q", "skip_A"])

        self.num_out = len(self.nw_out)
        self.num_in_features = self.num_in_features + self.num_out
        self.param_nw = RegParamMLP([self.num_in_features, self.num_in_features, self.num_in_features, self.num_out],
                                    batch_norm_input=True)

    def forward(self, Y, R, Omega, P, Q, A, prev_param_nw_out, last_layer=False):
        if self.two_lambda:
            raise NotImplementedError
        err = Omega * (Y - (R @ A))
        # num_timesteps = Y.shape[-1]

        feature_vec = _compile_features_improved(Y, R, Omega, P, Q, A, prev_param_nw_out)
        param_nw_out = self.param_nw(feature_vec)
        lam1 = torch.exp(param_nw_out[..., 0])
        self.lam_val = lam1.detach().clone()

        mu = torch.exp(param_nw_out[..., 1] + self.MU_LOG_OFFSET)
        self.mu_val = mu.detach().clone()
        mu = mu.unsqueeze(-1).unsqueeze(-1)  # makes it broadcastable with A

        lam2 = lam1

        if self.skip_connections:
            skip_P = param_nw_out[..., 2].unsqueeze(-1).unsqueeze(-1)  # expand into matrix dim
            skip_Q = param_nw_out[..., 3].unsqueeze(-1).unsqueeze(-1)  # expand into matrix dim
            skip_A = param_nw_out[..., 4].unsqueeze(-1).unsqueeze(-1)  # expand into matrix dim

        P_new = _bsca_update_P(Y, R, Omega, Q, A, lam1, err)
        if self.skip_connections:
            P_new = P + torch.sigmoid(skip_P) * (P_new - P)

        Q_new = _bsca_update_Q(Y, R, Omega, P_new, A, lam2, err)
        if self.skip_connections:
            Q_new = Q + torch.sigmoid(skip_Q) * (Q_new - Q)

        A_new = _bsca_update_A(Y, R, Omega, P_new, Q_new, A, mu, err)
        if self.skip_connections:
            A_new = A + torch.sigmoid(skip_A) * (A_new - A)

        return P_new, Q_new, A_new, param_nw_out

    def get_regularization_parameters(self, clone=False, batch_mean=True):
        # Returns default values to not crash rest of the code.
        param_dict = {}
        param_dict["lam"] = self.lam_val.mean(dim=0)
        param_dict["mu"] = self.mu_val.mean(dim=0)
        if self.two_lambda:
            param_dict["lam2"] = torch.tensor(1.0)

        return param_dict


class BSCAUnrolledIteration_Bayes(nn.Module):
    lam_val = None
    mu_val = None
    EPS = 1e-6

    MAX_DFIT = 8
    MAX_MU_LOG = 8

    def __init__(self,
                 datafit_options=None,
                 lam_options=None,
                 mu_options=None,
                 init_val=-2,
                 option="nrlx",  # non-relaxed/relaxed updates
                 batch_norm=False,
                 two_nu=False,
                 ):
        super().__init__()

        self.num_out = 1  # legacy
        self.batch_norm = batch_norm
        self.two_nu = two_nu

        if datafit_options is None:
            self.datafit_options = []
        else:
            self.datafit_options = datafit_options

        if lam_options is None:
            self.lam_options = []
        else:
            self.lam_options = lam_options

        if mu_options is None:
            self.mu_options = []
        else:
            self.mu_options = mu_options

        self.option = option

        if "static1" in self.datafit_options:
            datafit_ffun = _datafit_features_static
            self.datafit_nw = RegParamMLP([9, 9, 1], batch_norm_input=False)
        elif "static2" in self.datafit_options or "static2clamp" in self.datafit_options or "var" in self.datafit_options \
                or "static2e-3" in self.datafit_options or "static2e-0" in self.datafit_options:
            datafit_ffun = _datafit_features_static2
            self.datafit_nw = RegParamMLP([5, 5, 1], batch_norm_input=self.batch_norm)
        elif "staticn1ly" in self.datafit_options or "staticnc1ly" in self.datafit_options:
            datafit_ffun = _datafit_features_static2
            self.datafit_nw = RegParamMLP([5, 1], batch_norm_input=self.batch_norm)
        elif "staticn2ly" in self.datafit_options or "staticnc2ly" in self.datafit_options:
            datafit_ffun = _datafit_features_static2
            self.datafit_nw = RegParamMLP([5, 5, 1], batch_norm_input=self.batch_norm)
        elif "dynnc1ly" in self.datafit_options:
            datafit_ffun = _datafit_features_dynamic
            self.datafit_nw = RegParamMLP([6, 1], batch_norm_input=self.batch_norm)

        if "fmask" in self.datafit_options:
            self.datafit_ffun = lambda *args: (datafit_ffun(*args) * self.datafit_options["fmask"])
        elif len(self.datafit_options) > 0:
            self.datafit_ffun = datafit_ffun

        if len(self.datafit_options) > 0:
            self.df_dict = {"min": torch.tensor([], dtype=FP_DTYPE),
                            "max": torch.tensor([], dtype=FP_DTYPE),
                            "mean": torch.tensor([], dtype=FP_DTYPE)}
        else:
            self.datafit_nw = None

        if "1dyn1ly" in self.lam_options:
            lam_ffun = _lam_features_dynamic
            # Two networks to capture different statistics in each mode
            self.lam_nw = nn.ModuleList([RegParamMLP([2, 1], batch_norm_input=self.batch_norm), RegParamMLP([2, 1], batch_norm_input=True)])
            self.lam_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))  # kept since variable parameters clamped
        else:
            self.lam_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))
            self.lam_nw = None

        if "fmask" in self.lam_options:
            self.lam_ffun = lambda *args: (lam_ffun(*args) * self.lam_options["fmask"])
        elif len(self.lam_options) > 0:
            self.lam_ffun = lam_ffun

        if "static1" in self.mu_options:
            self.mu_nw = RegParamMLP([9, 9, 1], batch_norm_input=False)
            mu_ffun = _mu_features_static
        elif "dyn1ly" in self.mu_options or "dynnc1ly" in self.mu_options:
            self.mu_nw = RegParamMLP([20, 1], batch_norm_input=self.batch_norm)
            mu_ffun = _mu_features_dynamic
        elif "dy2nc1ly" in self.mu_options:
            self.mu_nw = RegParamMLP([22, 1], batch_norm_input=self.batch_norm)
            mu_ffun = _mu_features_dynamic2
        elif "dy2nc1lyi" in self.mu_options:
            self.mu_nw = RegParamMLP([22, 1], batch_norm_input=self.batch_norm)
            mu_ffun = _mu_features_dynamic2i
        elif "dy3nc1ly" in self.mu_options or "dy4nc1ly" in self.mu_options:
            self.mu_nw = RegParamMLP([22, 1], batch_norm_input=self.batch_norm)
            mu_ffun = _mu_features_dynamic2
            self.mu_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))  # allowed due to clamping
        else:
            self.mu_log = nn.Parameter(torch.tensor(init_val, dtype=torch.float))
            self.mu_nw = None

        if len(self.mu_options) > 0:
            self.mu_dict = {"min": torch.tensor([], dtype=FP_DTYPE),
                            "max": torch.tensor([], dtype=FP_DTYPE),
                            "mean": torch.tensor([], dtype=FP_DTYPE)}

        if "fmask" in self.mu_options:
            self.mu_ffun = lambda *args: (mu_ffun(*args) * self.mu_options["fmask"])
        elif len(self.mu_options) > 0:
            self.mu_ffun = mu_ffun

        if self.option == "nrlx":
            pass
        elif self.option == "rlx" or self.option == "rlx+nnf" or self.option == "rlx2x" or self.option == "rlx2x+nnf":
            if two_nu:
                self.nu_log = nn.Parameter(torch.tensor([0, 0], dtype=torch.float))
            else:
                self.nu_log = nn.Parameter(torch.tensor(0, dtype=torch.float))
        else:
            raise ValueError

    def forward(self, Y, R, Omega, P, Q, A, prev_nw_out=None, last_layer=False):

        nw_out = {}

        if "var" in self.datafit_options:
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_weight = 1 / (datafit_fvec[..., 1] * datafit_fvec[..., 3] + self.EPS).sqrt()
            datafit_weight_log = datafit_weight.log()
            WOmega = Omega * datafit_weight
        elif "static1" in self.datafit_options:
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_fvec = torch.log(datafit_fvec + self.EPS)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
        elif "static2" in self.datafit_options or "static2clamp" in self.datafit_options:
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_fvec = torch.log(datafit_fvec + self.EPS)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            if "static2clamp" in self.datafit_options:
                datafit_weight_log = torch.clamp(datafit_weight_log, min=-8, max=8)
            else:
                datafit_weight_log_mean = datafit_weight_log.mean(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
                datafit_weight_log = datafit_weight_log - datafit_weight_log_mean
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
        elif "static2e-3" in self.datafit_options:
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_fvec = torch.log(datafit_fvec + 1e-3)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            datafit_weight_log_mean = datafit_weight_log.mean(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
            datafit_weight_log = datafit_weight_log - datafit_weight_log_mean
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
        elif "static2e-0" in self.datafit_options:
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_fvec = torch.log(datafit_fvec + 1.0)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
        elif "staticn1ly" in self.datafit_options or "staticn2ly" in self.datafit_options:
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_fvec = torch.log(datafit_fvec + 1e-3)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            # datafit_weight_log_mean = datafit_weight_log.mean(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
            # datafit_weight_log = datafit_weight_log - datafit_weight_log_mean
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
        elif "staticnc1ly" in self.datafit_options or "staticnc2ly" in self.datafit_options:
            max_fit = self.MAX_DFIT
            datafit_fvec = self.datafit_ffun(Y, R, Omega)
            datafit_fvec = torch.log(datafit_fvec + 1e-3)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            datafit_weight_log = max_fit*torch.tanh(datafit_weight_log / max_fit)  # clamping
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
        elif "dynnc1ly" in self.datafit_options or "dynnc2ly" in self.datafit_options:
            prev_datafit_weight_log = prev_nw_out["datafit_weight_log"]
            max_fit = self.MAX_DFIT
            datafit_fvec = self.datafit_ffun(Y, R, Omega, P @ Q.mT, A, prev_datafit_weight_log)
            datafit_weight_log = self.datafit_nw(datafit_fvec)[..., 0]
            datafit_weight_log = max_fit*torch.tanh(datafit_weight_log / max_fit)
            datafit_weight = torch.exp(datafit_weight_log)
            WOmega = Omega * datafit_weight
            nw_out["datafit_weight_log"] = datafit_weight_log
        else:
            WOmega = Omega

        if "1dyn1ly" in self.lam_options:
            # prev_lam_weight_log = prev_nw_out["lam_weight_logs"]  # not implemented currently
            max_fit = 5
            lam_fvecs = self.lam_ffun(Y, R, Omega, P @ Q.mT, A)
            lam_weight_logs = [max_fit * torch.tanh(self.lam_nw[m](lam_fvecs[m])[..., 0] / max_fit) for m in range(2)]
            # datafit_weight_log = max_fit * torch.tanh(datafit_weight_log / max_fit)
            lam_weights = [torch.exp(lam_weight_logs[m] / 2) for m in range(2)]
            nw_out["lam_weight_logs"] = lam_weight_logs
        else:
            pass

        lam = torch.exp(self.lam_log)
        self.lam_val = lam.detach().clone()

        if not "feat_after_PQ" in self.mu_options:
            if "static1" in self.mu_options:
                mu_fvec = self.mu_ffun(Y, R, Omega)
                mu_fvec = torch.log(mu_fvec + self.EPS)
                mu = torch.exp(self.mu_nw(mu_fvec)[..., 0] - 5)  # -5 to start with small values, otherwise might get stuck
                self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
            elif "dyn1ly" in self.mu_options:
                mu_log_prev = prev_nw_out["mu_log"]
                mu_fvec = self.mu_ffun(Y, R, Omega, P @ Q.mT, A, mu_log_prev)
                mu_log = self.mu_nw(mu_fvec)[..., 0] - 5
                nw_out["mu_log"] = mu_log
                mu = torch.exp(mu_log)  # -5 to start with small values, otherwise might get stuck
                self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
                # print(mu_log.mean())
            elif "dynnc1ly" in self.mu_options or "dy2nc1ly" in self.mu_options or "dy2nc1lyi" in self.mu_options:
                max_mulog = self.MAX_MU_LOG
                mu_log_prev = prev_nw_out["mu_log"]
                mu_fvec = self.mu_ffun(Y, R, Omega, P @ Q.mT, A, mu_log_prev)
                mu_log = self.mu_nw(mu_fvec)[..., 0] - 5  #initial bias = -5
                mu_log = max_mulog * torch.tanh(mu_log / max_mulog)
                nw_out["mu_log"] = mu_log
                mu = torch.exp(mu_log)  # -5 to start with small values, otherwise might get stuck
                self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
                # print(mu_log.mean())
            elif "dy3nc1ly" in self.mu_options:
                """This similar to the case above save for some renaming.
                Propagated A is actually Atilde in this case"""
                Atil = A
                max_salog = 5
                sa_log_prev = prev_nw_out["sa_log"]
                sa_fvec = self.mu_ffun(Y, R, Omega, P @ Q.mT, Atil, sa_log_prev)
                sa_log = self.mu_nw(sa_fvec)[..., 0] + 3  #initial bias = -5
                sa_log = max_salog * torch.tanh(sa_log / max_salog)
                nw_out["sa_log"] = sa_log
                sa = torch.exp(sa_log)# -5 to start with small values, otherwise might get stuck
                mu = torch.exp(self.mu_log.unsqueeze(-1).unsqueeze(-1) - sa_log) # mu/S_A
                A = sa * Atil  # of previous iteration
                self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
                # print(mu_log.mean())
            elif "dy4nc1ly" in self.mu_options:
                """This similar to the case above save for some renaming.
                Propagated A is actually Atilde in this case"""
                Atil = A
                max_salog = 5
                sa_log_prev = prev_nw_out["sa_log"]

                sa_fvec = self.mu_ffun(Y, R, Omega, P @ Q.mT, Atil, sa_log_prev)
                sa_log = self.mu_nw(sa_fvec)[..., 0] + 3  #initial bias = -5
                sa_log = max_salog * torch.tanh(sa_log / max_salog)
                nw_out["sa_log"] = sa_log
                sa = torch.exp(sa_log)# -5 to start with small values, otherwise might get stuck
                mu = torch.exp(self.mu_log.unsqueeze(-1).unsqueeze(-1) - sa_log) # mu/S_A
                Adash = sa * Atil  # scaled for A
                A = Atil * sa_log_prev
                self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
                # print(mu_log.mean())
            else:
                mu = torch.exp(self.mu_log)
                self.mu_val = mu.detach().clone()


        # if "dy2nc1ly" in self.mu_options:
        #     A =

        err = WOmega * (Y - (R @ A))

        if self.option == "nrlx":
            if "1dyn1ly" in self.lam_options:
                P_new = _bsca_update_P(Y, R, WOmega * lam_weights[0], Q, A, lam, err)
                Q_new = _bsca_update_Q(Y, R, WOmega * lam_weights[1], P_new, A, lam, err)
            else:
                P_new = _bsca_update_P(Y, R, WOmega, Q, A, lam, err)
                Q_new = _bsca_update_Q(Y, R, WOmega, P_new, A, lam, err)

            if "feat_after_PQ" in self.mu_options:
                if "dynnc1ly" in self.mu_options or "dy2nc1ly" in self.mu_options or "dy2nc1lyi" in self.mu_options:
                    max_mulog = self.MAX_MU_LOG
                    mu_log_prev = prev_nw_out["mu_log"]
                    # mu_fvec = self.mu_ffun(Y, R, Omega, P_new @ Q_new.mT, A, mu_log_prev)
                    mu_fvec = self.mu_ffun(Y, R, WOmega, P_new @ Q_new.mT, A, mu_log_prev)
                    mu_log = self.mu_nw(mu_fvec)[..., 0] - 5  # initial bias = -5
                    mu_log = max_mulog * torch.tanh(mu_log / max_mulog)
                    nw_out["mu_log"] = mu_log
                    mu = torch.exp(mu_log)  # -5 to start with small values, otherwise might get stuck
                    self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
                    # print(mu_log.mean())
                else:
                    raise ValueError
            if "dy4nc1ly" in self.mu_options:
                A_new = _bsca_update_A(Y, R, WOmega, P_new, Q_new, Adash, mu, err)
            else:
                A_new = _bsca_update_A(Y, R, WOmega, P_new, Q_new, A, mu, err)

        elif self.option == "rlx" or self.option == "rlx+nnf" or self.option == "rlx2x" or self.option == "rlx2x+nnf":

            if self.two_nu:
                nu1 = torch.exp(self.nu_log[0])
                nu2 = torch.exp(self.nu_log[1])
                lam_div_nu = torch.exp(self.lam_log - self.nu_log[0])
            else:
                nu1 = torch.exp(self.nu_log)
                nu2 = torch.exp(self.nu_log)
                lam_div_nu = torch.exp(self.lam_log - self.nu_log)

            if self.option == "rlx+nnf" or self.option == "rlx2x+nnf":
                X_new = _bsca_update_X_rlx(Y, R, WOmega, P, Q, A, nu1, err, nnf=True)
            else:
                X_new = _bsca_update_X_rlx(Y, R, WOmega, P, Q, A, nu1, err)

            # lam_div_nu = torch.exp(self.lam_log - self.nu_log)
            P_new = _bsca_update_P_rlx(X_new, Q, lam_div_nu)
            Q_new = _bsca_update_Q_rlx(X_new, P_new, lam_div_nu)

            if self.option == "rlx2x+nnf":
                X_new = _bsca_update_X_rlx(Y, R, WOmega, P_new, Q_new, A, nu2, err, nnf=True)
            elif self.option == "rlx2x":
                X_new = _bsca_update_X_rlx(Y, R, WOmega, P_new, Q_new, A, nu2, err)
            else:
                pass

            if "feat_after_PQ" in self.mu_options:
                if "dynnc1ly" in self.mu_options or "dy2nc1ly" in self.mu_options or "dy2nc1lyi" in self.mu_options:
                    max_mulog = self.MAX_MU_LOG
                    mu_log_prev = prev_nw_out["mu_log"]
                    # mu_fvec = self.mu_ffun(Y, R, Omega, P_new @ Q_new.mT, A, mu_log_prev)
                    mu_fvec = self.mu_ffun(Y, R, WOmega, X_new, A, mu_log_prev)
                    mu_log = self.mu_nw(mu_fvec)[..., 0] - 5  # initial bias = -5
                    mu_log = max_mulog * torch.tanh(mu_log / max_mulog)
                    nw_out["mu_log"] = mu_log
                    mu = torch.exp(mu_log)  # -5 to start with small values, otherwise might get stuck
                    self.mu_val = mu.detach().clone().mean(dim=(-1, -2))
                    # print(mu_log.mean())
                else:
                    raise ValueError

            if "dy4nc1ly" in self.mu_options:
                A_new = _bsca_update_A_rlx(Y, R, WOmega, X_new, Adash, mu, err)
            else:
                A_new = _bsca_update_A_rlx(Y, R, WOmega, X_new, A, mu, err)
        else:
            raise ValueError

        if ("scaled_Aout" in self.mu_options and last_layer):
            # print("yo")
            A_new = A_new / mu
        elif "dy3nc1ly" in self.mu_options:
            A_new = A_new / sa  # actually Atil_new

        if self.training and len(self.datafit_options) > 0:
            self.df_dict["min"] = torch.cat([self.df_dict["min"], datafit_weight_log.detach().min().unsqueeze(0)])
            self.df_dict["max"] = torch.cat([self.df_dict["max"], datafit_weight_log.detach().max().unsqueeze(0)])
            self.df_dict["mean"] = torch.cat([self.df_dict["mean"], datafit_weight_log.detach().mean().unsqueeze(0)])

        if self.training and len(self.mu_options) > 0 and False:
            self.mu_dict["min"] = torch.cat([self.mu_dict["min"], torch.flatten(mu_fvec.detach(), end_dim=-2).min(dim=0)[0].unsqueeze(0)])
            self.mu_dict["max"] = torch.cat([self.mu_dict["max"], torch.flatten(mu_fvec.detach(), end_dim=-2).max(dim=0)[0].unsqueeze(0)])
            self.mu_dict["mean"] = torch.cat([self.mu_dict["mean"], torch.flatten(mu_fvec.detach(), end_dim=-2).mean(dim=0).unsqueeze(0)])

        return P_new, Q_new, A_new, nw_out

    # def get_regularization_parameters(self, clone=False, batch_mean=True):
    #     # Returns default values to not crash rest of the code.
    #     param_dict = {}
    #     param_dict["lam"] = self.lam_val.mean(dim=(0, -2, -1))
    #     param_dict["mu"] = self.mu_val.mean(dim=(0, -2, -1))
    #
    #     return param_dict


def _bsca_update_X_rlx(Y, R, Omega, P, Q, A, nu, err, nnf=False):
    """
    err = Omega * (Y - (R @ A))
    """
    if err is None:
        err = (Omega**2) * (Y - R @ A)
    else:
        err = Omega * err

    nu_usq = nu.unsqueeze(-1).unsqueeze(-1)
    divisor = Omega**2 + torch.ones_like(Omega) * nu_usq
    X_new = (Omega * err + nu_usq * P @ Q.mT) / divisor

    if nnf:
        X_new = torch.clamp(X_new, min=0)
    return X_new


def _bsca_update_P_rlx(X, Q, regpar):
    """X, Q matrices have leading batch dimension.
    regpar: normally lambda / nu, broadcastable with (*, batch_size)"""

    rank = Q.shape[-1]

    lhs = Q.mT @ Q
    if FP_DTYPE == torch.float:
        regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), regpar)
    else:
        regularizer = regpar
    lhs = lhs + regularizer.unsqueeze(-1).unsqueeze(-1) * torch.eye(rank)
    rhs = X @ Q
    P_new = torch.linalg.solve(lhs, rhs, left=False)

    return P_new


def _bsca_update_Q_rlx(X, P, regpar):
    """X, P matrices have leading batch dimension.
    regpar: normally lambda / nu, broadcastable with (*, batch_size)"""
    rank = P.shape[-1]

    lhs = P.mT @ P
    if FP_DTYPE == torch.float:
        regularizer = torch.maximum(PREC_EPS * utils.min_diff_machine_precision(lhs.abs().max(dim=-1)[0].max(dim=-1)[0]), regpar)
    else:
        regularizer = regpar
    lhs = lhs + regularizer.unsqueeze(-1).unsqueeze(-1) * torch.eye(rank)
    rhs = X.mT @ P

    Q_new = torch.linalg.solve(lhs, rhs, left=False)

    return Q_new


def _compile_features_improved(Y, R, Omega, P, Q, A, nw_out):
    # EPS = 1e-4
    datafit = Y - P @ Q.mT - R @ A
    datafit_sq = datafit ** 2
    datafit_sq_mean, datafit_sq_var = _masked_mean_var(datafit_sq, mask=Omega, dim=(-2, -1))

    p = P.abs()
    p_mean = p.mean(dim=(-2, -1))

    q = Q.abs()
    q_mean = q.mean(dim=(-2, -1))
    full_err = Omega * datafit

    # Direction
    A_scale = (R * R).transpose(-2, -1) @ Omega
    A_scale_zero = A_scale == 0
    BA_temp = A_scale * A + R.mT @ full_err
    A_scale_safezero = A_scale + A_scale_zero * 1
    BA = BA_temp / A_scale_safezero
    BA[A_scale_zero] = 0

    BA = BA.abs()

    BA_mean = BA.mean(dim=(-2, -1))
    BA_var = BA.var(dim=(-2, -1))
    BA_max = BA.flatten(start_dim=-2).max(dim=-1)[0]

    prob_observation = Omega.sum(dim=(-2, -1)) / Omega.shape[-1] / Omega.shape[-2]

    feature_vec = [datafit_sq_mean.log(), datafit_sq_var.log(), p_mean.log(), q_mean.log(),
                   BA_mean.log(), BA_var.log(), BA_max.log(), prob_observation,
                   *[t.squeeze(-1) for t in torch.split(nw_out, 1, dim=-1)]]

    feature_vec = torch.stack(feature_vec, dim=-1)

    return feature_vec


def _datafit_features_raw(Y, R, Omega):
    """Returns (batch_size, E, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
    HIGH = 1e18
    shape = Y.shape
    Omega = Omega > 0

    Yabs = Y.abs()

    # per link
    link_m, link_var = _masked_mean_var(Y, mask=Omega, dim=[-1])
    link_min, link_max = torch.min(Yabs + (~Omega) * HIGH, dim=-1)[0], torch.max(Yabs - (~Omega) * HIGH, dim=-1)[0]
    link_feat = torch.stack([link_m.abs(), link_var, link_min, link_max], dim=0).unsqueeze(-1).expand(4, *shape)

    # per time step
    time_m, time_var = _masked_mean_var(Y, mask=Omega, dim=[-2])
    time_min, time_max = torch.min(Yabs + (~Omega) * HIGH, dim=-2)[0], torch.max(Yabs - (~Omega) * HIGH, dim=-2)[0]
    time_feat = torch.stack([time_m.abs(), time_var, time_min, time_max], dim=0).unsqueeze(-2).expand(4, *shape)

    routing_feat = R.sum(dim=-1).unsqueeze(-1).expand(*shape).unsqueeze(0)

    datafit_feat = torch.movedim(torch.cat([link_feat, time_feat, routing_feat]), 0, -1)  # feature dim to last

    # features are nonnegative, but may be 0

    return datafit_feat


def _datafit_features_static(Y, R, Omega):
    return _datafit_features_raw(Y, R, Omega)


def _datafit_features_static2(Y, R, Omega):
    return _datafit_features_raw(Y, R, Omega)[..., [0, 1, 4, 5, 8]]


def _datafit_features_dynamic(Y, R, Omega, X, A, datafit_weight_log):
    """Returns (batch_size, E, T, num_features) feature tensor.
    Features: Y var per link, err var per link, Y var per t, err var per t, num_routes per link, prev_dataweight
    Omega - expected to have all elements in {0, 1}"""
    EPS = 1e-3
    shape = Y.shape
    Omega = Omega > 0

    Yabs = Y.abs()

    err = Y - X - R @ A

    # per link
    link_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-1])[1] + EPS).log()
    link_var2 = (_masked_mean_var(err, mask=Omega, dim=[-1])[1] + EPS).log()
    link_feat = torch.stack([link_var1, link_var2], dim=0).unsqueeze(-1).expand(2, *shape)

    # per time step
    time_var = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()
    time_var2 = (_masked_mean_var(err, mask=Omega, dim=[-2])[1] + EPS).log()
    time_feat = torch.stack([time_var, time_var2], dim=0).unsqueeze(-2).expand(2, *shape)

    routing_feat = (R.sum(dim=-1) + EPS).log().unsqueeze(-1).expand(*shape).unsqueeze(0)

    # lam_feat = lam_log.expand(*shape).unsqueeze(0)
    # omega_feat = Omega.expand(*shape).unsqueeze(0)
    datafit_weight_log = datafit_weight_log.unsqueeze(0)
    datafit_feat = torch.movedim(torch.cat([link_feat, time_feat, routing_feat, datafit_weight_log]), 0, -1)  # feature dim to last

    # features are nonnegative, but may be 0

    return datafit_feat


def _mu_features_static(Y, R, Omega):
    """Returns (batch_size, F, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
    HIGH = 1e18

    A_scale = (R ** 2).mT @ Omega
    A_noobs = A_scale == 0
    A_scale_safezero = A_scale + A_noobs * 1
    proj_flow = R.mT @ (Y * Omega) / A_scale_safezero
    observations = A_scale.unsqueeze(0)  # prepending feature dimension

    shape = proj_flow.shape
    proj_flow_abs = proj_flow.abs()
    # links (avg over time)
    flow_m, flow_var = _masked_mean_var(proj_flow, mask=None, dim=-1)
    flow_min, flow_max = torch.min(proj_flow_abs, dim=-1)[0], torch.max(proj_flow_abs - A_noobs * HIGH, dim=-1)[0]
    flow_feat = torch.stack([flow_m.abs(), flow_var, flow_min, flow_max], dim=0).unsqueeze(-1).expand(4, *shape)

    # links (avg over links)
    time_m, time_var = _masked_mean_var(Y, mask=Omega, dim=-2)
    time_min, time_max = torch.min(proj_flow_abs + A_noobs * HIGH, dim=-2)[0], \
                         torch.max(proj_flow_abs - A_noobs * HIGH, dim=-2)[0]
    time_feat = torch.stack([time_m.abs(), time_var, time_min, time_max], dim=0).unsqueeze(-2).expand(4, *shape)

    mu_feat = torch.movedim(torch.cat([flow_feat, time_feat, observations]), 0, -1)  # feature dim to last

    return mu_feat


def _mu_features_dynamic(Y, R, Omega, X, A, mu_log):
    """Returns (batch_size, F, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
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
    EPS = 1e-3
    HIGH = 1e18

    A_scale = (R ** 2).mT @ Omega
    A_noobs = A_scale == 0
    A_scale_safezero = A_scale + A_noobs * 1
    proj_flow = R.mT @ (Y * Omega) / A_scale_safezero
    proj_err = R.mT @ ((Y - X) * Omega) / A_scale_safezero
    proj_err2 = R.mT @ ((Y - X - R @ A) * Omega) / A_scale_safezero
    observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension

    shape = proj_flow.shape
    if DEBUG_GLOBAL_FLOW:
        global scenFlow
        proj_flow_abs = scenFlow.abs()
    else:
        proj_flow_abs = proj_flow.abs()
    proj_err_abs = proj_err.abs()
    proj_err2_abs = proj_err2.abs()
    # links (avg over time)
    flow_var = (proj_flow.var(dim=-1) + EPS).log()
    flow_var2 = (proj_err.var(dim=-1) + EPS).log()
    flow_var3 = (proj_err2.var(dim=-1) + EPS).log()
    flow_max = (torch.max(proj_flow_abs - A_noobs * HIGH, dim=-1)[0] + EPS).log()
    flow_max2 = (torch.max(proj_err_abs - A_noobs * HIGH, dim=-1)[0] + EPS).log()
    flow_max3 = (torch.max(proj_err2_abs - A_noobs * HIGH, dim=-1)[0] + EPS).log()
    flow_feat = torch.stack([flow_var, flow_var2, flow_var3, flow_max, flow_max2, flow_max3], dim=0).unsqueeze(-1).expand(6, *shape)

    # links (avg over links)
    time_var = (proj_flow.var(dim=-2) + EPS).log()
    time_var2 = (proj_err.var(dim=-2) + EPS).log()
    time_var3 = (proj_err2.var(dim=-2) + EPS).log()
    time_max = (torch.max(proj_flow_abs - A_noobs * HIGH, dim=-2)[0] + EPS).log()
    time_max2 = (torch.max(proj_err_abs - A_noobs * HIGH, dim=-2)[0] + EPS).log()
    time_max3 = (torch.max(proj_err2_abs - A_noobs * HIGH, dim=-2)[0] + EPS).log()
    time_feat = torch.stack([time_var, time_var2, time_var3, time_max, time_max2, time_max3], dim=0).unsqueeze(-2).expand(6, *shape)

    mu_log = mu_log.unsqueeze(0)

    err_pos_feat = (torch.relu(proj_err).unsqueeze(0) + EPS).log()
    err_neg_feat = (torch.relu(-proj_err).unsqueeze(0) + EPS).log()
    err2_pos_feat = (torch.relu(proj_err2).unsqueeze(0) + EPS).log()
    err2_neg_feat = (torch.relu(-proj_err2).unsqueeze(0) + EPS).log()
    Apos_feat = (torch.relu(A).unsqueeze(0) + EPS).log()
    Aneg_feat = (torch.relu(-A).unsqueeze(0) + EPS).log()
    mu_feat = torch.movedim(torch.cat([flow_feat, time_feat, observations,  # 13 features
                                       err_pos_feat, err_neg_feat, err2_pos_feat, err2_neg_feat, # 17
                                       Apos_feat, Aneg_feat, mu_log]), 0, -1)  # feature dim to last, 20 feats

    return mu_feat


def _mu_features_dynamic2(Y, R, Omega, X, A, mu_log):
    """Returns (batch_size, F, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
    """Set 22 features for now"""

    EPS = 1e-4
    HIGH = 1e18

    shape = A.shape

    A_scale = (R ** 2).mT @ Omega
    A_noobs = A_scale == 0
    A_scale_safezero = A_scale + A_noobs * 1
    proj_flow = R.mT @ (Y * Omega) / A_scale_safezero
    proj_err = R.mT @ ((Y - X) * Omega) / A_scale_safezero
    # proj_err2 = R.mT @ ((Y - X - R @ A) * Omega) / A_scale_safezero
    observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension

    # Ytime_var = _masked_mean_var(Y, mask=Omega, dim=[-2])[1]

    # time_var = proj_flow.var(dim=-2)
    # Ytime_var_log = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()

    if DEBUG_GLOBAL_FLOW:
        global scenA
        A = scenA
        Aabs = scenA.abs()
    else:
        Aabs = A.abs()

    proj_err_abs = proj_err.abs()

    proj_flow_flow_var = proj_flow.var(dim=-1).unsqueeze(-1) + EPS**2
    proj_flow_time_var = proj_flow.var(dim=-2).unsqueeze(-2) + EPS**2

    proj_err_flow_var = proj_err.var(dim=-1).unsqueeze(-1) + EPS**2
    proj_err_time_var = proj_err.var(dim=-2).unsqueeze(-2) + EPS**2

    A_flow_var = A.var(dim=-1).unsqueeze(-1) + EPS**2
    A_time_var = A.var(dim=-2).unsqueeze(-2) + EPS**2

    flow_proj_flow_var = proj_flow_flow_var.log() / 2
    flow_proj_err_var = proj_err_flow_var.log() / 2
    flow_Avar = A_flow_var.log() / 2
    flow_Amax = (torch.max(Aabs, dim=-1)[0].unsqueeze(-1) + EPS).log()

    flow_scpf_Amax = (torch.max(Aabs / proj_flow_time_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    flow_scpe_Amax = (torch.max(Aabs / proj_err_time_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    flow_scA_Amax = (torch.max(Aabs / A_time_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    flow_scpf_proj_err_max = (torch.max(proj_err_abs / proj_flow_time_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    flow_scpe_proj_err_max = (torch.max(proj_err_abs / proj_err_time_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()

    time_proj_flow_var = proj_flow_time_var.log() / 2
    time_proj_err_var = proj_err_time_var.log() / 2
    time_Avar = A_time_var.log() / 2
    time_Amax = (torch.max(Aabs, dim=-2)[0].unsqueeze(-2) + EPS).log()

    time_scpf_Amax = (torch.max(Aabs / proj_flow_flow_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    time_scpe_Amax = (torch.max(Aabs / proj_err_flow_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    time_scA_Amax = (torch.max(Aabs / A_flow_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    time_scpf_proj_err_max = (torch.max(proj_err_abs / proj_flow_flow_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    time_scpe_proj_err_max = (torch.max(proj_err_abs / proj_err_flow_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()

    flow_feat = torch.stack([flow_Avar, flow_Amax,
                             flow_proj_flow_var, flow_proj_err_var,
                             flow_scpf_Amax, flow_scpe_Amax, flow_scA_Amax,
                             flow_scpf_proj_err_max, flow_scpe_proj_err_max], dim=0).expand(9, *shape)

    # links (avg over links)
    time_feat = torch.stack([time_Avar, time_Amax,
                             time_proj_flow_var, time_proj_err_var,
                             time_scpf_Amax, time_scpe_Amax, time_scA_Amax,
                             time_scpf_proj_err_max, time_scpe_proj_err_max], dim=0).expand(9, *shape)

    padding = torch.zeros_like(A).unsqueeze(0)

    mu_log = mu_log.unsqueeze(0)
    mu_feat = torch.movedim(torch.cat([flow_feat, *([padding]*1),
                                       time_feat, *([padding]*1),
                                       observations, mu_log]), 0, -1)  # feature dim to last, 20 feats



    return mu_feat


def _mu_features_dynamic2i(Y, R, Omega, X, A, mu_log):
    """Returns (batch_size, F, T, num_features) feature tensor.
    Omega - expected to have all elements in {0, 1}"""
    """Set 22 features for now"""
    EPS = 1e-4
    HIGH = 1e15

    shape = A.shape
    Omega2 = Omega**2

    A_scale = (R ** 2).mT @ Omega2
    A_noobs = A_scale == 0
    A_scale_safezero = A_scale + A_noobs * 1
    # proj_flow = R.mT @ (Y * Omega) / A_scale_safezero
    proj_err = R.mT @ ((Y - X) * Omega2) / A_scale_safezero
    observations = (A_scale.unsqueeze(0) + EPS).log()  # prepending feature dimension

    # edge_corresp = R.mT.unsqueeze(-1) * Omega.unsqueeze(-3)  # dim (F, E, T)
    edge_corresp = R.mT.unsqueeze(-1)  # dim (F, E, 1)
    no_edge_corresp = edge_corresp == 0
    # unobserved = edge_corresp == 0
    reliable = torch.amin((R.mT.unsqueeze(-1) * Omega.unsqueeze(-3)) + no_edge_corresp * HIGH, dim=-2) != 0

    flows_corresp_to_ano = Y.unsqueeze(-3) / (edge_corresp + EPS)  # eps to avoid div by 0, division if routing matrix in [0, 1] instead of {0, 1}
    # min_corresp_flow_ind = torch.min(flows_corresp_to_ano.abs() + unobserved * HIGH, dim=-2, keepdim=True)[1]  # get only indices
    min_corresp_flow_ind = torch.min(flows_corresp_to_ano.abs() + no_edge_corresp * HIGH, dim=-2, keepdim=True)[1]  # get only indices
    min_corresp_flow = torch.gather(flows_corresp_to_ano, -2, min_corresp_flow_ind).squeeze(-2)  # collapse edge dim

    err_corresp_to_ano = (Y - X).unsqueeze(-3) / (edge_corresp + EPS)
    # min_corresp_err_ind = torch.min(err_corresp_to_ano.abs() + unobserved * HIGH, dim=-2, keepdim=True)[1]  # get only indices
    min_corresp_err_ind = torch.min(err_corresp_to_ano.abs() + no_edge_corresp * HIGH, dim=-2, keepdim=True)[1]
    min_corresp_err = torch.gather(err_corresp_to_ano, -2, min_corresp_err_ind).squeeze(-2)  # collapse edge dim

    Aabs = A.abs()
    proj_err_abs = proj_err.abs()

    # min_corresp_flow_mode1_var = min_corresp_flow.var(dim=-1).unsqueeze(-1) + EPS ** 2
    min_corresp_flow_mode1_var = _masked_mean_var(min_corresp_flow, mask=reliable, dim=-1)[1].unsqueeze(-1) + EPS ** 2
    # min_corresp_flow_mode2_var = min_corresp_flow.var(dim=-2).unsqueeze(-2) + EPS ** 2
    min_corresp_flow_mode2_var = _masked_mean_var(min_corresp_flow, mask=reliable, dim=-2)[1].unsqueeze(-2) + EPS ** 2

    # min_corresp_err_mode1_var = min_corresp_err.var(dim=-1).unsqueeze(-1) + EPS ** 2
    min_corresp_err_mode1_var = _masked_mean_var(min_corresp_err, mask=reliable, dim=-1)[1].unsqueeze(-1) + EPS ** 2
    # min_corresp_err_mode2_var = min_corresp_err.var(dim=-2).unsqueeze(-2) + EPS ** 2
    min_corresp_err_mode2_var = _masked_mean_var(min_corresp_err, mask=reliable, dim=-2)[1].unsqueeze(-2) + EPS ** 2

    proj_err_mode1_var = proj_err.var(dim=-1).unsqueeze(-1) + EPS ** 2
    proj_err_mode2_var = proj_err.var(dim=-2).unsqueeze(-2) + EPS ** 2

    A_mode1_var = A.var(dim=-1).unsqueeze(-1) + EPS ** 2
    A_mode2_var = A.var(dim=-2).unsqueeze(-2) + EPS ** 2

    # flow_proj_flow_var = proj_flow_flow_var.log() / 2
    mode1_min_corresp_flow_var = min_corresp_flow_mode1_var.log() / 2
    mode1_min_corresp_err_var = min_corresp_err_mode1_var.log() / 2
    mode1_proj_err_var = proj_err_mode1_var.log() / 2
    mode1_Avar = A_mode1_var.log() / 2

    # mode1_min_corresp_flow_max = (torch.max(min_corresp_flow.abs() / min_corresp_flow_mode2_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    # mode1_min_corresp_err_max = (torch.max(min_corresp_err.abs() / min_corresp_err_mode2_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    # mode1_proj_err_max = (torch.max(proj_err_abs / proj_err_mode2_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    # mode1_Amax = (torch.max(Aabs / A_mode2_var.sqrt(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    mode1_min_corresp_flow_max = (torch.max(min_corresp_flow.abs(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    mode1_min_corresp_err_max = (torch.max(min_corresp_err.abs(), dim=-1)[0].unsqueeze(-1) + EPS).log()
    mode1_proj_err_max = (torch.max(proj_err_abs, dim=-1)[0].unsqueeze(-1) + EPS).log()
    mode1_Amax = (torch.max(Aabs, dim=-1)[0].unsqueeze(-1) + EPS).log()

    mode2_min_corresp_flow_var = min_corresp_flow_mode2_var.log() / 2
    mode2_min_corresp_err_var = min_corresp_err_mode2_var.log() / 2
    mode2_proj_err_var = proj_err_mode2_var.log() / 2
    mode2_Avar = A_mode2_var.log() / 2

    # mode2_min_corresp_flow_max = (torch.max(min_corresp_flow.abs() / min_corresp_flow_mode1_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    # mode2_min_corresp_err_max = (torch.max(min_corresp_err.abs() / min_corresp_err_mode1_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    # mode2_proj_err_max = (torch.max(proj_err_abs / proj_err_mode1_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    # mode2_Amax = (torch.max(Aabs / A_mode1_var.sqrt(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    mode2_min_corresp_flow_max = (torch.max(min_corresp_flow.abs(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    mode2_min_corresp_err_max = (torch.max(min_corresp_err.abs(), dim=-2)[0].unsqueeze(-2) + EPS).log()
    mode2_proj_err_max = (torch.max(proj_err_abs, dim=-2)[0].unsqueeze(-2) + EPS).log()
    mode2_Amax = (torch.max(Aabs, dim=-2)[0].unsqueeze(-2) + EPS).log()

    mode1_feat = torch.stack([mode1_min_corresp_flow_var, mode1_min_corresp_err_var,
                              mode1_proj_err_var, mode1_Avar,
                              mode1_min_corresp_flow_max, mode1_min_corresp_err_max,
                              mode1_proj_err_max, mode1_Amax], dim=0).expand(8, *shape)

    mode2_feat = torch.stack([mode2_min_corresp_flow_var, mode2_min_corresp_err_var,
                              mode2_proj_err_var, mode2_Avar,
                              mode2_min_corresp_flow_max, mode2_min_corresp_err_max,
                              mode2_proj_err_max, mode2_Amax], dim=0).expand(8, *shape)

    padding = torch.zeros_like(A).unsqueeze(0)

    mu_log = mu_log.unsqueeze(0)
    mu_feat = torch.movedim(torch.cat([mode1_feat, *[padding]*2,
                                       mode2_feat, *[padding]*2,
                                       observations, mu_log]), 0, -1)  # feature dim to last, 20 feats

    return mu_feat


def _lam_features_dynamic(Y, R, Omega, X, A, prev_log=None):
    """Returns (batch_size, E, T, num_features) feature tensor.
    Features: Y var per link, err var per link, Y var per t, err var per t, num_routes per link, prev_dataweight
    Omega - expected to have all elements in {0, 1}"""

    EPS = 1e-3
    shape = Y.shape
    Omega = Omega > 0

    Yabs = Y.abs()

    err = Y - X - R @ A

    # per link
    link_var1 = (_masked_mean_var(Y, mask=Omega, dim=[-1])[1] + EPS).log()  # (B, E)
    link_var2 = (_masked_mean_var(err, mask=Omega, dim=[-1])[1] + EPS).log()
    mode1_feat = torch.movedim(torch.stack([link_var1, link_var2], dim=0).unsqueeze(-1), 0, -1)  # (*, E, 1, 2)

    # per time step
    time_var = (_masked_mean_var(Y, mask=Omega, dim=[-2])[1] + EPS).log()
    time_var2 = (_masked_mean_var(err, mask=Omega, dim=[-2])[1] + EPS).log()
    mode2_feat = torch.movedim(torch.stack([time_var, time_var2], dim=0).unsqueeze(-2), 0, -1)  # (*, 1, T, 2)

    # routing_feat = (R.sum(dim=-1) + EPS).log().unsqueeze(-1).expand(*shape).unsqueeze(0)

    # lam_feat = lam_log.expand(*shape).unsqueeze(0)
    # omega_feat = Omega.expand(*shape).unsqueeze(0)
    # datafit_weight_log = datafit_weight_log.unsqueeze(0)
    # datafit_feat = torch.movedim(torch.cat([link_feat, time_feat, routing_feat, datafit_weight_log]), 0, -1)  # feature dim to last

    # features are nonnegative, but may be 0

    return (mode1_feat, mode2_feat)

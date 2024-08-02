import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy.special
import random

import datagen
from config import FP_DTYPE, DFILEEXT



def set_rng_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)


def soft_thresholding(x, mu):
    """Elementwise soft thresholding."""
    res = torch.abs(x) - mu
    res[res < 0] = 0
    soft_x = torch.sign(x) * res
    return soft_x


class soft_thresholding_contgrad_cls(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mu):
        ctx.save_for_backward(x, mu)
        return soft_thresholding(x, mu)

    @staticmethod
    def backward(ctx, grad_xout):
        x, mu = ctx.saved_tensors
        res = torch.abs(x) - mu
        nonzero = res > 0

        grad_x = torch.zeros_like(grad_xout)
        grad_x[nonzero] = grad_xout[nonzero]

        grad_mu = torch.zeros_like(grad_xout)
        grad_mu[nonzero] = -torch.sign(x[nonzero])
        grad_mu[~nonzero] = -(x / mu)[~nonzero] / 100
        grad_mu2 = grad_mu * _broadcast_shape_sum(grad_mu, grad_xout)

        return grad_x, grad_mu2


def _broadcast_shape_sum(small, big):
    """Sums all dimensions of big which small would need to broadcast to. Does not check validity of broadcast."""
    dims = []
    for idx in range(1, len(small.shape)+1):
        if small.shape[-idx] != big.shape[-idx]:
            dims.append(-idx)
    for idx in range(len(small.shape)+1, len(big.shape)+1):
        dims.append(-idx)

    if len(dims) > 0:
        big = torch.sum(big, dim=dims)
    big = big.reshape(small.shape)

    return big


def soft_thresholding_contgrad(x, mu):
    return soft_thresholding_contgrad_cls.apply(x, mu)


def l1_norm(x, dim):
    return torch.sum(torch.abs(x), dim=dim)


def frob_norm_sq(x, dim):
    return torch.sum(torch.abs(x).square(), dim=dim)


def min_diff_machine_precision(x):
    """Find some small number to add to still have difference within machine precision (FP32)."""
    fp_exp = torch.frexp(x)[1]
    min_diff = 2.0 ** (fp_exp - 23)
    return min_diff


def adjust_lightness(color, amount=0.5):
    """From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def heavyside_cheby_coeff(degree):
    def chebychev_poly(n):
        assert(n > 1)
        coeff = torch.zeros(n+1, n+1)
        coeff[0, 0] = 1
        coeff[1, 1] = 1
        for i in range(2, n+1):
            coeff[i] = torch.cat((torch.zeros(1), 2*coeff[i-1, :-1])) - coeff[i-2]

        return coeff

    def heavyside(x):
        y = torch.zeros_like(x)
        y[x > 0] = 1
        y[x == 0] = 1/2
        return y

    jj = torch.arange(degree+1)
    in_cos = ((jj + 1/2)*torch.pi / (degree+1)).unsqueeze(-1)
    cheb_interp_coeff = heavyside(torch.cos(in_cos)) * torch.cos(jj * in_cos)
    cheb_interp_coeff = cheb_interp_coeff.sum(dim=0) * 2 / (degree+1)

    cheb_poly_coeff = chebychev_poly(degree)

    hs_poly_coeff = cheb_poly_coeff * cheb_interp_coeff.unsqueeze(-1)
    hs_poly_coeff = hs_poly_coeff.sum(dim=0)  # sum over cheb polys
    hs_poly_coeff[0] -= cheb_interp_coeff[0] / 2

    return hs_poly_coeff


def heavyside_monotonic_coeff(ord):
    temp = np.array([-1, 0, 1])
    qdash = np.array([-1, 0, 1])
    for i in range(ord-1):
        qdash = np.convolve(qdash, temp)
    q = np.concatenate((np.array([0]), qdash / np.arange(1, len(qdash)+1)))
    q1 = q.sum()
    coeff = torch.tensor(q / 2 / q1)
    coeff[0] = 1/2

    return coeff


def exact_auc(scores, anomalies, batch_mean=True):
    """
    Computes the exact area under curve. No gradient.
    :param scores: (*, M, N), can be negative since absolute values will be used anyways.
    :param anomalies: (*, M, N) broadcastable to scores.shape
    :param batch_mean:
    :return:
    """

    sh = scores.shape
    # assert(len(scores.shape) == 4)
    assert(scores.shape[-3:] == anomalies.shape[-3:])
    anomalies = anomalies.abs() > 0

    num_blocks = torch.prod(torch.tensor(sh[:-2]))
    scores = scores.abs().flatten(start_dim=-2).reshape(num_blocks, -1)
    anomalies = anomalies.expand(*sh)
    anomalies = anomalies.flatten(start_dim=-2).reshape(num_blocks, -1)

    num_anomalies = anomalies.sum(dim=-1)
    num_non_anomalies = (~anomalies).sum(dim=-1)

    auc = torch.zeros(num_blocks, dtype=torch.float, device=scores.device)
    for i in range(num_blocks):
        # print("Block {} of {}".format(i+1, num_blocks))
        comp = scores[i][anomalies[i]].unsqueeze(-1) > scores[i][~anomalies[i]]
        auc_temp = comp.sum(dim=(-2, -1)) / num_non_anomalies[i] / num_anomalies[i]
        # comp_zero_zero_correction = 0
        comp_equal_score_correction = (scores[i][anomalies[i]].unsqueeze(-1) == scores[i][~anomalies[i]])
        comp_equal_score_correction = comp_equal_score_correction.sum(dim=(-2, -1)) / 2 / num_non_anomalies[i] / num_anomalies[i]
        auc_temp = auc_temp + comp_equal_score_correction
        auc[i] = auc_temp
    auc = auc.reshape(sh[:(-2)])

    if batch_mean:
        auc = auc.mean(dim=-1)

    return auc


# def auc_diff_distr(scores, anomalies, norm=True):
#     sh = scores.shape
#     assert (len(scores.shape) == 4)
#     assert (scores.shape[-3:] == anomalies.shape[-3:])
#     anomalies = anomalies.abs() > 0
#
#     num_blocks = scores.shape[0] * scores.shape[1]
#     scores = scores.abs().flatten(start_dim=-2).reshape(num_blocks, -1)
#     anomalies = anomalies.expand(*sh)
#     anomalies = anomalies.flatten(start_dim=-2).reshape(num_blocks, -1)
#
#     scores = scores / scores.max(dim=-1)[0].unsqueeze(-1)
#     comps = []
#     ests = []
#     exs = []
#     for i in range(num_blocks):
#         print("Block {} of {}".format(i + 1, num_blocks))
#         comp = scores[i][anomalies[i]].unsqueeze(-1) - scores[i][~anomalies[i]]
#         guilty = (scores[i][anomalies[i]].unsqueeze(-1) > 0) * (scores[i][~anomalies[i]] == 0)
#         est = (scores[i][anomalies[i]] > 0).sum() * (scores[i][~anomalies[i]] == 0).sum()
#         est = est / (anomalies[i].sum() * (~anomalies[i]).sum())
#         exs.append(guilty.sum() / (anomalies[i].sum() * (~anomalies[i]).sum()))
#
#         ests.append(est)
#         comp = comp[guilty]
#         comps.append(comp.flatten())
#
#     comps = torch.cat(comps)
#     ests = torch.stack(ests)
#     exs = torch.stack(exs)
#     plt.hist(comps, bins="auto")
#     plt.show()
#     return


def binomial_coefficient(n, k):
    return scipy.special.comb(n, k, exact=False)


def lstsq(A, B):
    # Solves minX ||B - XA||_F^2. Required due to very bad conditioning
    X = torch.linalg.lstsq(A, B, driver="gelsd")[0]
    return X


# def _masked_mean_var(x, mask=None, dim=(-2, -1)):
#     if mask is None:
#         mean, var = x.mean(dim=dim), x.var(dim=dim)
#     else:
#         assert(mask.shape == x.shape)
#         mask_sum = mask.sum(dim=dim)
#         mask_sum = torch.clamp(mask_sum, min=2)
#         mean = (x * mask).sum(dim=dim) / mask_sum
#         var_temp = (x - mean.view(*mean.shape, *(len(dim)*[1]))) * mask
#         var = (var_temp ** 2).sum(dim=dim) / (mask_sum - 1)
#
#     return mean, var


def _masked_mean_var(x, mask=None, dim=(-2, -1)):
    if mask is None:
        mean, var = x.mean(dim=dim), x.var(dim=dim)
    else:
        assert (mask.shape == x.shape)
        mask = mask.to(torch.bool)
        # assert (mask.dtype == torch.bool)
        mask_sum = mask.sum(dim=dim, keepdims=True)
        mask_sum = torch.clamp(mask_sum, min=2)
        mean = (x * mask).sum(dim=dim, keepdims=True) / mask_sum
        var_temp = (x - mean) * mask
        var = (var_temp ** 2).sum(dim=dim, keepdims=True) / (mask_sum - 1)

        # super inefficient but I am lazy and don't want to upgrade to 2.0
        temp = mask.sum(dim=dim).shape
        mean, var = mean.view(*temp), var.view(*temp)

    return mean, var


def mask_vars(mask, *args):
    masked_args = []
    for arg in args:
        masked_args.append(arg[mask])

    return masked_args


# for testing
def _fm_corr(U):
    # U must be matrix
    cc = U.mT @ U
    cn = ((U**2).sum(dim=-2)).sqrt()
    cnc = cn.unsqueeze(-1) * cn.unsqueeze(-2)
    normcorr = cc / (cnc + 1e-6)
    normcorr.fill_diagonal_(0)

    maxind = np.unravel_index(normcorr.abs().argmax(), normcorr.shape)
    max_val = normcorr[maxind]
    print("Highest correlation {} at between {} and {}".format(max_val, maxind[0], maxind[1]))
    print("Factor {} has normsq {}".format(maxind[0], cn[maxind[0]]))
    print("Factor {} has normsq {}".format(maxind[1], cn[maxind[1]]))
    print("Factor {} has max normsq {}".format(cn.argmax(), cn[cn.argmax()]))


def _fm_corr_ind(U, ind0, ind1):
    # U must be matrix
    cc = U.mT @ U
    cn = ((U**2).sum(dim=-2)).sqrt()
    cnc = cn.unsqueeze(-1) * cn.unsqueeze(-2)
    normcorr = cc / (cnc + 1e-6)
    normcorr.fill_diagonal_(0)

    # maxind = np.unravel_index(normcorr.abs().argmax(), normcorr.shape)
    # max_val = normcorr[maxind]
    print("Correlation {} at between {} and {}".format(normcorr[ind0, ind1], ind0, ind1))
    print("Factor {} has normsq {}".format(ind0, cn[ind0]))
    print("Factor {} has normsq {}".format(ind1, cn[ind1]))


def _fm_corr_all(U, V, W):
    # U must be matrix
    R = U.shape[-1]
    prod = torch.ones(R, R)
    for X in (U, V, W):
        cc = X.mT @ X
        cn = ((X**2).sum(dim=-2)).sqrt()
        cnc = cn.unsqueeze(-1) * cn.unsqueeze(-2)
        normcorr = cc / (cnc + 1e-6)
        normcorr.fill_diagonal_(0)
        prod = prod * normcorr

    prod = prod.numpy()

    return prod


class RegParamMLP(nn.Module):
    def __init__(self, feat_sizes, batch_norm_input=False, bias=True, init=None, device=None):
        super().__init__()
        assert (isinstance(feat_sizes, list))
        assert (len(feat_sizes) > 1)  # At least input and output size required

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")
        self.feat_sizes = feat_sizes
        if batch_norm_input:
            self.batch_norm_input = nn.BatchNorm1d(self.feat_sizes[0], affine=False, momentum=0.01, device=self.device)
        else:
            self.batch_norm_input = None
        self.layers = []
        for l in range(len(self.feat_sizes) - 1):
            self.layers.append(nn.Linear(self.feat_sizes[l], self.feat_sizes[l + 1], bias=bias, device=self.device))

            if init:
                if init == "uniform_small":
                    """Smaller initialization compared to default. Bias initialized with 0."""
                    in_feat = self.layers[l].weight.shape[-1]
                    torch.nn.init.uniform_(self.layers[l].weight, a=-1/in_feat, b=+1/in_feat)
                    if bias:
                        torch.nn.init.constant_(self.layers[l].bias, 0.0)
                else:
                    raise ValueError
            else:
                pass  # default initialization

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = x.to(torch.float32)
        if self.batch_norm_input:
            shape = x.shape
            x = self.batch_norm_input(x.view(-1, shape[-1])).view(*shape)
        for l in range(len(self.layers)):
            x = self.layers[l](x)
            if not l == len(self.layers) - 1:
                x = torch.relu(x)

        return x.to(FP_DTYPE)


def retrieve_data(data_path_or_splitdict):
    if isinstance(data_path_or_splitdict, dict):
        data = torch.load(data_path_or_splitdict["path"])
        if "idx" in data_path_or_splitdict.keys():
            data = datagen.nw_scenario_subset(data, data_path_or_splitdict["idx"])
    else:
        data = torch.load(data_path_or_splitdict + DFILEEXT)  # OLD

    return data
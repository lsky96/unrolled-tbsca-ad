"""
Author: 
Lukas Schynol
lschynol@nt.tu-darmstadt.de
"""
import torch
import numpy as np


def hankel_transform(mats, W):
    # Also called multi delay transform, here convention: delay in last dimension, W in second to last
    # to reverse dimensions last dimensions, swap unsqueeze-ops and W and K in view(...)
    # Backgradient not tested.
    ldim = mats.shape[:-1]
    rdim = mats.shape[-1]
    K = rdim - W + 1
    unrolled_indices = (torch.arange(W).unsqueeze(-1) + torch.arange(K).unsqueeze(0)).flatten()
    hankel_tensors = mats[..., unrolled_indices].view(*ldim, W, K)
    # hankel_tensors = hankel_tensors.view(*ldim, W, K)
    return hankel_tensors


def inverse_hankel_transform(tens, W):
    # Implemented using a loop. Backgradient not tested. Just use small W, okay?
    assert(tens.shape[-2] == W)
    ldim = tens.shape[:-2]
    K = tens.shape[-1]
    rdim = K + W - 1

    inv_hankel_mats = torch.zeros(*(*ldim, rdim), dtype=tens.dtype, device=tens.device)
    for w in range(W):
        inv_hankel_mats[..., w:(w+K)] = inv_hankel_mats[..., w:(w+K)] + tens[..., w, :]
    scaling = torch.cat([torch.arange(1, W), torch.full([rdim - 2*W + 2], W), torch.arange(W-1, 0, -1)])
    inv_hankel_mats = inv_hankel_mats / scaling

    return inv_hankel_mats


def kron_mat(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = tuple(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a[..., np.newaxis, :, np.newaxis] * b[..., np.newaxis, :, np.newaxis, :]
    siz0 = res.shape[:-4]
    return res.view(siz0 + siz1)


def kron_vec(a, b):
    # Kronecker in last dimension
    siz0 = a.shape[:-1]
    siz1 = torch.tensor(a.shape[-1] * b.shape[-1])
    res = a.unsqueeze(-1) * b.unsqueeze(-2)

    return res.view(*siz0, siz1)


def tensor2mat_unfold(a, mode):
    """Unfolds a batch of 3D tensors into 2D matrices along mode."""
    siz0 = a.shape[:-3]
    if not mode == 1:
        a = a.moveaxis(-4+mode, -3)
    a_swp = a.swapaxes(-2, -1)
    siz1 = a_swp.shape[-3]
    return a_swp.reshape(*siz0, siz1, -1)  # view does not work for mode 2 and mode 3


def tensor2vec(a):
    """Unfolds a batch of 3D tensors into vector."""
    siz1 = a.shape[:-3]
    # res = a.movedim((-3, -2, -1), (-1, -2, -3)).reshape(*siz1, -1)
    return a.view(*siz1, -1)  # view does not work for mode 2 and mode 3
    # return res


def vec2tensor(a, shape):
    """Unfolds a batch of 3D tensors into 2D matrices along mode."""
    siz1 = a.shape[:-1]
    # res = a.reshape(*siz1, *shape[::-1]).movedim((-3, -2, -1), (-1, -2, -3))
    return a.view(*siz1, *shape)  # view does not work for mode 2 and mode 3
    # return res


def khatri_rao(A, B):
    # Effectively kronecker in second to last dimension
    siz0 = A.shape[:-2]
    siz1 = A.shape[-2] * B.shape[-2]
    siz2 = A.shape[-1]
    res = A.unsqueeze(-2) * B.unsqueeze(-3)

    return res.view(*siz0, siz1, siz2)


def cpd(U, V, W):
    # Rank dimension is last
    rank = U.shape[-1]
    bdim = U.shape[:-2]
    tensor = U.view(*bdim, U.shape[-2], 1, 1, rank) * V.view(*bdim, 1, V.shape[-2], 1, rank) * W.view(*bdim, 1, 1, W.shape[-2], rank)
    tensor = tensor.sum(dim=-1)
    return tensor


def tucker(G, U, V, W):
    res = U.unsqueeze(-3) @ G.movedim(-1, -3)
    res = W.unsqueeze(-3) @ res.movedim(-1, -3)
    res = V.unsqueeze(-3) @ res.movedim(-1, -3)
    return res


def tucker_unitycore(U, V, W):
    res = cpd(U.sum(dim=-1, keepdims=True), V.sum(dim=-1, keepdims=True), W.sum(dim=-1, keepdims=True))
    return res


def vectorize(X):
    siz1 = X.shape[:-2]
    siz2 = X.shape[-2:]
    vecX = X.swapaxes(-2, -1).reshape(siz1 + (-1,) + (1,))
    return vecX, siz2


def devectorize(vecX, sizemat):
    sizebatch = vecX.shape[:-2]
    X = vecX.reshape(sizebatch + sizemat).swapaxes(-2, -1)
    return X

def llmat(M, dtype=torch.double):
    n = (M - 1) * M / 2
    # temp = torch.floor(0.5 + torch.sqrt(0.25 + 2 * torch.flip(torch.arange(0, n), [-1])))   # old wrong llmat
    # indl_tran = M*M - (torch.flip(torch.arange(0, n), [-1]) + temp * (M - (temp - 1) / 2)).long() - 1  # old wrong llmat
    temp = torch.floor(0.5 + torch.sqrt(0.25 + 2 * (torch.arange(0, n))))
    indl_tran = (torch.arange(0, n) + temp * (M - (temp - 1) / 2)).long()
    ind_map = torch.arange(M*M).reshape(M, M).swapaxes(-2, -1).reshape(-1)
    indl = ind_map[indl_tran]
    Ll = torch.zeros((M*M, len(indl)))
    Ll[indl, torch.arange(0, len(indl))] = 1
    return Ll.to(dtype), indl


def lumat(M, dtype=torch.double):
    n = (M - 1) * M / 2
    temp = torch.floor(0.5 + torch.sqrt(0.25+2*(torch.arange(0, n))))
    indu = (torch.arange(0, n) + temp * (M-(temp-1)/2)).long()
    # print(temp, indu)
    Lu = torch.zeros((M*M, len(indu)))
    Lu[indu, torch.arange(0, len(indu))] = 1
    return Lu.to(dtype), indu


def ldmat(M, dtype=torch.double):
    n = M
    indd = torch.arange(0, n) * (M+1)
    Lu = torch.zeros((M*M, len(indd)))
    Lu[indd, torch.arange(0, len(indd))] = 1
    return Lu.to(dtype), indd


# test4()
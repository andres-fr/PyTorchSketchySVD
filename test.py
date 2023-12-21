#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


# import numpy as np
import torch
import torch_dct as dct

from pyssvd.utils import randperm, rademacher

# import scipy
# from scipy.sparse.linalg import LinearOperator

import matplotlib.pyplot as plt


#


DEVICE = "cuda"
SEED = 12345
HEIGHT, WIDTH = 5, 100_000

# aa = torch.linspace(0, 10, WIDTH).to(DEVICE)
# aa = aa[randperm(WIDTH, seed=SEED, device="cpu")]
# rademacher(aa, SEED, inplace=True)
# aa = dct.dct(aa)
# #
# aa = aa[randperm(WIDTH, seed=SEED + 1, device="cpu")]
# rademacher(aa, SEED + 1, inplace=True)
# aa = dct.dct(aa)


# rademacher(x, seed=None, inplace=True):
# rademacher(WIDTH, seed=None, inplace=True)


def ssrft(t, out_dims, seed=0b1110101001010101011, dct_norm="ortho"):
    """
    map = R @ F @ PI @ F @ PI'

    where R is an index-picker, F is a DCT, and PI, PI' are independent permutations

    :param int out_dims: If smaller than

    """
    assert len(t.shape) == 1, "Only flat tensors supported!"
    t_len = len(t)
    assert out_dims <= t_len, "Projection to larger dimensions not supported!"
    seeds = [seed + i for i in range(5)]
    # first scramble: permute, rademacher, and DCT
    perm1 = randperm(t_len, seed=seeds[0], device="cpu")
    t, rad1 = rademacher(t[perm1], seed=seeds[1], inplace=False)
    # del perm1, rad1
    t = dct.dct(t, norm=dct_norm)
    # second scramble: permute, rademacher and DCT
    perm2 = randperm(t_len, seed=seeds[2], device="cpu")
    t, rad2 = rademacher(t[perm2], seeds[3], inplace=False)
    # del perm2, rad2
    t = dct.dct(t, norm=dct_norm)
    # extract random indices and return
    out_idxs = randperm(t_len, seed=seeds[4], device="cpu")[:out_dims]
    t = t[out_idxs]

    print(perm1, rad1)
    print(perm2, rad2)
    print(out_idxs)
    return t


def ssrft_adjoint(t, out_dims, seed=0b1110101001010101011, dct_norm="ortho"):
    """
    map = R @ F @ PI @ F @ PI'

    where R is an index-picker, F is a DCT, and PI, PI' are independent permutations

    :param int out_dims: If smaller than

    """

    assert len(t.shape) == 1, "Only flat tensors supported!"
    t_len = len(t)
    assert (
        out_dims >= t_len
    ), "Backprojection into smaller dimensions not supported!"
    #
    seeds = [seed + i for i in range(5)]
    result = torch.zeros(
        out_dims,
        dtype=t.dtype,
    ).to(t.device)
    # first embed signal into original indices
    out_idxs = randperm(out_dims, seed=seeds[4], device="cpu")[:t_len]
    result[out_idxs] = t
    # then do the idct, followed by rademacher and inverse permutation
    result = dct.idct(result, norm=dct_norm)
    rademacher(result, seeds[3], inplace=True)
    perm2_inv = randperm(out_dims, seed=seeds[2], device="cpu", inverse=True)
    result = result[perm2_inv]
    del perm2_inv
    # second inverse pass
    result = dct.idct(result, norm=dct_norm)
    rademacher(result, seeds[1], inplace=True)
    perm1_inv = randperm(out_dims, seed=seeds[0], device="cpu", inverse=True)
    result = result[perm1_inv]
    #
    return result


IN, OUT, DTYPE = 100, 100, torch.float64
aa = torch.linspace(0, 10, IN, dtype=DTYPE).to(DEVICE)
oo = ssrft(aa, OUT)

bb = ssrft_adjoint(oo, IN)

print(aa)
print(oo)

# t=aa.cpu(); plt.clf(); plt.plot(t); plt.show()
breakpoint()


class SSRFT:
    """
    TODO:
    implement forward and adjoint pass, purely functional except for seed.

    TODO: DCT MUST PRESERVE SCALE, AS PERM AND RADEMACHER. HOW?


    """

    @classmethod
    def fjlt(
        cls, t, out_dims=None, seed=None, cached_idxs=None, cached_noise=None
    ):
        """
        Implementation of the Fast JLT as ``y = proj@hadamard@rademacher @ t``.
        :param t: Flat tensor of length ``2^k``.
        :param out_dims: See ``sparseproj`` docstring.
        :param cached_idxs: See ``sparseproj`` docstring.
        :param cached_noise: See ``sparseproj`` docstring.
        :returns: The tuple ``(y, mask, idxs, noise)``, where ``y`` is a flat
          tensor of same dtype and device as the input ``t``, with ``out_dims``
          entries corresponding to the Fast JLT of the input. The tensors
          ``mask, idxs, noise`` correspond to the parameters returned by the
          rademacher noise and sparse projection.
        """
        assert len(t.shape) == 1, "Only flat tensors supported!"
        in_len = len(t)
        assert in_len == cls.pow2roundup(
            in_len
        ), "Only inputs with pow2 elements supported!"
        #
        t, mask = cls.rademacher(t, seed=seed, inplace=False)
        cls.hadamard(t, inplace=True)
        #
        result, idxs, noise = cls.sparseproj(
            t, out_dims, seed, cached_idxs, cached_noise
        )
        return result, mask, idxs, noise

    @classmethod
    def transp_fjlt(
        cls,
        y,
        out_dims,
        seed=None,
        cached_idxs=None,
        cached_noise=None,
        out=None,
    ):
        """
        Implementation of the conjugate Fast JLT as
        ``x = (proj@hadamard@rademacher).T @ y``.

        :param y: Flat tensor of arbitrary length.
        :param out_dims: A power of 2 determining the output dimensionality
          (i.e. the row width of the Hadamard-Walsh operator).
        :param cached_idxs: Indices used in the forward projection, determining
          the column height of the Hadamard-Walsh operator. See ``sparseproj``
          docstring.
        :param cached_noise: Noise used in the forward projection. See
          ``sparseproj`` docstring.
        :param out: If given, flat tensor of shape ``(out_dims,)`` where the
          output will be written. Otherwise, a new tensor will be created.
        :returns: The tuple ``(x, mask, idxs, noise)`` (see ``fjlt``).
        """
        assert len(y.shape) == 1, "Only flat inputs supported!"
        assert out_dims == cls.pow2roundup(
            out_dims
        ), "Only out_dims == 2^k supported!"
        in_dims = len(y)
        # recreate projection parameters
        if cached_idxs is not None:
            idxs = cached_idxs
        else:
            idxs = cls.get_idxs(in_dims, out_dims, seed, y.device)
        #
        if cached_noise is not None:
            noise = cached_noise
        else:
            noise = cls.get_noise(in_dims, out_dims, 1, seed, y.dtype, y.device)
            assert noise.all(), "Noise contains zeros! Not allowed."
        # To invert projection, embed y and multiply using idxs and noise
        if out is None:
            out = torch.zeros(out_dims, dtype=y.dtype).to(y.device)
        else:
            out *= 0
        out[idxs] = y * noise
        # the inverse of Hadamard and Rademacher are themselves
        cls.hadamard(out, inplace=True)
        _, mask = cls.rademacher(out, seed=seed, inplace=True)
        #
        return out, mask, idxs, noise

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

aa = torch.linspace(0, 10, WIDTH).to(DEVICE)
aa = aa[randperm(WIDTH, seed=SEED, device="cpu")]
rademacher(aa, SEED, inplace=True)
aa = dct.dct(aa)
#
aa = aa[randperm(WIDTH, seed=SEED + 1, device="cpu")]
rademacher(aa, SEED + 1, inplace=True)
aa = dct.dct(aa)

# rademacher(x, seed=None, inplace=True):
# rademacher(WIDTH, seed=None, inplace=True)

breakpoint()


class SSRFT:
    """
    TODO:
    implement forward and adjoint pass, purely functional except for seed.

    TODO: DCT MUST PRESERVE SCALE, AS PERM AND RADEMACHER. HOW?


    map = R @ F @ PI @ F @ PI'

    where R is an index-picker, F is a DCT, and PI, PI' are independent permutations
    """

    SEED = 0b1110101001010101011

    # def __init__(
    #     self, shape, dtype=torch.float64, device="cpu", scale=1.0, seed=None
    # ):
    #     """ """
    #     coords = None
    #     perm1, perm2 = None, None
    #     eps1, eps2 = None, None

    @classmethod
    def ssrft(cls, t, out_dims):
        """ """
        pass

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

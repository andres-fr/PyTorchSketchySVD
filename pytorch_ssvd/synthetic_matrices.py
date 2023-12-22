#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
Section 7.3.1. from the paper

TTODO:
1. Implement the 3 types of matrices. We also want them symmetric!
2. Figure a way to test or verify them, e.g. a plot?

Once we are done, we can start implementing the SVD, with 2 zwischenstops:
1. Create a symm matrix
2. Do left rand measurement and sketch (also symmetric)
3. Do QR of left measurement (IMPLEMENT/TEST) and solve leas squares of core via CG (IMPLEMENT/TEST)
4. Put everything together to get the SVD
"""

import torch

from .base_matrix import BaseMatrix
from .utils import normal_noise


# ##############################################################################
# # ERRORS
# ##############################################################################
class BaseSyntheticMatrix(BaseMatrix):
    """ """

    def __init__(self, shape=(100, 100), **kwargs):
        """ """
        assert len(shape) == 2, "Matrix shape must be 2 numbers!"
        self.shape = shape
        #
        self._weights = self.build(shape, **kwargs)

    def matmul(self, x):
        """ """
        result = torch.matmul(self._weights, x)
        return result

    def rmatmul(self, x):
        """ """
        result = torch.matmul(x, self._weights)
        return result

    @staticmethod
    def build(shape, **kwargs):
        """ """
        raise NotImplementedError


# ##############################################################################
# #
# ##############################################################################
class LowRankNoiseMatrix(BaseSyntheticMatrix):
    """ """

    def __init__(
        self,
        shape=(100, 100),
        rank=10,
        snr=1e-4,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """ """
        assert shape[0] == shape[1], "LowRankNoiseMatrix must be square!"
        super().__init__(
            shape=shape,
            rank=rank,
            snr=snr,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        self.rank = rank
        self.snr = snr
        self.seed = seed
        self.dtype = dtype
        self.device = device

    def __repr__(self):
        """ """
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}), "
            + f"rank={self.rank}, snr={self.snr}>"
        )
        return s

    @staticmethod
    def build(shape, rank, snr, seed, dtype, device):
        """ """
        # create matrix as a scaled outer product of Gaussian noise
        result = normal_noise(
            shape,
            mean=0.0,
            std=1.0,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        result = (snr / shape[0]) * (result @ result.T)
        # add 1 to the first "rank" diagonal entries
        result[range(rank), range(rank)] += 1
        return result


# class PolyDecayMatrix(BaseSyntheticMatrix):
#     """ """

#     def __init__(self):
#         """ """
#         if symmetric:
#             assert shape[0] == shape[1], "Symmetric matrix must be square!"


# G = np.random.randn(*H.shape)
# S = np.tril(G) + np.tril(G, -1).T  # random symmetric matrix
# Q, _ = np.linalg.qr(G)  # random eigenspace
# ew, ev = np.linalg.eigh(H)
# # ews = np.linalg.eigvalsh(H)
# R = Q @ np.diag(ew) @ Q.T  # random matrix with same spectrum as H

# breakpoint()

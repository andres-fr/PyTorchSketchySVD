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
# # BASE SYNTH MATRIX
# ##############################################################################
class BaseSyntheticMatrix(BaseMatrix):
    """ """

    def __init__(
        self,
        shape=(100, 100),
        rank=10,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """ """
        assert len(shape) == 2, "Matrix shape must be 2 numbers!"
        self.shape = shape
        self.rank = rank
        self.seed = seed
        self.dtype = dtype
        self.device = device
        #
        self._weights = self.build()

    def __repr__(self, extra=""):
        """ """
        clsname = self.__class__.__name__
        s = (
            f"<{clsname}({self.shape[0]}x{self.shape[1]}), "
            + f"rank={self.rank}, seed={self.seed}{extra}>"
        )
        return s

    def matmul(self, x):
        """ """
        result = torch.matmul(self._weights, x)
        return result

    def rmatmul(self, x):
        """ """
        result = torch.matmul(x, self._weights)
        return result

    def build(self):
        """ """
        raise NotImplementedError


# ##############################################################################
# # LOWRANK+NOISE MATRIX
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
        self.snr = snr
        super().__init__(
            shape=shape,
            rank=rank,
            seed=seed,
            dtype=dtype,
            device=device,
        )

    def __repr__(self):
        """ """
        return super().__repr__(extra=f", snr={self.snr}")

    def build(self):
        """ """
        # create matrix as a scaled outer product of Gaussian noise
        result = normal_noise(
            self.shape,
            mean=0.0,
            std=1.0,
            seed=self.seed,
            dtype=self.dtype,
            device=self.device,
        )
        result = (self.snr / self.shape[0]) * (result @ result.T)
        # add 1 to the first "rank" diagonal entries
        result[range(self.rank), range(self.rank)] += 1
        return result


# ##############################################################################
# # POLY-DECAY MATRIX
# ##############################################################################
class PolyDecayMatrix(BaseSyntheticMatrix):
    """ """

    def __init__(
        self,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """ """
        assert shape[0] == shape[1], "LowRankNoiseMatrix must be square!"
        self.decay = decay
        self.symmetric = symmetric
        if symmetric:
            assert shape[0] == shape[1], "Symmetric matrix must be square!"
        #
        super().__init__(
            shape=shape,
            rank=rank,
            seed=seed,
            dtype=dtype,
            device=device,
        )

    def make_singular_vals(self):
        """ """
        min_shape = min(self.shape)
        # a few ones, followed by a poly decay
        svals = torch.zeros(min_shape, dtype=self.dtype).to(self.device)
        svals[: self.rank] = 1
        svals[self.rank :] = torch.arange(2, min_shape - self.rank + 2) ** (
            -float(self.decay)
        )
        return svals

    def build(self):
        """ """
        min_shape = min(self.shape)
        # build the singular values
        svals = self.make_singular_vals()
        # build singular bases using QR subgroup algorithm (Diaconis). QR is not
        # fastest, but these are test matrices so speed is not crucial.
        G_left = normal_noise(
            (self.shape[0], min_shape),
            mean=0.0,
            std=1.0,
            seed=self.seed,
            dtype=self.dtype,
            device=self.device,
        )
        U, _ = torch.linalg.qr(G_left)
        del G_left
        #
        if self.symmetric:
            result = U @ torch.diag(svals) @ U.T
        else:
            G_right = normal_noise(
                (min_shape, self.shape[1]),
                mean=0.0,
                std=1.0,
                seed=self.seed + 1,
                dtype=self.dtype,
                device=self.device,
            )
            V, _ = torch.linalg.qr(G_right)
            result = U @ torch.diag(svals) @ V
        #
        return result


# ##############################################################################
# # EXP-DECAY MATRIX
# ##############################################################################
class ExpDecayMatrix(PolyDecayMatrix):
    """ """

    def make_singular_vals(self):
        """ """
        min_shape = min(self.shape)
        # a few ones, followed by exp decay
        svals = torch.zeros(min_shape, dtype=self.dtype).to(self.device)
        svals[: self.rank] = 1
        svals[self.rank :] = 10
        #
        svals[self.rank :] **= -(
            self.decay * torch.arange(1, min_shape - self.rank + 1)
        )
        return svals

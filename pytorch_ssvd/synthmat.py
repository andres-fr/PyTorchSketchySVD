#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
Section 7.3.1. from the SSVD paper
"""

import torch

from .utils import normal_noise


# ##############################################################################
# # SYNTH MATRIX FACTORY
# ##############################################################################
class SynthMat:
    """ """

    @staticmethod
    def lowrank_noise(
        shape=(100, 100),
        rank=10,
        snr=1e-4,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """
        :param snr: Paper values are 1e-4 for low noise, 1e-2 for mid noise,
          and 1e-1 for high noise.
        """
        h, w = shape
        assert h == w, "lowrank_noise must be square! (and symmetric)"
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

    @staticmethod
    def _decay_helper(
        svals,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """
        Helper method for the polynomial and exp-decay matrices.
        """
        min_shape = min(shape)
        # build singular bases using QR subgroup algorithm (Diaconis). QR is not
        # fastest, but these are test matrices so speed is not crucial.
        G_left = normal_noise(
            (shape[0], min_shape),
            mean=0.0,
            std=1.0,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        U, _ = torch.linalg.qr(G_left)
        del G_left
        #
        if symmetric:
            result = U @ torch.diag(svals) @ U.T
        else:
            G_right = normal_noise(
                (shape[1], min_shape),
                mean=0.0,
                std=1.0,
                seed=seed + 1,
                dtype=dtype,
                device=device,
            )
            V, _ = torch.linalg.qr(G_right)
            result = U @ torch.diag(svals) @ V.T
        #
        return result

    @classmethod
    def poly_decay(
        cls,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """
        :param decay: Paper values are 0.5 for slow decay, 1 for medium, and
          2 for fast.
        """
        min_shape = min(shape)
        # a few ones, followed by a poly decay
        svals = torch.zeros(min_shape, dtype=dtype).to(device)
        svals[:rank] = 1
        svals[rank:] = torch.arange(2, min_shape - rank + 2) ** (-float(decay))
        #
        result = cls._decay_helper(
            svals, shape, rank, decay, symmetric, seed, dtype, device
        )
        return result

    @classmethod
    def exp_decay(
        cls,
        shape=(100, 100),
        rank=10,
        decay=0.5,
        symmetric=True,
        seed=0b1110101001010101011,
        dtype=torch.float64,
        device="cpu",
    ):
        """
        :param decay: Paper values are 0.01 for slow decay, 0.1 for medium, and
          0.5 for fast.
        """
        min_shape = min(shape)
        # a few ones, followed by exp decay
        svals = torch.zeros(min_shape, dtype=dtype).to(device)
        svals[:rank] = 1
        svals[rank:] = 10 ** -(decay * torch.arange(1, min_shape - rank + 1))
        #
        result = cls._decay_helper(
            svals, shape, rank, decay, symmetric, seed, dtype, device
        )
        return result

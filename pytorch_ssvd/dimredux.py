#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import torch
import torch_dct as dct
from .utils import randperm, rademacher, NoFlatError, BadShapeError


# ##############################################################################
# # SSRFT RANDOM PROJECTION
# ##############################################################################
def ssrft(x, out_dims, seed=0b1110101001010101011, dct_norm="ortho"):
    """
    executes R @ F @ PI @ F @ PI'

    where R is an index-picker, F is a DCT, and PI, PI' are independent
    permutations
    """
    if len(x.shape) != 1:
        raise NoFlatError("Only flat tensors supported!")
    x_len = len(x)
    assert out_dims <= x_len, "Projection to larger dimensions not supported!"
    seeds = [seed + i for i in range(5)]
    # first scramble: permute, rademacher, and DCT
    perm1 = randperm(x_len, seed=seeds[0], device="cpu")
    x, rad1 = rademacher(x[perm1], seed=seeds[1], inplace=False)
    del perm1, rad1
    x = dct.dct(x, norm=dct_norm)
    # second scramble: permute, rademacher and DCT
    perm2 = randperm(x_len, seed=seeds[2], device="cpu")
    x, rad2 = rademacher(x[perm2], seeds[3], inplace=False)
    del perm2, rad2
    x = dct.dct(x, norm=dct_norm)
    # extract random indices and return
    out_idxs = randperm(x_len, seed=seeds[4], device="cpu")[:out_dims]
    x = x[out_idxs]
    return x


def ssrft_adjoint(x, out_dims, seed=0b1110101001010101011, dct_norm="ortho"):
    """
    Adjoint operator of SSRFT.
    Note the following:
    * Permutations are orthogonal transforms
    * Rademacher are also orthogonal (plus diagonal and self-inverse)
    * DCT/DFT are also orthogonal transforms
    * The index-picker R is a subset of rows of I.

    Therefore, the adjoint operator takes the following form:
         (R @ F @ PI @ F @ PI')^T
      =  PI'^T  @ F^T @ PI^T @ F^T @ R^T
      =  inv(PI') @ inv(F) @ inv(PI) @ inv(F) @ R^T

    So we can make use of the inverses, except for R^T, which is a column-
    truncated identity, so we embed the entries picked by R into the
    corresponding indices, and leave the rest as zeros.
    """
    if len(x.shape) != 1:
        raise NoFlatError("Only flat tensors supported!")
    x_len = len(x)
    assert (
        out_dims >= x_len
    ), "Backprojection into smaller dimensions not supported!"
    #
    seeds = [seed + i for i in range(5)]
    result = torch.zeros(
        out_dims,
        dtype=x.dtype,
    ).to(x.device)
    # first embed signal into original indices
    out_idxs = randperm(out_dims, seed=seeds[4], device="cpu")[:x_len]
    result[out_idxs] = x
    del x
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


class SSRFT:
    """ """

    def __init__(self, shape, seed=0b1110101001010101011):
        """
        :param shape: Pair with ``(height, width)`` of this linear operator.
        :param scale: Ideally, ``1/l``, where ``l`` is the average diagonal
          value of the covmat ``A.T @ A``, where ``A`` is a FastJLT operator,
          so that ``l2norm(x)`` approximates ``l2norm(Ax)``.
        """
        h, w = shape
        if h > w:
            raise BadShapeError("Height > width not supported!")
        #
        self.shape = shape
        self.seed = seed
        self.scale = NotImplemented

    def __repr__(self):
        """ """
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]}), seed={self.seed}>"
        return s

    def check_input(self, x, adjoint):
        """ """
        try:
            assert len(x.shape) in {
                1,
                2,
            }, "Only vector or matrix input supported"
            #
            if adjoint:
                assert (
                    x.shape[-1] == self.shape[0]
                ), f"Mismatching shapes! {x.shape} <--> {self.shape}"
            else:
                assert (
                    x.shape[0] == self.shape[1]
                ), f"Mismatching shapes! {self.shape} <--> {x.shape}"
        except AssertionError as ae:
            raise BadShapeError from ae

    # operator interfaces
    def __matmul__(self, x):
        """
        Defining forward matrix-vector operation ``self @ x``.
        :param x: A tensor of shape ``(w,)`` or ``(w, k)``.
        """
        self.check_input(x, adjoint=False)
        try:
            return ssrft(x, self.shape[0], seed=self.seed, dct_norm="ortho")
        except NoFlatError:
            result = torch.zeros((self.shape[0], x.shape[1]), dtype=x.dtype).to(
                x.device
            )
            for i in range(x.shape[1]):
                result[:, i] = ssrft(
                    x[:, i], self.shape[0], seed=self.seed, dct_norm="ortho"
                )
            return result

    def __rmatmul__(self, x):
        """ """
        self.check_input(x, adjoint=True)
        try:
            return ssrft_adjoint(
                x, self.shape[1], seed=self.seed, dct_norm="ortho"
            )
        except NoFlatError:
            result = torch.zeros((x.shape[0], self.shape[1]), dtype=x.dtype).to(
                x.device
            )
            for i in range(x.shape[0]):
                result[i, :] = ssrft_adjoint(
                    x[i, :], self.shape[1], seed=self.seed, dct_norm="ortho"
                )
            return result

    def __imatmul__(self, x):
        """
        Defining assignment matmul operator ``@=``.
        """
        raise NotImplementedError("Matmul assignment not supported!")

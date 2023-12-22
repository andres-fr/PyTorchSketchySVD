#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import torch
from .utils import NoFlatError, BadShapeError


# ##############################################################################
# # BASE MATRIX
# ##############################################################################
class BaseMatrix:
    """ """

    def __init__(self, shape):
        """ """
        self.shape = shape

    def __repr__(self):
        """ """
        clsname = self.__class__.__name__
        s = f"<{clsname}({self.shape[0]}x{self.shape[1]})>"
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

    def matmul(self, x):
        """
        Defining forward matrix-vector operation ``self @ x``.
        :param x: Expected a tensor of shape ``(w,)``.
          Note that shapes ``(w, k)`` will be automatically passed as ``k``
          vectors of length ``w``.
        """
        raise NotImplementedError

    def rmatmul(self, x):
        """
        Defining adjoint matrix-vector operation ``x @ self``. Analogous to
        :meth:`matmul`.
        """
        raise NotImplementedError

    # operator interfaces
    def __matmul__(self, x):
        """ """
        self.check_input(x, adjoint=False)
        try:
            return self.matmul(x)
        except NoFlatError:
            result = torch.zeros((self.shape[0], x.shape[1]), dtype=x.dtype).to(
                x.device
            )
            for i in range(x.shape[1]):
                result[:, i] = self.matmul(x[:, i])
            return result

    def __rmatmul__(self, x):
        """ """
        self.check_input(x, adjoint=True)
        try:
            return self.rmatmul(x)
        except NoFlatError:
            result = torch.zeros((x.shape[0], self.shape[1]), dtype=x.dtype).to(
                x.device
            )
            for i in range(x.shape[0]):
                result[i, :] = self.rmatmul(x[i, :])
            return result

    def __imatmul__(self, x):
        """
        Defining assignment matmul operator ``@=``.
        """
        raise NotImplementedError("Matmul assignment not supported!")

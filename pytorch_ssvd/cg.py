#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Pure-PyTorch implementation of the iterative conjugate gradient method to solve
the linear system: find ``x`` in ``A @ x = b``. Reference:

    https://en.wikipedia.org/wiki/Conjugate_gradient_method
"""


import torch


# ##############################################################################
# # CG
# ##############################################################################
class CG:
    """
    Pure-PyTorch implementation of the iterative conjugate gradient method to
    solve the linear system: find ``x`` in ``A @ x = b``. Reference:

        https://en.wikipedia.org/wiki/Conjugate_gradient_method

    Usage example::
      D = 1000
      DEVICE = "cuda"
      A = torch.randn((D, D), dtype=torch.float64).to(DEVICE)
      A = A.T @ A
      x = torch.randn_like(A[0])
      b = A @ x
      #
      x_rec, code = CG.solve_ls(A, b, tol=1e-6, maxiter=10_000, x0=b/D,
                                check_iters=5, verbose=True)
      print(torch.linalg.norm(x).item(), torch.linalg.norm(x - x_rec).item())
      # import matplotlib.pyplot as plt
      # plt.clf(); plt.plot(x.cpu()); plt.plot(x.cpu()-x_rec.cpu()); plt.show()
    """

    MESSAGE = "CG[{}]: {}"

    @classmethod
    def termination_check(
        cls,
        step,
        maxiter,
        residual,
        residual_l2_thresh,
        check_iters=1,
        verbose=False,
    ):
        """
        :param maxiter: Positive scalar.
        :param residual_l2_thresh: Positive scalar.
        :param check_iters: If a positive int is given, check for termination
          every this many iterations. If instead a list of positive integers
          (expected in ascending order) is given, termination will be checked
          at those iterations, and every iteration after the last entry.
        :returns: 0 if no termination, 1 if residual norm is smaller than
          given threshold, and 2 if ``step >= maxiter``.
        """
        # first see if we actually have to check for termination.
        skip_check = True
        if isinstance(check_iters, int):
            if (step % check_iters) == 0:
                skip_check = False
        else:
            if (step >= check_iters[-1]) or (step in check_iters):
                skip_check = False
        # if we skip the check, don't terminate
        if skip_check:
            # print(cls.MESSAGE.format(step, "no check"))
            return 0
        # if we exceed iterations, terminate
        if step >= maxiter:
            if verbose:
                print(cls.MESSAGE.format(step, "maxiter reached"))
            return 2
        # if residual L2 norm is smaller than threshold, terminate
        resnorm = torch.linalg.norm(residual)
        if resnorm <= residual_l2_thresh:
            if verbose:
                print(cls.MESSAGE.format(step, f"converged to tolerance"))
            return 1
        # otherwise don't terminate
        if verbose:
            print(cls.MESSAGE.format(step, f"{resnorm}-->{residual_l2_thresh}"))
        return 0

    @classmethod
    def solve_ls(
        cls, A, b, tol=1e-5, maxiter=10, x0=None, check_iters=1, verbose=False
    ):
        """
        :param A: Any PyTorch-compatible linear (square, symmetric and PSD)
          operator that implements the ``shape=(h, w)`` attribute and the
          matmul (``A @ x``) and adjoint matmul (``x @ A``) operations.
        :param b: Any vector of shape ``(h,)`` such that ``A @ x = b``.
        :param tol: Terminate if ``l2norm(residual) <= (tol * l2norm(b))``.
        :param x0: Initial guess for the x-solution of ``A @ x = b``. A good
          guess may speed up results.
        :param check_iters: See ``termination_check`` docstring.
        :returns: Proposed solution ``x`` of length ``w``.
        """
        height, width = A.shape
        assert height == width, "Matrix must be square, symmetric and PSD!"
        #
        with torch.no_grad():
            # metadata
            dtype, device = b.dtype, b.device
            x_dim = A.shape[1]
            residual_l2_thresh = torch.linalg.norm(b) * tol
            # initial values
            x = (
                torch.zeros(x_dim, dtype=dtype).to(device)
                if x0 is None
                else x0.clone()
            )
            r = b.clone() if x0 is None else (b - A @ x)
            p = r.clone()

            # check if can terminate with initial values
            step = 0
            termination_code = cls.termination_check(
                step, maxiter, r, residual_l2_thresh, check_iters, verbose
            )
            if termination_code > 0:
                return (x, termination_code)
            # CG loop
            while True:
                step += 1
                rr = r @ r
                Ap = A @ p
                alpha = rr / (p @ Ap)
                x += alpha * p
                r -= alpha * Ap
                #
                termination_code = cls.termination_check(
                    step, maxiter, r, residual_l2_thresh, check_iters, verbose
                )
                if termination_code > 0:
                    return (x, termination_code)
                #
                beta = (r @ r) / rr  # note that numerator has been updated
                p *= beta
                p += r

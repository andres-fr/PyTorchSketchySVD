#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import math
import torch

#
from .sketching import SSRFT


# ##############################################################################
# # HYPERPARAMETERS
# ##############################################################################
def a_priori_hyperparams(
    matrix_shape,
    memory_budget,
    complex_data=False,
):
    """
    :param int memory_budget: In number of matrix entries.
    :returns: The pair ``(k, s)``, where the first integer is the optimal
      number of outer sketch measurements, and the second one is the
      corresponding number of core measurements.
    """
    m, n = matrix_shape
    alpha = 0 if complex_data else 1
    mn4a = m + n + 4 * alpha
    budget_root = 16 * (memory_budget - alpha**2)
    #
    outer_dim = math.floor(
        (1 / 8) * (math.sqrt(mn4a**2 + budget_root) - mn4a)
    )
    core_dim = math.floor(math.sqrt(memory_budget - outer_dim * (m + n)))
    #
    return outer_dim, core_dim


# ##############################################################################
# # SSVD
# ##############################################################################
def ssvd(
    matrix,
    matrix_device,
    matrix_dtype,
    outer_dim,
    core_dim,
    seed=0b1110101001010101011,
):
    """ """
    h, w = matrix.shape
    # instantiate random sketching matrices
    left_outer_ssrft = SSRFT((outer_dim, h), seed=seed)
    right_outer_ssrft = SSRFT((outer_dim, w), seed=seed + 1)
    left_core_ssrft = SSRFT((core_dim, h), seed=seed + 2)
    right_core_ssrft = SSRFT((core_dim, w), seed=seed + 3)
    # perform random measurements
    eye = torch.eye(outer_dim, dtype=matrix_dtype).to(matrix_device)
    lo_measurements = (eye @ left_outer_ssrft) @ matrix
    ro_measurements = matrix @ (eye @ right_outer_ssrft).T
    #
    eye = torch.eye(core_dim, dtype=matrix_dtype).to(matrix_device)
    core_measurements = (
        (eye @ left_core_ssrft) @ matrix @ (eye @ right_core_ssrft).T
    )
    del eye
    # QR decompositions of outer measurements
    lo_Q = torch.linalg.qr(lo_measurements.T)[0].T
    del lo_measurements
    ro_Q = torch.linalg.qr(ro_measurements)[0]
    del ro_measurements
    # Solve core matrix to yield initial approximation
    left_core = left_core_ssrft @ ro_Q
    right_core = right_core_ssrft @ lo_Q.T
    core = torch.linalg.lstsq(left_core, core_measurements).solution
    core = torch.linalg.lstsq(right_core, core.T).solution
    # SVD of core matrix for truncated approximation
    core_U, core_S, core_Vt = torch.linalg.svd(core)
    #
    return lo_Q.T, core_U, core_S, core_Vt, ro_Q.T


def truncate_core(core_U, core_S, core_Vt, k):
    """ """
    return core_U[:, :k], core_S[:k], core_Vt[:k, :]

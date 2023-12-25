#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import math
import torch

#
from .sketching import SSRFT
from .utils import normal_noise


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


# ##############################################################################
# # A POSTERIORI ERROR ESTIMATION
# ##############################################################################
def a_posteriori_error_bounds(num_measurements, rel_err):
    """
    Implements probabilistic error bounds from Section 6.4.

    :param int num_measurements: How many Gaussian measurements will be
      performed for the a-posteriori error estimation
    :param float rel_err: A float between 0 and 1, indicating the relative
      error that we want to consider.
    :returns: Probabilities that, given the indicated number
      of measurements, the a-posteriori method yields values outside of
      ``actual_error * (1 +/- rel_err)``. Ideally, we want a sufficient number
      of measurements, such that the returned probabilities for small
      ``rel_err`` are themselves small (this means that the corresponding error
      estimation is tight).
    """
    assert 0 <= rel_err <= 1, "rel_err expected between 0 and 1"
    experr = math.exp(rel_err)
    meas_half = num_measurements / 2
    #
    lo_p = (experr * (1 - rel_err)) ** meas_half
    hi_p = (experr / (1 + rel_err)) ** (-meas_half)
    #
    result = {
        f"P(err<={1 - rel_err}x)": lo_p,
        f"P(err>={1 + rel_err}x)": hi_p,
    }
    return result


def a_posteriori_error(
    mat1,
    mat2,
    num_measurements,
    seed=0b1110101001010101011,
    dtype=torch.float64,
    device="cpu",
):
    """ """
    assert mat1.shape == mat2.shape, "Mismatching shapes!"
    h, w = mat1.shape
    #
    frob1, frob2, diff = [], [], []
    for i in range(num_measurements):
        rand = normal_noise(
            h, mean=0.0, std=1.0, seed=seed + i, dtype=dtype, device=device
        )
        meas1 = rand @ mat1
        meas2 = rand @ mat2
        frob1.append(sum(meas1**2).item())
        frob2.append(sum(meas2**2).item())
        diff.append(sum((meas1 - meas2) ** 2).item())
    #
    frob1_mean = sum(frob1) / num_measurements
    frob2_mean = sum(frob2) / num_measurements
    diff_mean = sum(diff) / num_measurements
    return (frob1_mean, frob2_mean, diff_mean), (frob1, frob2, diff)


def scree(core_S, ori_frob, err_frob):
    """
    Upper and lower bounds for the proportion of energy remaining as a function
    of truncated rank. This can be used to assess the rank of the original
    matrix.
    """
    S_squared = core_S**2
    residuals = S_squared.flip(0).cumsum(0).flip(0) ** 0.5
    lo_scree = (residuals / ori_frob) ** 2
    hi_scree = ((residuals + err_frob) / ori_frob) ** 2
    return lo_scree, hi_scree

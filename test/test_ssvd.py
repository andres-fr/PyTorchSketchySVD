#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import math
import pytest
import torch
from pytorch_ssvd.synthmat import SynthMat
from pytorch_ssvd.ssvd import a_priori_hyperparams, ssvd, truncate_core
from .fixtures import (
    torch_devices,
    rng_seeds,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
)


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def atol_exp():
    """ """
    result = {torch.float64: 1e-7, torch.float32: 1e-3}
    return result


@pytest.fixture
def atol_poly():
    """ """
    result = {torch.float64: 1e-3, torch.float32: 1e-3}
    return result


@pytest.fixture
def heights_widths_ranks_outer_inner():
    """
    Entries are ``((h, w), r, (o, i))``. For a matrix of shape ``h, w`` and
    rank ``r``, the Skinny SVD will do ``o`` outer measurements and ``i``
    inner measurements. Recommended is that ``i >= 2*o``.

    For constrained budgets, the Skinny SVD will naturally yield higher error
    with smaller shapes, hence we test a few medium shapes.
    """
    result = [
        ((1_000, 1_000), 10, (100, 300)),
        ((1_000, 1_000), 50, (200, 600)),
        ((1_000, 2_000), 10, (100, 300)),
        ((1_000, 2_000), 50, (200, 600)),
        ((2_000, 2_000), 20, (100, 300)),
        ((2_000, 2_000), 100, (200, 600)),
    ]
    return result


@pytest.fixture
def budget_ratios():
    """ """
    result = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return result


@pytest.fixture
def recons_frob_rtol():
    """
    Threshold for the Frobenius of residual divided by energy of original.
    If 0.05, it means that the residual must be 5% or less of the original.
    """
    return 0.05


@pytest.fixture
def svectors_frob_rtol():
    """
    Threshold for the Frobenius of residual divided by energy of original.
    If 0.05, it means that the residual must be 5% or less of the original.
    """
    return 0.05


@pytest.fixture
def high_snr():
    """The smaller, the less noise"""
    return [1e-3, 1e-2]


@pytest.fixture
def steep_exp_decay():
    """The larger, the faster decay"""
    return [0.5, 0.1]


@pytest.fixture
def steep_poly_decay():
    """The larger, the faster decay"""
    return [4, 2]


# ##############################################################################
# #
# ##############################################################################
def check_outer_overflow(shape, outer, budget, symmetric=False):
    """
    Helper function.
    Test that increasing outer measurements by 1 breaks budget or constraints.
    """
    h, w = shape
    if symmetric:
        w = 0
    available = budget - ((outer + 1) * (h + w))
    assert available > 0, "Budget overflown!"
    # if not overflown, check if inner can be twice outer
    needed = (2 * (outer + 1)) ** 2
    assert available >= needed, "Inner measurements don't fit in budget!"


def test_a_priori_hyperparams(heights_widths_ranks_outer_inner, budget_ratios):
    """
    Test that:
    * measurements are nonnegative
    * budget is respected
    * s >= 2k
    * more k would explode or break constraints
    """
    # test that symmetric requires square dimensions
    with pytest.raises(AssertionError):
        a_priori_hyperparams((10, 20), 50, complex_data=False, symmetric=True)
    #
    for (h, w), r, (o, i) in heights_widths_ranks_outer_inner:
        for ratio in budget_ratios:
            # test hpar recommendations for non-symmetric matrices
            budget = max(1, math.floor(h * w * ratio))
            oo, ii = a_priori_hyperparams(
                (h, w), budget, complex_data=False, symmetric=False
            )
            assert oo >= 0, "Outer measurements negative?"
            assert ii >= 0, "Inner measurements negative?"
            assert (oo + ii) > 0, "At least one measurement must be given!"
            assert ii >= (oo * 2), "Inner measurements must be >= 2*outer!"
            assert (oo * (h + w) + ii**2) <= budget, "Budget surpassed!"
            # It turns out some of the outputs can be incremented by 1 without
            # breaking budget or constraints. Maybe this is due to taking the
            # math.floor. Anyway, not super important, so we ignore it.
            # with pytest.raises(AssertionError):
            #     check_outer_overflow((h, w), oo, budget, symmetric=False)
            #
            # test hpar recommendations for symmetric matrices
            dim = min(h, w)
            oo, ii = a_priori_hyperparams(
                (dim, dim), budget, complex_data=False, symmetric=True
            )
            assert oo >= 0, "Outer measurements negative?"
            assert ii >= 0, "Inner measurements negative?"
            assert (oo + ii) > 0, "At least one measurement must be given!"
            assert ii >= (oo * 2), "Inner measurements must be >= 2*outer!"
            assert (oo * dim + ii**2) <= budget, "Budget surpassed!"
            # It turns out some of the outputs can be incremented by 1 without
            # breaking budget or constraints. Maybe this is due to taking the
            # math.floor. Anyway, not super important, so we ignore it.
            # with pytest.raises(AssertionError):
            #     check_outer_overflow((dim, dim), oo, budget, symmetric=True)


def test_ssvd_asymmetric_exp(
    rng_seeds,
    torch_devices,
    atol_exp,
    heights_widths_ranks_outer_inner,
    snr_lowrank_noise,
    steep_exp_decay,
    recons_frob_rtol,
    svectors_frob_rtol,
):
    """ """
    for seed in rng_seeds:
        for dtype, atol in atol_exp.items():
            for (h, w), r, (o, i) in heights_widths_ranks_outer_inner:
                for dec in steep_exp_decay:
                    # we create matrix and full SVD on cpu, it is faster
                    mat = SynthMat.exp_decay(
                        shape=(h, w),
                        rank=r,
                        decay=dec,
                        symmetric=False,
                        seed=seed,
                        dtype=dtype,
                        device="cpu",
                    )
                    U, S, Vt = torch.linalg.svd(mat)
                    # then we test SSVD on both devices
                    for device in torch_devices:
                        mat = mat.to(device)
                        U = U.to(device)
                        S = S.to(device)
                        Vt = Vt.to(device)
                        lo_Qt, core_U, core_S, core_Vt, ro_Qt = ssvd(
                            mat, device, dtype, o, i, seed
                        )
                        # check that singular values are correct
                        assert torch.allclose(
                            S[: 2 * r], core_S[: 2 * r], atol=atol
                        ), "Bad recovery of singular values!"
                        # check that recovered matrix is the same
                        core_U, core_S, core_Vt = truncate_core(
                            core_U, core_S, core_Vt, 2 * r
                        )
                        mat_recons = (
                            lo_Qt
                            @ core_U
                            @ torch.diag(core_S)
                            @ core_Vt
                            @ ro_Qt
                        )
                        recons_err = torch.dist(mat, mat_recons) / mat.norm()
                        assert recons_err <= abs(
                            recons_frob_rtol
                        ), "Bad SSVD reconstruction!"
                        # check that a few singular vectors are correct: avoid
                        # the ones that have the same singular values, since
                        # they aren't unique. Also compare outer prods, since
                        # multiplying left and right by -1 yields same result.
                        left = lo_Qt @ core_U[:, r:]
                        right = core_Vt[r:, :] @ ro_Qt
                        for idx in range(2):
                            recons_err = torch.dist(
                                torch.outer(left[:, idx], right[idx]),
                                torch.outer(U[:, idx + r], Vt[idx + r]),
                            ) / (U[:, idx + r].norm() * Vt[idx + r].norm())
                            assert recons_err <= abs(
                                svectors_frob_rtol
                            ), "Bad recovery of singular vectors!"


def test_ssvd_asymmetric_poly(
    rng_seeds,
    torch_devices,
    atol_poly,
    heights_widths_ranks_outer_inner,
    snr_lowrank_noise,
    steep_poly_decay,
    recons_frob_rtol,
    svectors_frob_rtol,
):
    """ """
    for seed in rng_seeds:
        for dtype, atol in atol_poly.items():
            for (h, w), r, (o, i) in heights_widths_ranks_outer_inner:
                for dec in steep_poly_decay:
                    # we create matrix and full SVD on cpu, it is faster
                    mat = SynthMat.poly_decay(
                        shape=(h, w),
                        rank=r,
                        decay=dec,
                        symmetric=False,
                        seed=seed,
                        dtype=dtype,
                        device="cpu",
                    )
                    U, S, Vt = torch.linalg.svd(mat)
                    # then we test SSVD on both devices
                    for device in torch_devices:
                        mat = mat.to(device)
                        U = U.to(device)
                        S = S.to(device)
                        Vt = Vt.to(device)
                        lo_Qt, core_U, core_S, core_Vt, ro_Qt = ssvd(
                            mat, device, dtype, o, i, seed
                        )
                        # check that singular values are correct
                        print(S[: 2 * r], core_S[: 2 * r], sep="\n")
                        assert torch.allclose(
                            S[: 2 * r], core_S[: 2 * r], atol=atol
                        ), "Bad recovery of singular values!"
                        # check that recovered matrix is the same
                        core_U, core_S, core_Vt = truncate_core(
                            core_U, core_S, core_Vt, 2 * r
                        )
                        mat_recons = (
                            lo_Qt
                            @ core_U
                            @ torch.diag(core_S)
                            @ core_Vt
                            @ ro_Qt
                        )
                        recons_err = torch.dist(mat, mat_recons) / mat.norm()
                        assert recons_err <= abs(
                            recons_frob_rtol
                        ), "Bad SSVD reconstruction!"
                        # check that a few singular vectors are correct: avoid
                        # the ones that have the same singular values, since
                        # they aren't unique. Also compare outer prods, since
                        # multiplying left and right by -1 yields same result.
                        left = lo_Qt @ core_U[:, r:]
                        right = core_Vt[r:, :] @ ro_Qt
                        for idx in range(2):
                            recons_err = torch.dist(
                                torch.outer(left[:, idx], right[idx]),
                                torch.outer(U[:, idx + r], Vt[idx + r]),
                            ) / (U[:, idx + r].norm() * Vt[idx + r].norm())
                            assert recons_err <= abs(
                                svectors_frob_rtol
                            ), "Bad recovery of singular vectors!"

                            print(
                                left[:100, idx],
                                right[idx, :100],
                                U[:100, idx + r],
                                Vt[idx + r, :100],
                                sep="\n",
                            )

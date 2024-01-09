#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch
from pytorch_ssvd.synthmat import SynthMat
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
def f64_rtol():
    """ """
    result = {torch.float64: 1e-10}
    return result


@pytest.fixture
def f32_rtol():
    """ """
    result = {torch.float32: 1e-3}
    return result


@pytest.fixture
def dims_ranks_square():
    """ """
    result = [
        (1, 1),
        (10, 1),
        (100, 10),
        (1_000, 10),
        (1_000, 50),
    ]
    return result


@pytest.fixture
def heights_widths_ranks_fat():
    """ """
    result = [
        (1, 10, 1),
        (10, 100, 1),
        (100, 1_000, 10),
        (1_000, 10_000, 100),
    ]
    return result


@pytest.fixture
def decay_ew_atol():
    """ """
    result = 1e-5
    return result


##############################################################################
#
##############################################################################
def test_symmetric(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    dims_ranks_square,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
    decay_ew_atol,
):
    """
    Create square, symmetric synthetic matrices and test that:
    * there are no NaNs
    * they are indeed symmetric
    * their diagonals/spectra are correct
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for dim, r in dims_ranks_square:
                    # lowrank+noise
                    for snr in snr_lowrank_noise:
                        mat = SynthMat.lowrank_noise(
                            shape=(dim, dim),
                            rank=r,
                            snr=snr,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        assert not mat.isnan().any(), f"{ssrft, device, dtype}"
                        assert torch.allclose(
                            mat, mat.T, rtol=rtol
                        ), "Matrix not symmetric?"
                        assert all(
                            mat[range(r), range(r)] >= 1
                        ), "mat[:rank] is not >=1 for given rank?"
                    # exp decay
                    for dec in exp_decay:
                        mat = SynthMat.exp_decay(
                            shape=(dim, dim),
                            rank=r,
                            decay=dec,
                            symmetric=True,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        ew = torch.linalg.eigvalsh(mat).flip(0)  # desc order
                        assert not mat.isnan().any(), f"{ssrft, device, dtype}"
                        assert torch.allclose(
                            mat, mat.T, rtol=rtol
                        ), "Matrix not symmetric?"
                        assert torch.allclose(
                            ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                        ), "ew[:rank] should be == 1"
                        ew_dec = 10.0 ** -(
                            dec * torch.arange(1, dim - r + 1, dtype=dtype)
                        ).to(device)
                        assert torch.allclose(
                            ew[r:],
                            ew_dec,
                            rtol=rtol,
                            atol=decay_ew_atol,  # added atol due to eigvalsh
                        ), "Eigenval decay mismatch!"
                    # poly decay
                    for dec in poly_decay:
                        mat = SynthMat.poly_decay(
                            shape=(dim, dim),
                            rank=r,
                            decay=dec,
                            symmetric=True,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        ew = torch.linalg.eigvalsh(mat).flip(0)  # desc order
                        assert not mat.isnan().any(), f"{ssrft, device, dtype}"
                        assert torch.allclose(
                            mat, mat.T, rtol=rtol
                        ), "Matrix not symmetric?"
                        assert torch.allclose(
                            ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                        ), "ew[:rank] should be == 1"
                        ew_dec = (
                            torch.arange(2, dim - r + 2, dtype=dtype)
                            ** (-float(dec))
                        ).to(device)
                        assert torch.allclose(
                            ew[r:],
                            ew_dec,
                            rtol=rtol,
                            atol=decay_ew_atol,  # added atol due to eigvalsh
                        ), f"Eigenval decay mismatch!  \n\n{ew[r:]}\n\n{ew_dec}"


def test_asymmetric_nonsquare(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    heights_widths_ranks_fat,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
    decay_ew_atol,
):
    """
    Create square, symmetric synthetic matrices and test that:
    * there are no NaNs
    * they are indeed symmetric
    * their diagonals/spectra are correct

    Since lowrank+noise must be symmetric, it is omitted here.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for h, w, r in heights_widths_ranks_fat:
                    min_dim = min(h, w)
                    # exp decay
                    for dec in exp_decay:
                        mat = SynthMat.exp_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        ew = torch.linalg.svdvals(mat)  # desc order
                        assert not mat.isnan().any(), f"{ssrft, device, dtype}"
                        assert torch.allclose(
                            ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                        ), f"ew[:rank] should be == 1, {ew}"
                        ew_dec = 10.0 ** -(
                            dec * torch.arange(1, min_dim - r + 1, dtype=dtype)
                        ).to(device)
                        assert torch.allclose(
                            ew[r:],
                            ew_dec,
                            rtol=rtol,
                            atol=decay_ew_atol,  # added atol due to eigvalsh
                        ), "Eigenval decay mismatch!"
                    # poly decay
                    for dec in poly_decay:
                        mat = SynthMat.poly_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        ew = torch.linalg.svdvals(mat)  # desc order
                        assert not mat.isnan().any(), f"{ssrft, device, dtype}"
                        assert torch.allclose(
                            ew[:r], torch.ones_like(ew[:r]), rtol=rtol
                        ), "ew[:rank] should be == 1"
                        ew_dec = (
                            torch.arange(2, min_dim - r + 2, dtype=dtype)
                            ** (-float(dec))
                        ).to(device)
                        assert torch.allclose(
                            ew[r:],
                            ew_dec,
                            rtol=rtol,
                            atol=decay_ew_atol,  # added atol due to eigvalsh
                        ), f"Eigenval decay mismatch!  \n\n{ew[r:]}\n\n{ew_dec}"


def test_seed_consistency(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    heights_widths_ranks_fat,
    snr_lowrank_noise,
    exp_decay,
    poly_decay,
    decay_ew_atol,
):
    """
    Test that same seed and shape lead to same operator with same results,
    and different otherwise.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for h, w, r in heights_widths_ranks_fat:
                    # lowrank+noise
                    for snr in snr_lowrank_noise:
                        mat1 = SynthMat.lowrank_noise(
                            shape=(h, h),  # L+N must be square
                            rank=r,
                            snr=snr,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        mat2 = SynthMat.lowrank_noise(
                            shape=(h, h),  # L+N must be square
                            rank=r,
                            snr=snr,
                            seed=seed + 1,
                            dtype=dtype,
                            device=device,
                        )
                        mat3 = SynthMat.lowrank_noise(
                            shape=(h, h),  # L+N must be square
                            rank=r,
                            snr=snr,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        assert torch.allclose(
                            mat1, mat3, rtol
                        ), "Same seed different matrix?"
                        assert (
                            mat1 != mat2
                        ).any(), f"Different seed same matrix?, {mat1, mat3}"
                    # exp decay
                    for dec in exp_decay:
                        mat1 = SynthMat.exp_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        mat2 = SynthMat.exp_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed + 1,
                            dtype=dtype,
                            device=device,
                        )
                        mat3 = SynthMat.exp_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        assert torch.allclose(
                            mat1, mat3, rtol
                        ), "Same seed different matrix?"
                        assert (
                            mat1 != mat2
                        ).any(), f"Different seed same matrix?, {mat1, mat3}"
                    # poly decay
                    for dec in poly_decay:
                        mat1 = SynthMat.poly_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        mat2 = SynthMat.poly_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed + 1,
                            dtype=dtype,
                            device=device,
                        )
                        mat3 = SynthMat.poly_decay(
                            shape=(h, w),
                            rank=r,
                            decay=dec,
                            symmetric=False,
                            seed=seed,
                            dtype=dtype,
                            device=device,
                        )
                        assert torch.allclose(
                            mat1, mat3, rtol
                        ), "Same seed different matrix?"
                        assert (
                            mat1 != mat2
                        ).any(), f"Different seed same matrix?, {mat1, mat3}"

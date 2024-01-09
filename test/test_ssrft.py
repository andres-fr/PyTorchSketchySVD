#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch
from pytorch_ssvd.sketching import SSRFT
from pytorch_ssvd.utils import BadShapeError

# f32 tolerance goes to zero for larger shapes (which is where SSVD makes
# sense), so we only test f64. Use f32 at own risk
from .fixtures import (
    torch_devices,
    rng_seeds,
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
def square_shapes():
    """ """
    result = [
        (1, 1),
        (10, 10),
        (100, 100),
        (10_000, 10_000),
        (1_000_000, 1_000_000),
    ]
    return result


@pytest.fixture
def fat_shapes():
    """ """
    result = [
        (1, 10),
        (10, 100),
        (100, 1_000),
        (10_000, 100_000),
        (1_000_000, 10_000_000),
    ]
    return result


# ##############################################################################
# # TESTS
# ##############################################################################
def test_no_nans(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """ """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, rtol in f64_rtol.items():
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    assert not x.isnan().any(), f"{ssrft, device, dtype}"
                    assert not y.isnan().any(), f"{ssrft, device, dtype}"
                    assert not xx.isnan().any(), f"{ssrft, device, dtype}"


def test_invertible(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """
    Test that, when input and output dimensionality are the same, the SSRFT
    operator is orthogonal, i.e. we can recover the input exactly via an
    adjoint operation.
    Also test that it works for mat-vec and mat-mat formats.
    """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, rtol in f64_rtol.items():
                    # matvec
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    print("\n" * 3)
                    print(x)
                    print(xx)
                    assert torch.allclose(
                        x, xx, rtol=rtol
                    ), f"MATVEC: {ssrft, device, dtype}"
                    # matmat
                    x = torch.randn((w, 2), dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = (y.T @ ssrft).T
                    #
                    assert torch.allclose(
                        x, xx, rtol=rtol
                    ), f"MATMAT: {ssrft, device, dtype}"
                    # matmat-shape tests
                    assert len(y.shape) == 2
                    assert len(xx.shape) == 2
                    assert y.shape[-1] == 2
                    assert xx.shape[-1] == 2


def test_seed_consistency(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """
    Test that same seed and shape lead to same operator with same results,
    and different otherwise.
    """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            ssrft_same = SSRFT((h, w), seed=seed)
            ssrft_diff = SSRFT((h, w), seed=seed + 1)
            for device in torch_devices:
                for dtype, rtol in f64_rtol.items():
                    # matvec
                    x = torch.randn(w, dtype=dtype).to(device)
                    assert ((ssrft @ x) == (ssrft_same @ x)).all()
                    # here, dim=1 may indeed result in same output, since
                    # there are no permutations or index-pickings, so 50/50.
                    # therefore we ignore that case.
                    if x.numel() > 1:
                        assert ((ssrft @ x) != (ssrft_diff @ x)).any()


def test_unsupported_tall_ssrft(rng_seeds, fat_shapes):
    """ """
    for seed in rng_seeds:
        for h, w in fat_shapes:
            with pytest.raises(BadShapeError):
                # If this line throws a BadShapeError, the test passes
                ssrft = SSRFT((w, h), seed=seed)


def test_input_shape_mismatch(rng_seeds, fat_shapes, torch_devices, f64_rtol):
    """ """
    for seed in rng_seeds:
        for h, w in fat_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, rtol in f64_rtol.items():
                    # forward matmul
                    x = torch.empty(w + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        ssrft @ x
                    # adjoint matmul
                    x = torch.empty(h + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        x @ ssrft

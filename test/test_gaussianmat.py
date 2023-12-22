#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch
from pytorch_ssvd.dimredux import GaussianIIDMatrix
from pytorch_ssvd.utils import BadShapeError
from .fixtures import (
    torch_devices,
    torch_dtypes_rtols,
    rng_seeds,
)


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def square_shapes():
    """ """
    result = [
        (1, 1),
        (10, 10),
        (100, 100),
        (1_000, 1_000),
    ]
    return result


@pytest.fixture
def fat_shapes():
    """ """
    result = [
        (1, 10),
        (10, 100),
        (100, 1_000),
        (1_000, 10_000),
    ]
    return result


# ##############################################################################
# # POSITIVE TESTS
# ##############################################################################
def test_no_nans(torch_devices, torch_dtypes_rtols, rng_seeds, square_shapes):
    """ """
    for seed in rng_seeds:
        for h, w in square_shapes:
            for device in torch_devices:
                for dtype, rtol in torch_dtypes_rtols.items():
                    G = GaussianIIDMatrix(
                        (h, w), seed, dtype, device, mean=0, std=1
                    )
                    # matvec
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = G @ x
                    xx = y @ G
                    assert not x.isnan().any(), f"{ssrft, device, dtype}"
                    assert not y.isnan().any(), f"{ssrft, device, dtype}"
                    assert not xx.isnan().any(), f"{ssrft, device, dtype}"
                    # matmat
                    x = torch.randn((w, 2), dtype=dtype).to(device)
                    y = G @ x
                    xx = (y.T @ G).T
                    assert not x.isnan().any(), f"{ssrft, device, dtype}"
                    assert not y.isnan().any(), f"{ssrft, device, dtype}"
                    assert not xx.isnan().any(), f"{ssrft, device, dtype}"
                    # matmat-shape tests
                    assert len(y.shape) == 2
                    assert len(xx.shape) == 2
                    assert y.shape[-1] == 2
                    assert xx.shape[-1] == 2


def test_seed_consistency(
    torch_devices, torch_dtypes_rtols, rng_seeds, square_shapes
):
    """
    Test that same seed and shape lead to same operator with same results,
    and different otherwise.
    """
    for seed in rng_seeds:
        for h, w in square_shapes:
            for device in torch_devices:
                for dtype, rtol in torch_dtypes_rtols.items():
                    G = GaussianIIDMatrix(
                        (h, w), seed, dtype, device, mean=0, std=1
                    )
                    G_same = GaussianIIDMatrix(
                        (h, w), seed, dtype, device, mean=0, std=1
                    )
                    G_diff = GaussianIIDMatrix(
                        (h, w), seed + 1, dtype, device, mean=0, std=1
                    )
                    #
                    assert (G._weights == G_same._weights).all()
                    assert (G._weights != G_diff._weights).any()
                    #
                    x = torch.randn(w, dtype=dtype).to(device)
                    assert ((G @ x) == (G_same @ x)).all()
                    assert ((G @ x) != (G_diff @ x)).any()


# ##############################################################################
# # NEGATIVE TESTS
# ##############################################################################
def test_input_shape_mismatch(
    rng_seeds, fat_shapes, torch_devices, torch_dtypes_rtols
):
    """ """
    for seed in rng_seeds:
        for h, w in fat_shapes:
            for device in torch_devices:
                for dtype, rtol in torch_dtypes_rtols.items():
                    G = GaussianIIDMatrix(
                        (h, w), seed, dtype, device, mean=0, std=1
                    )
                    # forward matmul
                    x = torch.empty(w + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        G @ x
                    # adjoint matmul
                    x = torch.empty(h + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        x @ G

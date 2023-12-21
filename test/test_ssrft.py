#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch
from pytorch_ssvd.dimredux import SSRFT
from pytorch_ssvd.utils import BadShapeError


# ##############################################################################
# # SET UP WORKING FIXTURES
# ##############################################################################
@pytest.fixture
def torch_devices():
    """ """
    result = ["cpu"]
    if torch.cuda.is_available():
        result.append("cuda")
    return result


@pytest.fixture
def torch_dtypes_rtols():
    """ """
    # f32 tolerance goes to zero for larger shapes (which is where SSVD makes
    # sense), so we only test f64. Use f32 at own risk
    result = {torch.float64: 1e-10}
    return result


@pytest.fixture
def rng_seeds():
    """ """
    result = [0, 1, -1, 12345, 0b1110101001010101011]
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
# # POSITIVE TESTS
# ##############################################################################
def test_no_nans(torch_devices, torch_dtypes_rtols, rng_seeds, square_shapes):
    """ """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, rtol in torch_dtypes_rtols.items():
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    assert not x.isnan().any(), f"{ssrft, device, dtype}"
                    assert not y.isnan().any(), f"{ssrft, device, dtype}"
                    assert not xx.isnan().any(), f"{ssrft, device, dtype}"


def test_invertible(
    torch_devices, torch_dtypes_rtols, rng_seeds, square_shapes
):
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
                for dtype, rtol in torch_dtypes_rtols.items():
                    # matvec
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    assert torch.allclose(
                        x, xx, rtol=rtol
                    ), f"{ssrft, device, dtype}"
                    # matmat
                    x = torch.randn((w, 2), dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = (y.T @ ssrft).T
                    #
                    assert torch.allclose(
                        x, xx, rtol=rtol
                    ), f"{ssrft, device, dtype}"


# ##############################################################################
# # NEGATIVE TESTS
# ##############################################################################
def test_unsupported_tall_ssrft(rng_seeds, fat_shapes):
    """ """
    for seed in rng_seeds:
        for h, w in fat_shapes:
            with pytest.raises(BadShapeError):
                # If this line throws a BadShapeError, the test passes
                ssrft = SSRFT((w, h), seed=seed)


def test_input_shape_mismatch(
    rng_seeds, fat_shapes, torch_devices, torch_dtypes_rtols
):
    """ """
    for seed in rng_seeds:
        for h, w in fat_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, rtol in torch_dtypes_rtols.items():
                    # forward matmul
                    x = torch.empty(w + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        ssrft @ x
                    # adjoint matmul
                    x = torch.empty(h + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        x @ ssrft

#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch
from pytorch_ssvd.synthmat import SynthMat
from .fixtures import (
    torch_devices,
    f64_rtol,
    f32_rtol,
    rng_seeds,
)


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def heights_widths_ranks_square():
    """ """
    result = [
        (1, 1, 1),
        (10, 10, 1),
        (100, 100, 10),
        (1_000, 1_000, 10),
        (1_000, 1_000, 50),
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


# ##############################################################################
# #
# ##############################################################################
def test_nans_and_symmetry(
    rng_seeds,
    torch_devices,
    f64_rtol,
    f32_rtol,
    heights_widths_ranks_square,
    heights_widths_ranks_fat,
):
    """
    Create sym and asym, and test for that
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, rtol in {**f64_rtol, **f32_rtol}.items():
                # symmetric tests
                for h, w, r in heights_widths_ranks_square:
                    mm = SynthMat.lowrank_noise(
                        (h, w), rank=10, snr=1e-1, device="cuda"
                    )
                    em = SynthMat.exp_decay(
                        (OUT, IN), rank=10, decay=0.1, symmetric=True
                    )
                    mat = SynthMat.exp_decay(
                        (OUT, IN),
                        rank=RANK,
                        decay=0.1,
                        symmetric=True,
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                # asym tests
                for h, w, r in heights_widths_ranks_square:
                    pass
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    assert not x.isnan().any(), f"{ssrft, device, dtype}"
                    assert not y.isnan().any(), f"{ssrft, device, dtype}"
                    assert not xx.isnan().any(), f"{ssrft, device, dtype}"


def test_svd():
    """
    Take actual SVD and check that it is the exact same as the wanted spectrum
    """
    pass


def test_seed_consistency(
    torch_devices, torch_dtypes_rtols, rng_seeds, square_shapes
):
    """
    Test that same seed and shape lead to same operator with same results,
    and different otherwise.
    """
    pass
    # for seed in rng_seeds:
    #     for h, w in square_shapes:
    #         ssrft = SSRFT((h, w), seed=seed)
    #         ssrft_same = SSRFT((h, w), seed=seed)
    #         ssrft_diff = SSRFT((h, w), seed=seed + 1)
    #         for device in torch_devices:
    #             for dtype, rtol in torch_dtypes_rtols.items():
    #                 # matvec
    #                 x = torch.randn(w, dtype=dtype).to(device)
    #                 assert ((ssrft @ x) == (ssrft_same @ x)).all()
    #                 # here, dim=1 may indeed result in same output, since
    #                 # there are no permutations or index-pickings, so 50/50.
    #                 # therefore we ignore that case.
    #                 if x.numel() > 1:
    #                     assert ((ssrft @ x) != (ssrft_diff @ x)).any()

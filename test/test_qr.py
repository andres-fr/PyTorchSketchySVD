#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch
from pytorch_ssvd.utils import normal_noise
from .fixtures import (
    torch_devices,
    rng_seeds,
)


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def heights_widths():
    """ """
    result = [
        (1, 1),
        (10, 10),
        (100, 100),
        (1_000, 1_000),
    ]
    result += [
        (10, 1),
        (100, 10),
        (1_000, 100),
        (10_000, 1_000),
    ]
    return result


@pytest.fixture
def mean_frob_atol():
    """ """
    result = 1e-5
    return result


# ##############################################################################
# #
# ##############################################################################
def test_orth_q(rng_seeds, torch_devices, heights_widths, mean_frob_atol):
    """ """
    for h, w in heights_widths:
        assert h >= w, "This test doesn't need/admit fat matrices!"
    #
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype in (torch.float64, torch.float32):
                for h, w in heights_widths:
                    mat = normal_noise(
                        (h, w),
                        mean=0.0,
                        std=1.0,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    Q = torch.linalg.qr(mat)[0]
                    I_residual = Q.T @ Q
                    I_residual[range(w), range(w)] -= 1
                    worst = I_residual.abs().max().item()
                    assert worst <= abs(
                        mean_frob_atol
                    ), "Q matrix not orthogonal?"

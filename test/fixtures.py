#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import pytest
import torch


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

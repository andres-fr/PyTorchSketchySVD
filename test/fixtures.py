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
def rng_seeds():
    """ """
    result = [0, 1, -1, 12345, 0b1110101001010101011]
    return result

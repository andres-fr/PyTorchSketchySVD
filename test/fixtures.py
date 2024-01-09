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
def rng_seeds():
    """ """
    result = [0, 1, -1, 12345, 0b1110101001010101011]
    return result


@pytest.fixture
def snr_lowrank_noise():
    """ """
    result = [1e-3, 1e-2, 1e-1, 1]
    return result


@pytest.fixture
def exp_decay():
    """ """
    result = [0.5, 0.1, 0.01]
    return result


@pytest.fixture
def poly_decay():
    """ """
    result = [2, 1, 0.5]
    return result

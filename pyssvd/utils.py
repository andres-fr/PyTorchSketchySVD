#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import torch


# ##############################################################################
# # ERRORS
# ##############################################################################
class NoFlatError(Exception):
    """ """

    pass


# ##############################################################################
# # REPRODUCIBLE NOISE
# ##############################################################################
def uniform_noise(shape, seed=None, dtype=torch.float64, device="cpu"):
    """
    Reproducible ``torch.rand`` (uniform noise between 0 and 1).
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    noise = torch.rand(shape, generator=rng, dtype=dtype, device=device)
    return noise


def normal_noise(
    shape, mean=0.0, std=1.0, seed=None, dtype=torch.float64, device="cpu"
):
    """
    Reproducible ``torch.normal_`` (Gaussian noise with given mean and
    standard deviation).
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    #
    noise = torch.zeros(shape, dtype=dtype, device=device)
    noise.normal_(mean=mean, std=std, generator=rng)
    return noise


def randperm(max_excluded, seed=None, device="cpu", inverse=False):
    """
    Reproducible random permutation between 0 (included) and
    ``max_excluded`` (excluded, but max-1 is included).
    :param bool inverse: If False, a random permutation ``P`` is provided. If
      true, an inverse permutation ``Q`` is provided, such that
      ``arr == arr[P][Q] == arr[Q][P]``.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    #
    perm = torch.randperm(max_excluded, generator=rng, device=device)
    if inverse:
        # we take the O(N) approach since we anticipate large N
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        perm = inv
    return perm


def rademacher(x, seed=None, inplace=True):
    """
    Reproducible sign-flipping via Rademacher noise.
    .. note::
      This function makes use of :func:`uniform_noise` to sample the Rademacher
      noise. If ``x`` itself has been generated using ``uniform_noise``, make
      sure to use a different seed to mitigate correlations.
    """
    mask = (
        uniform_noise(x.shape, seed=seed, dtype=torch.float32, device=x.device)
        > 0.5
    ) * 2 - 1
    if inplace:
        x *= mask
        return x, mask
    else:
        return x * mask, mask

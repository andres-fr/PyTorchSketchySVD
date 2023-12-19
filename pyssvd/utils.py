#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import torch


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


def randperm(max_excluded, seed=None, device="cpu"):
    """
    Reproducible random permutation between 0 (included) and
    ``max_excluded`` (excluded, but max-1 is included).
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    #
    perm = torch.randperm(max_excluded, generator=rng, device=device)
    return perm


def randperm_subset(height, width, seed=None, device="cpu"):
    """
    In randperm, a fully reproducible random permutation is generated. This
    function retrieves only the first ``height`` entries, so the result
    contains the indices of a reproducible random-sparse projection.
    :returns: Int64 tensor of shape ``(num_idxs,)`` containing ``height``
      random, non-repeated numbers between 0 and ``width - 1`` (both included).
    """
    idxs = randperm(width, seed, device)[:height]
    return idxs


def rademacher(x, seed=None, inplace=True):
    """
    Reproducible sign-flipping via Rademacher noise.
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


def sparseproj(
    x, out_dims=None, seed=None, cached_idxs=None, cached_noise=None
):
    """
    Reproducible sparse projection consisting in random index selection
    of the input ``x`` followed by multiplication by noise.
    :param out_dims: If an integer is given, this many random indices
      will be randomly chosen. Alternatively, ``cached_idxs`` must be
      explicitly given.
    :param seed: Random seed if indices are randomly chosen.
    :param cached_idxs: Optional integer, flat tensor indicating which
      indices from ``x`` will be gathered.
    :param cached_noise: If given, this will be used as the multiplicative
      noise. Otherwise, zero-mean, scaled Gaussian noise of the same dtype
      as ``x`` will be generated.
    :returns: ``(projection, idxs, noise)``.
    """
    assert len(x.shape) == 1, "Only flat tensors supported!"
    len_x = len(x)
    #
    if cached_idxs is not None:
        idxs = cached_idxs
    else:
        idxs = randperm_subset(out_dims, len_x, seed, x.device)
    out_dims = idxs.numel()
    #
    if cached_noise is not None:
        noise = cached_noise
    else:
        noise = cls.get_noise(out_dims, len_x, 1, seed, x.dtype, x.device)
        assert noise.all(), "Noise contains zeros! Not allowed."
    # the actual projection: multiply random inputs by noise and return
    result = x[idxs] * noise
    return result, idxs, noise

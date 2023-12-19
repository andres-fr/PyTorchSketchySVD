#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Pure-PyTorch implementation of the matrix-free Fast Johnson-Lindenstrauss
Transform to efficiently perform random projections ``y = JLT @ x``. Reference:

    https://johnthickstun.com/docs/fast_jlt.pdf

The module provides a lower-level ``FastJLT`` static class providing flexible
functionality to perform FJLT-related actions, and a higher-level, more rigid
implementation via ``JLTSensingMatrix`` that can be used as a linear operator,
saving some computation and reducing code overhead.
"""


import torch
import hadamard_transform as fht


# ##############################################################################
# # FJLT
# ##############################################################################
class FastJLT:
    """
    To perform a random projection from N dims to K, normally we need to
    create and store a KxN matrix. For large N, K, this can be infeasible
    both in terms of storage and computations.

    The Fast Johnson-Lindenstrauss Transform allows to efficiently perform
    a Kx(2^k) random projection without explicitly storing the matrix or
    having to do a full matrix multiplication. For that, it leverages the
    fast Welsh-Hadamard transform, which is symmetric and self-inverse.
    Note that it is numerically unstable and float64 is highly encouraged.

    This static class implements the FJLT as a linear operator using PyTorch
    via the ``fjlt`` and ``conj_fjlt`` methods, and is fast for N well into
    the millions, even on CPU. More info:

    https://johnthickstun.com/docs/fast_jlt.pdf
    """

    SEED = 0b1110101001010101011

    @classmethod
    def uniform_noise(cls, shape, seed=None, dtype=torch.float64,
                      device="cpu"):
        """
        Reproducible ``torch.rand`` (uniform noise between 0 and 1).
        """
        if seed is None:
            seed = cls.SEED
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        noise = torch.rand(shape, generator=rng, dtype=dtype, device=device)
        return noise

    @classmethod
    def normal_noise(cls, shape, mean=0.0, std=1.0, seed=None,
                     dtype=torch.float64, device="cpu"):
        """
        Reproducible ``torch.normal_`` (Gaussian noise with given mean and
        standard deviation).
        """
        if seed is None:
            seed = cls.SEED
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        #
        noise = torch.zeros(shape, dtype=dtype, device=device)
        noise.normal_(mean=mean, std=std, generator=rng)
        return noise

    @classmethod
    def randperm(cls, max_excluded, seed=None, device="cpu"):
        """
        Reproducible random permutation between 0 (included) and
        ``max_excluded`` (excluded, but max-1 is included).
        """
        if seed is None:
            seed = cls.SEED
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        #
        perm = torch.randperm(max_excluded, generator=rng, device=device)
        return perm

    @classmethod
    def rademacher(cls, x, seed=None, inplace=True):
        """
        Reproducible sign-flipping via Rademacher noise.
        """
        if seed is None:
            seed = cls.SEED
        mask = (cls.uniform_noise(x.shape, seed=seed, dtype=torch.float32,
                                  device=x.device) > 0.5) * 2 - 1
        if inplace:
            x *= mask
            return x, mask
        else:
            return x * mask, mask

    @staticmethod
    def hadamard(x, inplace=True):
        """
        :param x: A flat tensor of length ``2^k`` (recommended float64, since
          the transform can be numerically unstable).
        Fast Hadamard transform of ``x`` (symmetric and self-inverse).
        """
        if inplace:
            fht.hadamard_transform_(x)
            return x
        else:
            return fht.hadamard_transform(x)

    @classmethod
    def explicit_hadamard(cls, n, idxs, dtype=torch.float64, device="cpu"):
        """
        Explicit representation of ``(n, n)`` Hadamard for given column
        indices, i.e. returns a matrix of shape ``(n, len(idxs))``.
        """
        result = torch.empty((n, len(idxs)), dtype=dtype).to(device)
        for i, idx in enumerate(idxs):
            onehot = torch.zeros_like(result[:, 0])
            onehot[idx] = 1
            result[:, i] = cls.hadamard(onehot, inplace=False)
        return result

    @classmethod
    def get_idxs(cls, height, width, seed=None, device="cpu"):
        """
        Random projection indices as a function of shape and seed.
        :returns: Int64 tensor of shape ``(num_idxs,)`` containing random,
          non-repeated numbers between 0 and ``max_excluded - 1``
          (both included).
        """
        idxs = cls.randperm(width, seed, device)[:height]
        return idxs

    @classmethod
    def get_noise(cls, height, width, std_scale=1.0, seed=None,
                  dtype=torch.float64, device="cpu"):
        """
        Random projection noise as a function of shape and seed.
        :param std_scale: The standard deviation of the output is defined as
          ``scale/p``, where ``p`` is the sampling ratio
          ``output_dims/input_dims``.
        """
        scale = std_scale / (height / width)
        noise = cls.normal_noise(
            height, std=scale, seed=seed, dtype=dtype, device=device)
        return noise

    @classmethod
    def sparseproj(cls, x, out_dims=None, seed=None,
                   cached_idxs=None, cached_noise=None):
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
            idxs = cls.get_idxs(out_dims, len_x, seed, x.device)
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

    @staticmethod
    def pow2roundup(x):
        """
        :param x: A scalar
        :returns: The nearest power of 2 above ``x`` as an integer.
        """
        result = int(2 ** torch.ceil(torch.log2(torch.tensor(x))))
        return result

    @classmethod
    def fjlt(cls, t, out_dims=None, seed=None,
             cached_idxs=None, cached_noise=None):
        """
        Implementation of the Fast JLT as ``y = proj@hadamard@rademacher @ t``.
        :param t: Flat tensor of length ``2^k``.
        :param out_dims: See ``sparseproj`` docstring.
        :param cached_idxs: See ``sparseproj`` docstring.
        :param cached_noise: See ``sparseproj`` docstring.
        :returns: The tuple ``(y, mask, idxs, noise)``, where ``y`` is a flat
          tensor of same dtype and device as the input ``t``, with ``out_dims``
          entries corresponding to the Fast JLT of the input. The tensors
          ``mask, idxs, noise`` correspond to the parameters returned by the
          rademacher noise and sparse projection.
        """
        assert len(t.shape) == 1, "Only flat tensors supported!"
        in_len = len(t)
        assert in_len == cls.pow2roundup(in_len), \
            "Only inputs with pow2 elements supported!"
        #
        t, mask = cls.rademacher(t, seed=seed, inplace=False)
        cls.hadamard(t, inplace=True)
        #
        result, idxs, noise = cls.sparseproj(
            t, out_dims, seed, cached_idxs, cached_noise)
        return result, mask, idxs, noise

    @classmethod
    def transp_fjlt(cls, y, out_dims, seed=None,
                    cached_idxs=None, cached_noise=None, out=None):
        """
        Implementation of the conjugate Fast JLT as
        ``x = (proj@hadamard@rademacher).T @ y``.

        :param y: Flat tensor of arbitrary length.
        :param out_dims: A power of 2 determining the output dimensionality
          (i.e. the row width of the Hadamard-Walsh operator).
        :param cached_idxs: Indices used in the forward projection, determining
          the column height of the Hadamard-Walsh operator. See ``sparseproj``
          docstring.
        :param cached_noise: Noise used in the forward projection. See
          ``sparseproj`` docstring.
        :param out: If given, flat tensor of shape ``(out_dims,)`` where the
          output will be written. Otherwise, a new tensor will be created.
        :returns: The tuple ``(x, mask, idxs, noise)`` (see ``fjlt``).
        """
        assert len(y.shape) == 1, "Only flat inputs supported!"
        assert out_dims == cls.pow2roundup(out_dims), \
            "Only out_dims == 2^k supported!"
        in_dims = len(y)
        # recreate projection parameters
        if cached_idxs is not None:
            idxs = cached_idxs
        else:
            idxs = cls.get_idxs(in_dims, out_dims, seed, y.device)
        #
        if cached_noise is not None:
            noise = cached_noise
        else:
            noise = cls.get_noise(
                in_dims, out_dims, 1, seed, y.dtype, y.device)
            assert noise.all(), "Noise contains zeros! Not allowed."
        # To invert projection, embed y and multiply using idxs and noise
        if out is None:
            out = torch.zeros(out_dims, dtype=y.dtype).to(y.device)
        else:
            out *= 0
        out[idxs] = y * noise
        # the inverse of Hadamard and Rademacher are themselves
        cls.hadamard(out, inplace=True)
        _, mask = cls.rademacher(out, seed=seed, inplace=True)
        #
        return out, mask, idxs, noise


# ##############################################################################
# # SENSING MATRIX
# ##############################################################################
class JLTSensingMatrix:
    """
    Matrix-free PyTorch implementation of a Random projection via the Fast
    Johnson-Lindenstrauss Transform. It does not support transposition via
    ``self.T``, but it does support forward and backward matrix-vector
    multiplications e.g. via::

      self @ x
      y @ self

    The left- and right-handside matmuls can be used then to replace the need
    for transposition. For example, ``(self.T @ self) @ x`` can be computed
    via ``(self @ x) @ self``.
    """

    def __init__(self, shape, dtype=torch.float64, device="cpu",
                 scale=1.0, seed=None):
        """
        :param shape: Pair with ``(height, width)``.
        :param scale: Ideally, ``1/l``, where ``l`` is the average diagonal
          value of the covmat ``A.T @ A``, where ``A`` is a FastJLT operator,
          so that ``l2norm(x)`` approximates ``l2norm(Ax)``.
        """
        self.shape = shape
        h, w = shape
        assert w == FastJLT.pow2roundup(w), \
            "Only inputs with pow2 width supported!"
        assert h <= w, "Height > width not supported!"
        #
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.scale = scale
        # projection parameters: avoid having to compute these at each pass.
        self.idxs = FastJLT.get_idxs(h, w, seed, device)
        self.noise = FastJLT.get_noise(h, w, scale, seed, dtype, device)

    def matvec(self, x):
        """
        Forward matrix-vector multiplication.

        :param x: A flat torch tensor with ``self.shape[1]``.
        :returns: A flat torch tensor ``y= JLT @ x`` of same dtype and device
          as ``x`` and with ``self.shape[0]``.
        """
        y, _, _, _ = FastJLT.fjlt(
            x, self.shape[0], self.seed, self.idxs, self.noise)
        return y

    def rmatvec(self, y, out=None):
        """
        Backward vector-matrix multiplication.

        :param y: A flat torch tensor with ``self.shape[0]``.
        :param out: Optionally, tensor where to write the output ``x``. Must
          match ``x`` in shape, dtype and device.
        :returns: A flat torch tensor ``x= JLT.T @ y`` of same dtype and device
          as ``y`` and with ``self.shape[1]``. If ``out`` was given, it returns
          ``out``.
        """
        x, _, _, _ = FastJLT.transp_fjlt(y, self.shape[1], self.seed,
                                         self.idxs, self.noise, out=out)
        return x

    def getrow(self, idx, out=None):
        """
        Make explicit the ``idx`` row of this matrix and return it.
        :param out: If given, the row will be written here.
        :returns: The corresponding row with ``self.shape[1]`` entries.
        """
        v = torch.zeros(self.shape[0], dtype=self.dtype).to(self.device)
        v[idx] = 1
        result = self.rmatvec(v, out)
        return result

    def getcol(self, idx):
        """
        Make explicit the ``idx`` column of this matrix and return it.
        :returns: The corresponding column with ``self.shape[0]`` entries.
        """
        v = torch.zeros(self.shape[1], dtype=self.dtype).to(self.device)
        v[idx] = 1
        result = self.matvec(v)
        return result

    def __repr__(self):
        """
        """
        clsname = self.__class__.__name__
        s = f"<{clsname} {self.shape} ({self.dtype}, " + \
            "{self.device}, scale={self.scale})>"
        return s

    # operator interfaces
    def __matmul__(self, x):
        """
        Defining forward matrix-vector operation ``self @ x``.
        """
        result = self.matvec(x)
        return result

    def __rmatmul__(self, x):
        """
        Defining backward vector-matrix operation ``x @ self``.
        """
        result = self.rmatvec(x)
        return result

    def __imatmul__(self, x):
        """
        Defining assignment matmul operator ``@=``.
        """
        raise NotImplementedError

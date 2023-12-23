# PyTorchSketchySVD


### TODO:

- [x] Change design of Gaussian and synth: just functions returning the tensor, don't reimplement matmul!
- [x] Generate synthetic test matrices


- [x] Fast random GPU-compatible projections
  - [x] scrambled subsampled randomized Fourier transform (SSRFT): recommended, outperforms Gaussian
  - [x] Gaussian (for numerical testing)
- [ ] GPU-compatible, stable QR factorization  https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
- [ ] Solve least squares pseudoinverse via CG https://math.stackexchange.com/a/4533431
- [x] A priori parametrization: implement equations from section 5.4.2 (page 15)
- [ ] Put everything together to get the SSVD
- [ ] Centering step?
- [ ] Post-hoc precision estimation
  - [ ] Also use to determine rank (how? see paper)
- [ ] Tests
  - [x] SSRFT
  - [ ] Synthetic matrices (take actual SVD and compare spectra, check symmetry...)
  - [ ] CG
  - [ ] QR
- [ ] Maybe
  - [ ] Pivoted QR? https://github.com/pytorch/pytorch/issues/82092



### Implementation notes:


* https://github.com/alpyurtsever/SKETCH

```
pip install torch-dct
```

NOTE: A symmetric matrix allows us to shave half the computations, but IT NEEDS TO BE PSD, otherwise if the eigvals are negative a SVD is not defined/doable.

# PyTorchSketchySVD


### TODO:

- [x] Change design of Gaussian and synth: just functions returning the tensor, don't reimplement matmul!
- [x] Generate synthetic test matrices


- [x] Fast random GPU-compatible projections
  - [x] scrambled subsampled randomized Fourier transform (SSRFT): recommended, outperforms Gaussian
  - [x] Gaussian (for numerical testing)
- [x] GPU-compatible, stable QR factorization  https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
- [x] Solve least squares pseudoinverse via CG https://math.stackexchange.com/a/4533431
- [x] A priori parametrization: implement equations from section 5.4.2 (page 15)
- [x] Put everything together to get the SSVD
- [x] Post-hoc precision estimation
  - [x] Also use to determine rank (how? see paper, 6.5)
- [ ] Tests
  - [x] SSRFT
  - [ ] Synthetic matrices (take actual SVD and compare spectra, check symmetry...)
  - [ ] QR (check different -large shapes, Q should be orthogonal)
  - [ ] Test SVD (hard correctness) for family of matrices
  - [ ] Test post-hoc analysis: close enough estimate, correctness of lower/upper scree bounds
- [ ] Implement and test symmetric SSVD (needs to be PSD!!!)
- [ ] Add CI, autodocs, codecov, readme, packager


- [ ] Maybe

  - [ ] Centering matrix?
  - [ ] Pivoted QR? https://github.com/pytorch/pytorch/issues/82092



### Implementation notes:


* https://github.com/alpyurtsever/SKETCH

```
pip install torch-dct
```

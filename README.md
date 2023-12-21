# PyTorchSketchySVD


### TODO:

- [ ] Finalize setup and write unit tests for CG and SSRFT

- [ ] Fast random GPU-compatible projections
  - [x] scrambled subsampled randomized Fourier transform (SSRFT): recommended, outperforms Gaussian
  - [ ] Gaussian (for numerical testing)
  - [ ] FJLT (unstable, needs float64 precision)
- [ ] GPU-compatible, stable QR factorization  https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
  - [ ] Optionally in-place (not default since matrix evaluations are more expensive than memory)
- [ ] Post-hoc precision estimation


### Implementation notes:

* Centering step seems important

* https://github.com/alpyurtsever/SKETCH


USING CG TO SOLVE PSEUDOINVERSES:
https://math.stackexchange.com/a/4533431


```
pip install torch-dct
```

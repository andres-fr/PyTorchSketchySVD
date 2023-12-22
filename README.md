# PyTorchSketchySVD


### TODO:

- [ ] Change design of Gaussian and synth: just functions returning the tensor, don't reimplement matmul!
- [ ] Generate synthetic test matrices


- [ ] Fast random GPU-compatible projections
  - [x] scrambled subsampled randomized Fourier transform (SSRFT): recommended, outperforms Gaussian
  - [x] Gaussian (for numerical testing)
- [ ] GPU-compatible, stable QR factorization  https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
  - [ ] Optionally in-place (not default since matrix evaluations are more expensive than memory)
- [ ] Post-hoc precision estimation
- [ ] Tests
  - [x] SSRFT
  - [x] Gaussian
  - [ ] CG
  - [ ] QR
- [ ] Maybe
  - [ ] Pivoted QR? https://github.com/pytorch/pytorch/issues/82092

### Implementation notes:

* Centering step seems important

* https://github.com/alpyurtsever/SKETCH


USING CG TO SOLVE PSEUDOINVERSES:
https://math.stackexchange.com/a/4533431


```
pip install torch-dct
```

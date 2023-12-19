# PyTorchSketchySVD


### TODO:

- [ ] Fast random GPU-compatible projections
  - [ ] via scrambled subsampled randomized Fourier transform (SSRFT): recommended, outperforms Gaussian
  - [ ] via FJLT (unstable, needs float64 precision)
- [ ] GPU-compatible, stable QR factorization
  - [ ] Optionally in-place (not default since matrix evaluations are more expensive than memory)
- [ ] Post-hoc precision estimation


### Implementation notes:

* Centering step seems important

* https://github.com/alpyurtsever/SKETCH



```
pip install torch-dct
```

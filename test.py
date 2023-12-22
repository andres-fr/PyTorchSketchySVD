#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


# import numpy as np
import torch

from pytorch_ssvd.dimredux import SSRFT
from pytorch_ssvd.cg import CG

from pytorch_ssvd.synthetic_matrices import (
    LowRankNoiseMatrix,
    PolyDecayMatrix,
    ExpDecayMatrix,
)
import matplotlib.pyplot as plt


#


DEVICE = "cuda"
SEED = 0b1110101001010101011
IN, OUT, DTYPE = 100, 100, torch.float64


mm = LowRankNoiseMatrix((OUT, IN), rank=10, snr=1e-1, device="cuda")
pm = PolyDecayMatrix((OUT, IN), rank=10, decay=0.5, symmetric=False)
em = ExpDecayMatrix((OUT, IN), rank=10, decay=0.1, symmetric=True)


# import matplotlib.pyplot as plt
# plt.clf(); plt.imshow(em._weights.cpu()); plt.show()

# aa, bb, cc = torch.linalg.svd(em._weights)
# plt.clf(); plt.loglog(bb); plt.show()
breakpoint()


ssrft = SSRFT((OUT, IN), seed=SEED)
# aa = torch.linspace(0, 10, IN, dtype=DTYPE).to(DEVICE)
# bb = ssrft @ aa
# cc = bb @ ssrft

# torch.allclose(aa, cc)


kk = torch.randn((100, 13), dtype=DTYPE).to(DEVICE)
qq = ssrft @ kk

zz = (qq.T @ ssrft).T


# t=aa.cpu(); plt.clf(); plt.plot(t); plt.show()
breakpoint()

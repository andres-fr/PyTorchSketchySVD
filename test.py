#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


# import numpy as np
import torch

from pyssvd.dimredux import SSRFT
from pyssvd.cg import CG

import matplotlib.pyplot as plt


#


DEVICE = "cuda"
SEED = 0b1110101001010101011
IN, OUT, DTYPE = 100, 100, torch.float64

ssrft = SSRFT((OUT, IN), seed=SEED)
aa = torch.linspace(0, 10, IN, dtype=DTYPE).to(DEVICE)
bb = ssrft @ aa
cc = bb @ ssrft

torch.allclose(aa, cc)


# t=aa.cpu(); plt.clf(); plt.plot(t); plt.show()
breakpoint()

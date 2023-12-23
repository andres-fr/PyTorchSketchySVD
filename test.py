#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


# import numpy as np
import torch

from pytorch_ssvd.synthmat import SynthMat
from pytorch_ssvd.ssvd import a_priori_hyperparams
from pytorch_ssvd.sketching import SSRFT
from pytorch_ssvd.cg import CG


import matplotlib.pyplot as plt


# ##############################################################################
# # GLOBALS
# ##############################################################################
DEVICE = "cuda"
SEED = 0b1110101001010101011
IN, OUT, DTYPE = 1000, 500, torch.float64
RANK = 10
#
MEMORY_BUDGET = 100_000
OUTER_K, CORE_K = a_priori_hyperparams((OUT, IN), MEMORY_BUDGET)


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################

# 1. create test matrix and compute its hard SVD
# mm = SynthMat.lowrank_noise((OUT, IN), rank=10, snr=1e-1, device="cuda")
# em = SynthMat.exp_decay((OUT, IN), rank=10, decay=0.1, symmetric=True)
# mat = SynthMat.exp_decay(
#     (OUT, IN), rank=RANK, decay=0.1, symmetric=False, dtype=DTYPE, device=DEVICE
# )
mat = SynthMat.poly_decay(
    (OUT, IN), rank=RANK, decay=2, symmetric=False, dtype=DTYPE, device=DEVICE
)
U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
# plt.clf(); plt.imshow(em.cpu()); plt.show()
# plt.clf(); plt.loglog(S.cpu()); plt.show()


# 2. draw random measurements and perform QR decomposition
left_outer_ssrft = SSRFT((OUTER_K, OUT), seed=SEED)
right_outer_ssrft = SSRFT((OUTER_K, IN), seed=SEED + 1)
left_core_ssrft = SSRFT((CORE_K, OUT), seed=SEED + 2)
right_core_ssrft = SSRFT((CORE_K, IN), seed=SEED + 3)
#
lo_measurements = (
    torch.eye(OUTER_K, dtype=DTYPE).to(DEVICE) @ left_outer_ssrft
) @ mat
ro_measurements = (
    mat @ (torch.eye(OUTER_K, dtype=DTYPE).to(DEVICE) @ right_outer_ssrft).T
)
core_measurements = (
    (torch.eye(CORE_K, dtype=DTYPE).to(DEVICE) @ left_core_ssrft)
    @ mat
    @ (torch.eye(CORE_K, dtype=DTYPE).to(DEVICE) @ right_core_ssrft).T
)

# 3. Perform QR decompositions of outer measurements
lo_Q = torch.linalg.qr(lo_measurements.T)[0].T
ro_Q = torch.linalg.qr(ro_measurements)[0]


# 4. Solve core matrix to yield initial approximation
left_core = left_core_ssrft @ ro_Q
right_core = right_core_ssrft @ lo_Q.T
core = torch.linalg.lstsq(left_core, core_measurements).solution
core = torch.linalg.lstsq(right_core, core.T).solution

# 5. SVD of core matrix and truncated approximation
core_U, core_S, core_Vt = torch.linalg.svd(core)
# plt.clf(); plt.plot(core_S.cpu()); plt.show()

"""
a) take SVD, and grab top r pairs
b) Multiply Q matrices by truncated SVD ones, we are done.

See algorithm 4.4 in page 13

"""


breakpoint()

# 6. A posteriori precision and rank estimation

"""
???
"""

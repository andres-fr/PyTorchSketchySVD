#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


# import numpy as np
import torch

from pytorch_ssvd.synthmat import SynthMat
from pytorch_ssvd.sketching import SSRFT
from pytorch_ssvd.ssvd import a_priori_hyperparams, ssvd, truncate_core

import matplotlib.pyplot as plt


# ##############################################################################
# # GLOBALS
# ##############################################################################
DEVICE = "cuda"
SEED = 0b1110101001010101011
IN, OUT, DTYPE = 1000, 1000, torch.float64
RANK = 10
#
MEMORY_BUDGET = 100_000
OUTER_DIM, CORE_DIM = a_priori_hyperparams((OUT, IN), MEMORY_BUDGET)


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################

# 1. create test matrix and compute its hard SVD
# mm = SynthMat.lowrank_noise((OUT, IN), rank=10, snr=1e-1, device="cuda")
# em = SynthMat.exp_decay((OUT, IN), rank=10, decay=0.1, symmetric=True)
mat = SynthMat.exp_decay(
    (OUT, IN), rank=RANK, decay=0.1, symmetric=True, dtype=DTYPE, device=DEVICE
)
# mat = SynthMat.poly_decay(
#     (OUT, IN), rank=RANK, decay=2, symmetric=False, dtype=DTYPE, device=DEVICE
# )
U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
# plt.clf(); plt.imshow(em.cpu()); plt.show()
# plt.clf(); plt.loglog(S.cpu()); plt.show()


lo_Qt, core_U, core_S, core_Vt, ro_Qt = ssvd(
    mat, DEVICE, DTYPE, OUTER_DIM, CORE_DIM, SEED
)
mat_recons = lo_Qt @ core_U @ torch.diag(core_S) @ core_Vt @ ro_Qt
# plt.clf(); plt.plot(core_S.cpu()); plt.show()
# plt.clf(); plt.imshow(mat.cpu()); plt.show()


trunc_U, trunc_S, trunc_Vt = truncate_core(core_U, core_S, core_Vt, 10)

mat10 = U[:, :10] @ torch.diag(S[:10]) @ Vt[:10, :]
mat10_recons = lo_Qt @ trunc_U @ torch.diag(trunc_S) @ trunc_Vt @ ro_Qt


print(torch.dist(mat, mat_recons))
print(torch.dist(mat10, mat10_recons))
breakpoint()

# 6. A posteriori precision and rank estimation

"""
???
"""

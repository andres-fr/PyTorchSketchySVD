#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
Section 7.3.1. from the paper

TTODO:
1. Implement the 3 types of matrices. We also want them symmetric!
2. Figure a way to test or verify them, e.g. a plot?

Once we are done, we can start implementing the SVD, with 2 zwischenstops:
1. Create a symm matrix
2. Do left rand measurement and sketch (also symmetric)
3. Do QR of left measurement (IMPLEMENT/TEST) and solve leas squares of core via CG (IMPLEMENT/TEST)
4. Put everything together to get the SVD
"""

import torch


# ##############################################################################
# # ERRORS
# ##############################################################################
class NoFlatError(Exception):
    """ """

    pass


class BadShapeError(Exception):
    """ """

    pass


# ##############################################################################
# # REPRODUCIBLE NOISE
# ##############################################################################

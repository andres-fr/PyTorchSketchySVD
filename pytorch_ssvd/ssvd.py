#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""

import math


# ##############################################################################
# # HYPERPARAMETERS
# ##############################################################################
def a_priori_hyperparams(
    matrix_shape,
    memory_budget,
    complex_data=False,
):
    """
    :param int memory_budget: In number of matrix entries.
    :returns: The pair ``(k, s)``, where the first integer is the optimal
      number of outer sketch measurements, and the second one is the
      corresponding number of core measurements.
    """
    m, n = matrix_shape
    alpha = 0 if complex_data else 1
    mn4a = m + n + 4 * alpha
    budget_root = 16 * (memory_budget - alpha**2)
    #
    k = math.floor((1 / 8) * (math.sqrt(mn4a**2 + budget_root) - mn4a))
    s = math.floor(math.sqrt(memory_budget - k * (m + n)))
    #
    return k, s

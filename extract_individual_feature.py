# -*- coding: utf-8 -*-
import numpy as np


def solve_all():

    A = np.array([[6, 4, 1],
                  [1, 8, -2],
                  [1, 8, -2]])
    b = np.array([np.array([5, 3, 1]), np.array([8, 6, 2]), np.array([7, 1, 4])])

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.linalg.LinAlgError:
        x = "singular"

    print(x)
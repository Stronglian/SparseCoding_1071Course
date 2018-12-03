# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:38:32 2018

@author: StrongPria
"""

import numpy as np

A = np.matrix([[ 1, -1], [ 0,  1],[ 1,  0]])

u, s, vh = np.linalg.svd(A)
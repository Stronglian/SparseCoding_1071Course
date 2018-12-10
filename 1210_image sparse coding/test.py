# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:27:41 2018

@author: StrongPria
"""
import numpy as np
a = np.array([[ i*j+i for i in range(0,5)] for j in range(0,3)])
print(a)
print(a.shape)
for j in range(5):
    
    for i in range(5):
        print(i*j+i, end="->")
    print()
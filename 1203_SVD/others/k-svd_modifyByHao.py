# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:25:52 2018

@author: Jun
"""
import numpy as np
from prettytable import PrettyTable


#totalCoefficients = np.array([[0, 0.4706, 0, 0, 1.6313, 0, 0, -0.1935, 0, 0],
#                  [0, 0, 0.0348, 0, 0, 0, 0, 0, -0.2480, 1.8362],
#                  [0, 0, -0.6771, 0, 0, 0, -0.9391, 0, 0.7716, 0],
#                  [0, 1.0685, 0, 0, 0, 0, 0, -0.8023, 0, 0],
#                  [-1.2111, 0, 0.2873, 0, 0, 0, 0, 0, 0, 0.1716],
#                  [0, 0.1206, 0, 0, 1.3596, 0, 0, -0.046, 0, 0]]).T
#
#new_coefficients = np.copy(totalCoefficients)
#
#DICTIONARY = np.array([[      0, -0.8837, 0.7837,  0.7355, -0.2433, -0.5711,  0.8960,  0.5046,  0.2006, -0.5627],
#                       [-0.2788,  0.2019, 0.6196,  0.6180, -0.1908,  0.2891, -0.2179,  0.8941,  0.2251,  0.8207],
#                       [ 0.9604,  0.4223, 0.0432, -0.2777, -0.9510,  0.7683,  0.3869, -0.6265, -0.9535, -0.0997]])
#
#X = np.array([[-0.9471, -1.0559, -1.2173, -1.3493,  0.1286, -0.4606],
#              [-0.3744,  1.4725, -0.0412, -0.2611,  0.6565, -0.2624],
#              [-1.1859,  0.0557, -1.1283,  0.9535, -1.1678, -1.2132]])



X = np.array([[-0.9471, -0.3744, -1.1859 ],
                 [-1.0559, 1.4725, 0.0557],
                 [-1.2173, -0.0412, -1.1283 ],
                 [-1.3493, -0.2611, 0.9535 ],
                 [0.1286, 0.6565, -1.1678 ],
                 [-0.4606, -0.2624, -1.2132 ]]).T

DICTIONARY = np.array([[0, -0.2788,  0.9604],
                       [-0.8837, 0.2019, 0.4223],
                       [0.7837, 0.6196, 0.0432],
                       [0.7355, 0.6180, -0.2777],
                       [-0.2433, -0.1908, -0.9510],
                       [-0.5711, 0.2891, 0.7683],
                       [0.8960, -0.2179, 0.3869],
                       [0.5046, 0.8941, -0.6265],
                       [0.2006, 0.2251, -0.9535],
                       [-0.5627, 0.8207, -0.0997]]).T

totalCoefficients = np.array([[0, 0.4706, 0, 0, 1.6313, 0, 0, -0.1935, 0, 0],
                  [0, 0, 0.0348, 0, 0, 0, 0, 0, -0.2488, 1.8362],
                  [0, 0, -0.6771, 0, 0, 0, -0.9391, 0, 0.7716, 0],
                  [0, 1.0685, 0, 0, 0, 0, 0, -0.8023, 0, 0],
                  [-1.2111, 0, 0.2873, 0, 0, 0, 0, 0, 0, 0.1716],
                  [0, 0.1206, 0, 0, 1.3596, 0, 0, -0.046, 0, 0]]).T




#data = np.array([[-0.9471, -0.3744, -1.1859 ],
#                 [-1.0559, 1.4725, 0.0557],
#                 [-1.2173, -0.0412, -1.1283 ],
#                 [-1.3493, -0.2611, 0.9535 ],
#                 [0.1286, 0.6565, -1.1678 ],
#                 [-0.4606, -0.2624, -1.2132 ]])
#
#dictionary = np.array([[0, -0.2788,  0.9604],
#                       [-0.8837, 0.2019, 0.4223],
#                       [0.7837, 0.6196, 0.0432],
#                       [0.7355, 0.6180, -0.2777],
#                       [-0.2433, -0.1908, -0.9510],
#                       [-0.5711, 0.2891, 0.7683],
#                       [0.8960, -0.2179, 0.3869],
#                       [0.5046, 0.8941, -0.6265],
#                       [0.2006, 0.2251, -0.9535],
#                       [-0.5627, 0.8207, -0.0997]])



DATA_NUMBERS = X.shape[1]
D_NUMBERS = DICTIONARY.shape[1]
D_DIM = DICTIONARY.shape[0]

def update_dictionary(totalCoefficients):
    for i in range(D_NUMBERS):
        data_idxs = np.argwhere(totalCoefficients[i,:] != 0)
        if data_idxs.size > 0 :
            t_C = totalCoefficients.copy()
            t_C[i, data_idxs] = 0
            print(t_C)
            print('D'+str(i+1)+'修正中')
            new_sparse_data = DICTIONARY @ t_C[:, data_idxs].reshape(10, data_idxs.size)
            e = np.array(X[:, data_idxs].reshape(3, data_idxs.size) - new_sparse_data)
            print(e.T)
#            print('E',E)
            u,s,v = np.linalg.svd(e)
            print('u',u)
            print('v',v)
            print('s',s)
            new_dic = u[:,0]
#            print('new_dic',new_dic)
            DICTIONARY[:,i] = new_dic
#            print(DICTIONARY[:,i])
            a = s[0] * v[:, 0]
            totalCoefficients[i, data_idxs] = a.reshape(data_idxs.size,1)
            print('===========',a)
#print(totalCoefficients)  
            
            
            
print('Coefficients')
table1 = PrettyTable()
temp = ['字典']
temp.extend(['x'+str(i+1) for i in range(DATA_NUMBERS)])
table1.field_names = temp
for i in range(D_NUMBERS):
    n = [i+1]
    n.extend(totalCoefficients[i,:])
    table1.add_row(n)
print(table1)

DICTIONARY = np.round(DICTIONARY,4)
table = PrettyTable()
temp = ['DIM']
temp.extend(['atom'+str(i+1) for i in range(D_NUMBERS) ])
table.field_names = temp
for i in range(D_DIM):
    n = [i+1]
    n.extend(DICTIONARY[i,:])
    table.add_row(n)
print(table)

print('===============================================================')
update_dictionary(totalCoefficients)     

print('姿均')
print('【DICTIONARY】')
DICTIONARY = np.round(DICTIONARY,4)
table = PrettyTable()
temp = ['DIM']
temp.extend(['atom'+str(i+1) for i in range(D_NUMBERS) ])
table.field_names = temp
for i in range(D_DIM):
    n = [i+1]
    n.extend(DICTIONARY[i,:])
    table.add_row(n)
print(table)

totalCoefficients = np.round(totalCoefficients,4)
print('【Coefficients】')
table1 = PrettyTable()
temp = ['字典']
temp.extend(['x'+str(i+1) for i in range(DATA_NUMBERS)])
table1.field_names = temp
for i in range(D_NUMBERS):
    n = [i+1]
    n.extend(totalCoefficients[i,:])
    table1.add_row(n)
print(table1)
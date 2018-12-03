# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:24:18 2018

@author: StrongPria
"""
import numpy as np
np.set_printoptions(precision=5)
#%%
class ISTA:
    def __init__(self, dictionary, data, learningRate, weightL1, initialCode_one = None):
        self.D = dictionary.copy()
        self.Y = data.copy()
        self.beta  = learningRate
        self.gamma = weightL1
        
        if initialCode_one is None:
            self.alpha_one = np.matrix(0.1 * np.ones((dictionary.shape[1],1)))
        else:
            self.alpha_one = initialCode_one.copy()
        
        print( "D: shape", self.D.shape, "There are", self.D.shape[1], "atoms with","length", self.D.shape[0])
        print( "Y: shape", self.Y.shape, "There are", self.Y.shape[1], "signal with","length", self.Y.shape[0])
        return
    
    def normDist(self, inputCode):
        return np.linalg.norm(inputCode, ord = 2)
    
    def FLOW_oneAlpha_oneStep(self, data, initailCode = None):
        # initail
        if initailCode is None:
            alpha = self.alpha_one.copy()
        else:
            alpha = initailCode.copy()
        zeroMatrix = np.matrix(np.zeros((alpha.shape[0],1)))
        # START
        oldCode = alpha.copy()
        # Alpha lpha lpha is updated from reconstruction error 
        
        alpha = alpha - (self.beta * (self.D.T * (self.D * alpha - data)))
        coe1 = alpha.copy()
        # Alpha lpha lpha is updated from L1-norm; i.e., shrinkage by th
        absTmp = np.abs(alpha) - (self.beta * self.gamma)
        
        maxTmp = np.max( [absTmp, zeroMatrix], axis = 0)
        
        alpha = np.multiply(np.sign(alpha) , maxTmp)
        
        coe2 = alpha.copy()
        #residual
        r = data - self.D * alpha 
        resd = np.sqrt(r.T  * r) 
        
        dif = oldCode - alpha
        stop = np.sqrt(dif.T * dif) # stop = self.normDist(dif)
        
        return alpha, np.squeeze(stop), np.squeeze(resd), coe1, coe2
    
    def FLOW_oneAlpha(self, data, convergeThreshold = 0.005, max_iter = 10):
        count = 0
        tmpList = []
        
        info = self.FLOW_oneAlpha_oneStep(data, initailCode = None)
        tmpList.append(info)
        while info[1] >= convergeThreshold and ( True if( max_iter is None) else (count < max_iter -1 )):# 外面做一次了
#            print("iter:", count)
#            print(info[1])
            info = self.FLOW_oneAlpha_oneStep(data, initailCode = info[0])
            
            count += 1
            tmpList.append(info)
            
#        print("resd", info[-1])
        return info[0], tmpList
    
    def FLOW_allData(self, convergeThreshold = 0.005, max_iter = 10):
        alphaALL = np.matrix(np.zeros((self.D.shape[1], self.Y.shape[1])))
#        print()
#        for i, data in enumerate(self.Y):
        for i in range(self.Y.shape[1]):
#            print("data", i, self.Y[:,i].T)
            tmpAlpha = self.FLOW_oneAlpha(self.Y[:,i], convergeThreshold, max_iter)[0]
            alphaALL[:, i] = tmpAlpha.copy()
        return alphaALL
    
    def VarifyAll(self, sparseCode, boolRound = True):
        Y_hat = self.D * np.matrix(sparseCode)
        residualALL = np.linalg.norm(self.Y - Y_hat, ord = 2, axis = 0)
        if boolRound:
            Y_hat = np.round(Y_hat, 5)
            residualALL = np.round(residualALL, 5)
        return Y_hat, residualALL
#%% 
from prettytable import PrettyTable #pip install PTable

def ShowAnswer(alpha, residual, data, data_hat, boolRound = True, intRoundDigits = 5):
    """ """
    if boolRound:
        alpha    = np.round(alpha, intRoundDigits)
        residual = np.round(residual, intRoundDigits)
        data     = np.round(data, intRoundDigits)
        data_hat = np.round(data_hat, intRoundDigits)
    # 建表 - alpha
    table_alpha = PrettyTable()
    tmp = ['#']
    tmp.extend(["x"+str(i+1)+"'s SC" for i in range(alpha.shape[1])])
    table_alpha.field_names = tmp.copy()
    for i, tmpLi in enumerate(alpha):
        tmp = [ str(i+1) ]
        tmp.extend(np.squeeze(np.array(tmpLi)))
        table_alpha.add_row(tmp)
    # 建表 - residual
    table_residual = PrettyTable()
    table_residual.field_names = ["#", "origin", "After Coding", "residual"]
    for i in range(data.shape[1]):
        table_residual.add_row([str(i+1), data[:, i].T, data_hat[:, i].T, residual[i]])
    # 顯示
    print(table_alpha)
    print(table_residual)
    return
#%% 應用
if __name__ == "__main__":
    import time
    startTime = time.time()
#    inputDataSet = 1
    
#    if inputDataSet == 0:
#        #輸入資料 - 範例
#        D_org = np.matrix([[0.5774, -0.4083,  0.7071], 
#                           [0.5774, -0.4083,  0.7071],
#                           [0.5774,  0.8165,  0.    ]]) 
#        Y = np.matrix([[0.5,  0.8,  1. ],
#                       [0.3, -0.1,  2. ], 
#                       [0.1,  2. , -1.5]])
#        initialCode = np.matrix(np.zeros((D_org.shape[1],1)))
#        ista = ISTA(D_org, Y, 0.07, 0.15, initialCode_one = initialCode)
#        tmp = ista.FLOW_oneAlpha(Y[:,0], convergeThreshold = 0.001)
#    
#    
#    elif inputDataSet == 1:
    ## 輸入資料 - 題目
    D_org = np.matrix([[ 0.0000, -0.8837, 0.7837,  0.7355, -0.2433, -0.5711,  0.8960,  0.5046,  0.2006, -0.5627],
                       [-0.2788,  0.2019, 0.6196,  0.6180, -0.1908,  0.2891, -0.2179,  0.8941,  0.2251,  0.8207],
                       [ 0.9604,  0.4223, 0.0432, -0.2777, -0.9510,  0.7683,  0.3869, -0.6265, -0.9535, -0.0997]])
    Y = np.matrix([[-0.9471, -1.0559, -1.2173, -1.3493,  0.1286, -0.4606],
                   [-0.3744,  1.4725, -0.0412, -0.2611,  0.6565, -0.2624],
                   [-1.1859,  0.0557, -1.1283,  0.9535, -1.1678, -1.2132]])

#        initialCode = np.matrix(0.1*np.ones((D_org.shape[1],1)))
    ista = ISTA(D_org, Y, 0.05, 0.1)
#        tmp = ista.FLOW_oneAlpha(Y[:,0]); raise ValueError
    alphaAll = ista.FLOW_allData()
#        alphaAll = np.array(alphaAll)
    # 殘差計算 residual
    Y_hat, residual = ista.VarifyAll(alphaAll)
        
#%%
#        # 顯示 - alpha

    ShowAnswer(alphaAll, residual, Y, Y_hat)
    print("DONE", time.time() - startTime,"sec.")
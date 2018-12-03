# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:07:40 2018

@author: StrongPria
"""

#%%
import numpy as np
#from scipy.spatial import distance
from prettytable import PrettyTable #pip install PTable
#%%
#題目
D_org = np.matrix([[ 0.0000, -0.8837, 0.7837,  0.7355, -0.2433, -0.5711,  0.8960,  0.5046,  0.2006, -0.5627],
                  [-0.2788,  0.2019, 0.6196,  0.6180, -0.1908,  0.2891, -0.2179,  0.8941,  0.2251,  0.8207],
                  [ 0.9604,  0.4223, 0.0432, -0.2777, -0.9510,  0.7683,  0.3869, -0.6265, -0.9535, -0.0997]])
X     = np.matrix([[-0.9471, -1.0559, -1.2173, -1.3493,  0.1286, -0.4606],
                  [-0.3744,  1.4725, -0.0412, -0.2611,  0.6565, -0.2624],
                  [-1.1859,  0.0557, -1.1283,  0.9535, -1.1678, -1.2132]])
##範例
#D_org = np.matrix([[0.5774, -0.4083,  0.7071], 
#                  [0.5774, -0.4083,  0.7071],
#                  [0.5774,  0.8165,  0.    ]]) 
#X = np.matrix([[0.5,  0.8,  1. ],
#              [0.3, -0.1,  2. ], 
#              [0.1,  2. , -1.5]])
#%% OMP
class OMP:
    def __init__(self, dictionary, dataAll, letter):
        """  """
        self.D = dictionary.copy()
        self.X = dataAll.copy()
        self.L = letter
        
        self.atomNum = self.D.shape[1]
        
        #print( "長度", D_org.shape[0], "的 有", D_org.shape[1], "個")
        print( "D: There are", self.D.shape[1], "atoms with","length", self.D.shape[0])
        print( "X: There are", self.X.shape[1], "signal with","length", self.X.shape[0])
        
        return
    
    def Cal_OneX_alpha_OMP(self, inputX, L):
        """ 
        算單一 signal (Xi) 的 alpha
        """
        #
        residual = inputX.copy()
        alpha = np.matrix(np.zeros((self.D.shape[1], 1))) #sparse code
        # D 轉換
        unusedD = self.D.copy()
        usedD = np.zeros_like(self.D.copy())
        for i in range(L):
            print("\niter:", i)
            # 算相似度
            correlated = residual.T * unusedD # np.dot(residual.T , unusedD) in matrix
            print("correlated", correlated)
            d_i = np.argmax( np.abs(correlated))
            print("d_i", d_i, alpha.shape)
            # 使用過不再使用
            usedD[:, d_i] = unusedD[:, d_i].copy()
            unusedD[:, d_i].fill(0)
            # 重新計算比例
#            if i != 0 and True:
            alpha = np.linalg.lstsq(usedD, inputX, rcond = None) [0]
#            else:
#                alpha[d_i, :] = correlated[:, d_i]
            print("alpha", alpha, sep = "\n")
            # 殘差計算
            residual = inputX - usedD * alpha
#            print("residual", residual, sep="\n")
        return alpha
    
    def Flow_all_OMP(self):
        """  """
        alphaALL = np.matrix(np.zeros((self.D.shape[1], self.X.shape[1])))
        
        for i in range(self.X.shape[1]):
            print("data", i, self.X[:,i].T)
            alpha = self.Cal_OneX_alpha_OMP(self.X[:, i], self.L)
            alphaALL[:, i] = alpha.copy()
        return alphaALL
    
    def VarifyAll(self, sparseCode, boolRound = False):
        """  """
        X_hat = self.D * np.matrix(sparseCode)
        residualALL = np.linalg.norm(self.X - X_hat, ord = 2, axis = 0)
        if boolRound:
            X_hat = np.round(X_hat, 4)
            residualALL = np.round(residualALL, 4)
        return X_hat, residualALL
    
#%% ISTA
class ISTA:
    def __init__(self, dictionary, data, learningRate, weightL1, initialCode_one = None):
        """  """
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
        """  """
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
        for i in range(self.Y.shape[1]):
#            print("data", i, self.Y[:,i].T)
            tmpAlpha = self.FLOW_oneAlpha(self.Y[:,i], convergeThreshold, max_iter)[0]
            alphaALL[:, i] = tmpAlpha.copy()
        return alphaALL
    
    def VarifyAll(self, sparseCode, boolRound = False):
        Y_hat = self.D * np.matrix(sparseCode)
        residualALL = np.linalg.norm(self.Y - Y_hat, ord = 2, axis = 0)
        if boolRound:
            Y_hat = np.round(Y_hat, 5)
            residualALL = np.round(residualALL, 5)
        return Y_hat, residualALL
    
#%% SHOW
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
#%%
if __name__ == "__main__":
    import time
    start_time = time.time()
    print("START")
    
    print("\nOMP", "="*50)
    omp = OMP(D_org, X, letter = 3)
#    tmp = omp.Cal_OneX_alpha_OMP(X[:, 1], 3)
    alphaALL_omp = omp.Flow_all_OMP()
    # 殘差計算 residual
    X_hat, residual_omp = omp.VarifyAll(alphaALL_omp)
    # 結果顯示
    ShowAnswer(alphaALL_omp, residual_omp, X, X_hat)
    
    print("\nISTA", "="*50)
    ista = ISTA(D_org, X, 0.05, 0.1)
    alphaAll_ista = ista.FLOW_allData(convergeThreshold = 0.005, max_iter = 10)
    # 殘差計算 residual
    X_hat, residual = ista.VarifyAll(alphaAll_ista)
    # 結果顯示
    ShowAnswer(alphaAll_ista, residual, X, X_hat)
    
    
    
    print("\nEND", time.time() - start_time, "sec.")
    
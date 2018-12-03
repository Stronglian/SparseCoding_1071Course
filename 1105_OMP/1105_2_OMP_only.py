    # -*- coding: utf-8 -*-
"""
交截圖與程式碼。
file: "ex for OMPISTA (student).doc"
OMP only
"""
"""
Use OMP to solve A sparse coding of X such that  min || DA - X|| 
where D = [d1  d2  …  d10], X = [x1  x2 … x6] OMP with L = 3, 
<br\>Hand-in both (1) python code; (2) snapshots of corresponding answer.

"""
"""
REFER:
    alog:
        https://blog.csdn.net/breeze5428/article/details/25122977
    2 - vecotr cosine
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.cosine.html
    norm - 0:
        https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.norm.html
    暫時解決 Singular matrix: 
        pseudo-inverse of a matrix(np.linalg.pinv)

"""
"""
目標：
"""
#%%
import numpy as np
#from scipy.spatial import distance
#%%
#題目
D_org = np.array([[ 0.0000, -0.8837, 0.7837,  0.7355, -0.2433, -0.5711,  0.8960,  0.5046,  0.2006, -0.5627],
                  [-0.2788,  0.2019, 0.6196,  0.6180, -0.1908,  0.2891, -0.2179,  0.5941,  0.2251,  0.8207],
                  [ 0.9604,  0.4223, 0.0432, -0.2777, -0.9510,  0.7683,  0.3869, -0.6265, -0.9535, -0.0997]])
X=np.array([[-0.9471, -1.0559, -1.2173, -1.3493,  0.1286, -0.4606],
            [-0.3744,  1.4725, -0.0412, -0.2611,  0.6565, -0.2624],
            [-1.1859,  0.0557, -1.1283,  0.9535, -1.1678, -1.2132]])
###範例
#D_org = np.array([[0.5774, -0.4083,  0.7071], 
#                  [0.5774, -0.4083,  0.7071],
#                  [0.5774,  0.8165,  0.    ]]) 
#X = np.array([[0.5,  0.8,  1. ],
#              [0.3, -0.1,  2. ], 
#              [0.1,  2. , -1.5]])
#%%
#其餘參數
L = 3
#L = 2
#%% 建置
class OMP():
    def __init__(self, dictInput, signalInput, letter):
        self.D = dictInput
        self.X = signalInput
        self.L = letter
        
        self.atomNum = self.D.shape[1]
        
        #print( "長度", D_org.shape[0], "的 有", D_org.shape[1], "個")
        print( "D: There are", self.D.shape[1], "atoms with","length", self.D.shape[0])
        print( "X: There are", self.X.shape[1], "signal with","length", self.X.shape[0])
        
    def Normal(self, dictInput):
        """ """
        dictAfterNorm = np.zero_likes(dictInput)
        
        return dictAfterNorm
    
    def Cal_OneX_alpha_OMP(self, inputX, L, boolRound = False):
        """ 
        算單一 signal (Xi) 的 alpha
        boolRound: 是否約分
        """
        inputX = np.matrix(inputX).T
        alpha = np.zeros(self.D.shape[1]) #sparse code
        residual = inputX.copy()
        # D 轉換區
        unuseD = np.matrix(self.D.copy())
        usedD  = np.matrix(np.zeros_like(unuseD)) 
        for i in range(L):
            #算相似度
            correlated = np.dot(residual.T, unuseD)
#            correlated = np.multiply(residual.T, unuseD)
#            correlated = residual.T * unuseD
            if boolRound:
                correlated = np.round(correlated, 4)
            d_i = np.abs(correlated).argmax()
            print( d_i, "from", correlated[0])
            # 殘差計算
            residual = inputX - correlated[0, d_i] * unuseD[:, d_i]
            if boolRound:
                residual = np.round(residual, 4)
#            print("new-residual", residual[0])
            # 使用過不再使用
            usedD[:, d_i] = unuseD[:, d_i].copy()
            unuseD[:, d_i].fill(0)
#            print(unuseD, usedD, sep="\n")
            # 重新計算比例
            alpha = np.linalg.lstsq(usedD, inputX, rcond = 0)[0]
#            alpha = np.linalg.pinv(usedD.T * usedD) * usedD.T * inputX
#            alpha = np.linalg.inv(usedD.T.T * usedD.T) * usedD.T.T * np.matrix(inputX)
#            print("alpha",alpha)
            
        if boolRound:
            return np.round(np.array(alpha).T, 4)
        else:
            return np.array(alpha).T
        
    def Flow_all_OMP(self, boolRound = False):
        """ 
        以 OMP 算所有的 X 配合 D 產生的 alpha
        """
#        alpha_X_All = np.zeros((self.X.shape[1], self.D.shape[1]))
        alpha_X_All = np.zeros((self.D.shape[1], self.X.shape[1]))
        for x_i in range(self.X.shape[1]):
            print("\n#"+str(x_i+1), self.X[:, x_i], "="*20)
            alpha_x = self.Cal_OneX_alpha_OMP(self.X[: ,x_i], self.L, boolRound)
#            print("alpha", alpha_x)
            alpha_X_All[:, x_i] = alpha_x.copy()
#            alpha_X_All[x_i] = alpha_x.copy()
#        self.sparseCode_OMP = alpha_X_All.T
        self.sparseCode_OMP = alpha_X_All
        return self.sparseCode_OMP
    
    def VarifyAll_OMP(self, boolRound = False):
        if boolRound:
            X_heat = np.round(self.D * np.matrix(self.sparseCode_OMP), 4)
            return np.round(np.linalg.norm(X-X_heat, ord = 2, axis = 0), 4)
        else:
            X_heat = self.D * np.matrix(self.sparseCode_OMP)
            return np.linalg.norm(X-X_heat, ord = 2, axis = 0)
                
#        print("residual", np.linalg.norm(X-X_heat, ord = 2, axis = 0))
    
#%%
boolRound = True # 進位與否
omp = OMP(D_org, X, letter=L)
alpha_X_All = omp.Flow_all_OMP(boolRound = boolRound)
#%% 驗證
residual = omp.VarifyAll_OMP(boolRound = False)
#for i in range(X.shape[1]):
#    print(X[:, i], "==="*5)
#print(D_org * np.matrix(alpha_X_All))
#%% 表格顯示
print( "有約分" if boolRound else "沒約分")
from prettytable import PrettyTable #pip install PTable
table = PrettyTable()

tmp = ['0']
tmp.extend(["x"+str(i+1)+"'s SC" for i in range(X.shape[1])])
table.field_names = tmp

for i, tmpLi in enumerate(alpha_X_All):
    tmp = [i+1]
    tmp.extend(tmpLi)
    table.add_row(tmp)
    
table.add_row(["="*6 for i in range(X.shape[1]+1)])

tmp = ['residual']
tmp.extend(residual)
table.add_row(tmp)
print(table)
    
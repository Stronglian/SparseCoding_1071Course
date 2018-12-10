# -*- coding: utf-8 -*-
"""
@author: StrongPria
@from: 1207_02_SVD_updateDict.py
"""
#%%
import numpy as np
from prettytable import PrettyTable #pip install PTable
#%%
def OwnPrint(inputArr, end = "\n", sep = " "):
    print(inputArr, "=>", inputArr.shape, end = "\n", sep = " ")
#    print(inputArr.shape)
    return
def ShowAnswer(alpha, dic, boolRound = True, intRoundDigits = 5):
    """ """
    alpha_ed = np.array(alpha)
    dic_ed   = np.array(dic)
    if boolRound:
        alpha_ed = np.round(alpha_ed,intRoundDigits)
        dic_ed   = np.round(dic_ed,  intRoundDigits)
    # 建表 - alpha
    table_alpha = PrettyTable()
    tmp = ['#']
    tmp.extend(["x"+str(i+1)+"'s SC" for i in range(alpha_ed.shape[1])])
    table_alpha.field_names = tmp.copy()
    for i, tmpLi in enumerate(alpha_ed):
        tmp = [ str(i+1) ]
        tmp.extend(np.squeeze(np.array(tmpLi)))
        table_alpha.add_row(tmp)
    # 建表 - dict
    table_dict = PrettyTable()
    tmp = ['#']
    tmp.extend(["atom"+str(i+1) for i in range(dic_ed.shape[1])])
    table_dict.field_names = tmp.copy()
    for i, tmpLi in enumerate(dic_ed):
        tmp = [ str(i+1) ]
        tmp.extend(np.squeeze(np.array(tmpLi)))
        table_dict.add_row(tmp)
    # 顯示
    print("="*10)
    print(table_alpha)
    print(table_dict)
    return
#%%
class SVD:
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
    def SetAlpha(self, inputAlpha):
        self.A = inputAlpha.copy()
        return
    def UpdateDict_useSVD_one(self, indexOfAtom, indexOfData, boolHaoSol = False):
        print(indexOfAtom, "=>", indexOfData, "="*10)
#        tmpD = self.D.copy()
#        tmpD[:, indexOfAtom].fill(0)
#        print("tmpD",tmpD)
#        print("X:")
#        print( *indexOfData, sep = " "*7)
#        OwnPrint(self.X[:, indexOfData])
#        print("A:")
#        OwnPrint(self.A[:, indexOfData])
#        print("X~ ", end = "")
#        print( *indexOfData, sep = " "*7)
#        OwnPrint(self.D * self.A[:, indexOfData])
#        print("D"+str(indexOfAtom))
#        OwnPrint(self.D[:, indexOfAtom])
#        print("A"+str(indexOfAtom))
#        OwnPrint(self.A[indexOfAtom, indexOfData])
#        print("D"+str(indexOfAtom), "*", "A"+str(indexOfAtom))
#        OwnPrint(self.D[:, indexOfAtom] * self.A[indexOfAtom, indexOfData])
#        print("E"*5)
#        OwnPrint(self.X[:, indexOfData] - self.D * self.A[:, indexOfData] + self.D[:, indexOfAtom] * self.A[indexOfAtom, indexOfData])
        if boolHaoSol:
            # 一樣結果
            t_C = self.A.copy()
            t_C[indexOfAtom, indexOfData] = 0
#            print(t_C)
            tmp_data = self.D * t_C[:, indexOfData]
            E = self.X[:, indexOfData] - tmp_data
        else:
            E = self.X[:, indexOfData] - self.D * self.A[:, indexOfData] + self.D[:, indexOfAtom] * self.A[indexOfAtom, indexOfData]
        print("E", E)
        #        print("SVD_T")
        #        u, s, v = np.linalg.svd(E.T) # need trans?
        #        print(u, s, v, sep = "\n")
        #        print("="*10, 1)
        #        print(np.dot(np.multiply(u,s),v))
        #        print(np.dot(np.dot(u, np.diag(s)),v))
        #        print("="*10, 1)
#        print("SVD")
        u, s, vh = np.linalg.svd(E) # need trans? 看E的變化應該選這個
#        v = vh.T
        print("u:", u, "s:", s, "vh:", vh, sep = "\n")
        #        print("="*10, 2)
        #        print(np.dot(np.multiply(u,s),v))
        #        print(np.dot(np.dot(u, np.diag(s)),v))
        #        print("="*10, 2)
        ### 哪一邊呢?
        print("->new d"+str(indexOfAtom), u[:, 0].T)
        self.D[:, indexOfAtom] = u[:, 0].copy()
        print("new d"+str(indexOfAtom), u[0, :])
#        self.D[:, indexOfAtom] = u[0, :].T.copy()
        
        #        print("new coe?", s[0] * v[:, 0])
        #        self.A[indexOfAtom, indexOfData] = (s[0] * v[:, 0]).T
#        newCoe = s[0] * vh[0, :]
        newCoe = s[0] * vh[:, 0].T
        print("new coe", newCoe)
        self.A[indexOfAtom, indexOfData] = newCoe
        return
    
    def UpdateDict_FLOW(self):
        usedAtomIndex, userX = np.where(self.A!=0)
#        print(usedAtomIndex)
        for indexOfAtom in range(self.D.shape[1]):
            print("\nindexOfAtom:", indexOfAtom)#, "=> d"+str((indexOfAtom)))
            tmp = np.where(usedAtomIndex == indexOfAtom)[0]
            if tmp.shape[0] == 0:
                continue
#            print(np.where(usedAtomIndex == indexOfAtom))
#            print("=>", userX[tmp])
            self.UpdateDict_useSVD_one(indexOfAtom, userX[tmp])
#            break
        return

#%%
if __name__ == "__main__":
    #題目
    D_org = np.matrix([[ 0.0000, -0.8837, 0.7837,  0.7355, -0.2433, -0.5711,  0.8960,  0.5046,  0.2006, -0.5627],
                       [-0.2788,  0.2019, 0.6196,  0.6180, -0.1908,  0.2891, -0.2179,  0.8941,  0.2251,  0.8207], #0.5941
                       [ 0.9604,  0.4223, 0.0432, -0.2777, -0.9510,  0.7683,  0.3869, -0.6265, -0.9535, -0.0997]])
    X     = np.matrix([[-0.9471, -1.0559, -1.2173, -1.3493,  0.1286, -0.4606],
                       [-0.3744,  1.4725, -0.0412, -0.2611,  0.6565, -0.2624],
                       [-1.1859,  0.0557, -1.1283,  0.9535, -1.1678, -1.2132]])
    A     = np.matrix([[ 0.    ,  0.    ,  0.    ,  0.    , -1.2111,  0.    ],
                       [ 0.4706,  0.    ,  0.    ,  1.0685,  0.    ,  0.1206],
                       [ 0.    ,  0.0348, -0.6771,  0.    ,  0.2873,  0.    ],
                       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
                       [ 1.6313,  0.    ,  0.    ,  0.    ,  0.    ,  1.3596],
                       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
                       [ 0.    ,  0.    , -0.9391,  0.    ,  0.    ,  0.    ],
                       [-0.1935,  0.    ,  0.    , -0.8023,  0.    , -0.046],
                       [ 0.    , -0.2488,  0.7716,  0.    ,  0.    ,  0.    ],
                       [ 0.    ,  1.8362,  0.    ,  0.    ,  0.1716,  0.    ]])
    
    #%%
    omp_tmp = SVD(D_org, X, 3)
    omp_tmp.SetAlpha(A)
    omp_tmp.UpdateDict_FLOW()
    ans_D = omp_tmp.D
    ans_A = omp_tmp.A
    #%%
    
    ShowAnswer(ans_A, ans_D)
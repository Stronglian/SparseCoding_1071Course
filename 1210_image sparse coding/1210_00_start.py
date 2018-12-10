# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:17:06 2018

@author: StrongPria
"""
#%%
import numpy as np
import cv2
import time
_startTime = time.time()
print("START\n\n")
#%%
#%% @from: 1126_6_BOTH_2.py
class OMP:
    def __init__(self, dictionary, dataAll, letter):
        """  """
        self.D = np.matrix(dictionary.copy())
        self.X = np.matrix(dataAll.copy())
        self.L = letter
        
        self.atomNum = self.D.shape[1]
        
        #print( "長度", D_org.shape[0], "的 有", D_org.shape[1], "個")
        print( "D: There are", self.D.shape[1], "atoms with","length", self.D.shape[0])
        print( "X: There are", self.X.shape[1], "signal with","length", self.X.shape[0])
        
        return
    
    def Cal_OneX_alpha_OMP(self, inputX, L, boolDeubg = False):
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
            if boolDeubg:
                print("\niter:", i)
            # 算相似度
            correlated = residual.T * unusedD # np.dot(residual.T , unusedD) in matrix
            if boolDeubg:
                print("correlated", correlated)
            d_i = np.argmax( np.abs(correlated))
            
            if boolDeubg:
                print("d_i", d_i, alpha.shape)
            # 使用過不再使用
            usedD[:, d_i] = unusedD[:, d_i].copy()
            unusedD[:, d_i].fill(0)
            # 重新計算比例
            if i != 0 :#or True: # 第一次到底要不要更新呢~
                alpha = np.linalg.lstsq(usedD, inputX, rcond = None) [0]
#                alpha = np.linalg.pinv(usedD.T * usedD) * usedD.T * inputX
            else:
                alpha[d_i, :] = correlated[:, d_i]
            
            if boolDeubg:
                print("alpha", alpha, sep = "\n")
            # 殘差計算
            residual = inputX - usedD * alpha
            if boolDeubg:
                print("residual", residual, sep="\n")
        return alpha
    
    def Flow_all_OMP(self, boolDeubg = False):
        """  """
        alphaALL = np.matrix(np.zeros((self.D.shape[1], self.X.shape[1])))
        
        for i in range(self.X.shape[1]):
            if boolDeubg:
                print("data", i, self.X[:,i].T)
            alpha = self.Cal_OneX_alpha_OMP(self.X[:, i], self.L, boolDeubg = boolDeubg)
            alphaALL[:, i] = alpha.copy()
        return alphaALL
    
    def VarifyAll(self, sparseCode, boolRound = True):
        """  """
        X_hat = self.D * np.matrix(sparseCode)
        residualALL = np.linalg.norm(self.X - X_hat, ord = 2, axis = 0)
        if boolRound:
            X_hat = np.round(X_hat, 4)
            residualALL = np.round(residualALL, 4)
        return X_hat, residualALL
#%% @from: 1207_02_SVD_updateDict.py
class SVD:
    def __init__(self, dictionary, dataAll):#, letter):
        """  """
        self.D = np.matrix(dictionary.copy())
        self.X = np.matrix(dataAll.copy())
        
        self.atomNum = self.D.shape[1]
        
        #print( "長度", D_org.shape[0], "的 有", D_org.shape[1], "個")
        print( "D: There are", self.D.shape[1], "atoms with","length", self.D.shape[0])
        print( "X: There are", self.X.shape[1], "signal with","length", self.X.shape[0])
        
        return
    def SetAlpha(self, inputAlpha):
        self.A = inputAlpha.copy()
        return
    def UpdateDict_useSVD_one(self, indexOfAtom, indexOfData, boolHaoSol = False):
#        print(indexOfAtom, "=>", indexOfData, "="*10)
        if boolHaoSol:
            # 一樣結果
            t_C = self.A.copy()
            t_C[indexOfAtom, indexOfData] = 0
#            print(t_C)
            tmp_data = self.D * t_C[:, indexOfData]
            E = self.X[:, indexOfData] - tmp_data
        else:
            E = self.X[:, indexOfData] - self.D * self.A[:, indexOfData] + self.D[:, indexOfAtom] * self.A[indexOfAtom, indexOfData]
#        print("E", E)
        u, s, vh = np.linalg.svd(E) # need trans? 看E的變化應該選這個
#        v = vh.T
#        print("u:", u, "s:", s, "vh:", vh, sep = "\n")
#        print("->new d"+str(indexOfAtom), u[:, 0].T)
        self.D[:, indexOfAtom] = u[:, 0].copy()
#        print("new d"+str(indexOfAtom), u[0, :])
        newCoe = s[0] * vh[:, 0].T
#        print("new coe", newCoe)
        self.A[indexOfAtom, indexOfData] = newCoe
        return
    
    def UpdateDict_FLOW(self):
        usedAtomIndex, userX = np.where(self.A!=0)
#        print(usedAtomIndex)
        for indexOfAtom in range(self.D.shape[1]):
#            print("\nindexOfAtom:", indexOfAtom)#, "=> d"+str((indexOfAtom)))
            tmp = np.where(usedAtomIndex == indexOfAtom)[0]
            if tmp.shape[0] == 0:
                continue
#            print(np.where(usedAtomIndex == indexOfAtom))
#            print("=>", userX[tmp])
            self.UpdateDict_useSVD_one(indexOfAtom, userX[tmp])
#            break
        return self.D
#%%
#if __name__ == "__main__":
# 讀取、顯示
initailDict = np.load("dct_64_256.npy")
noiseImgName = "noise_barbara_var0.004.jpg"
originalImgName = "original_128x128_barbara.jpg"
noiseImg = cv2.imread(noiseImgName, cv2.IMREAD_GRAYSCALE)
#originalImg = cv2.imread(originalImgName)
#cv2.imshow("noise image", noiseImg)
#cv2.imshow("original image", originalImg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#%% 切片
_rows_patch, _cols_patch = 8,8
_rows, _cols = noiseImg.shape[:2]
X_img = np.zeros((_cols_patch * _rows_patch, (_cols - _cols_patch) * (_rows - _rows_patch)))

count = 0 # i*j+j 一直算不好
for _i in range(_cols - _cols_patch):
    for _j in range(_rows - _rows_patch):
        tmpImg =  noiseImg[_i:_i+_cols_patch, _j:_j+_rows_patch].flatten(order="F")
#        print(i*j+j, tmpImg.shape)
        X_img[:, count] = tmpImg
        count += 1
#%%
D = initailDict.copy()
# 
omp = OMP(D, X_img, 3)
A = omp.Flow_all_OMP()
svd = SVD(D, X_img)
svd.SetAlpha(A)
D = svd.UpdateDict_FLOW()


_endTime = time.time()
print("\n\nIt cost", round(_endTime - _startTime, 4), "sec.")
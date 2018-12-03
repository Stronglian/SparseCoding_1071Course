import numpy as np
from scipy.spatial import distance

"""
    d1 = [0, -0.2788, 0.9604]
    d2 = [-0.8837, 0.2019, 0.4223]
    d3 = [0.7837, 0.6196, 0.0432]
    d4 = [0.7355, 0.6180, -0.2777]
    d5 = [-0.2433, -0.1908, -0.9510]
    d6 = [-0.5711, 0.2891, 0.7683]
    d7 = [0.8960, -0.2179, 0.3869]
    d8 = [0.5046, 0.8941, -0.6265]
    d9 = [0.2006, 0.2251, -0.9535]
    d10 = [-0.5627, 0.8207, -0.0997]


    x1 = [-0.9471, -0.3744, -1.1859]
    x2 = [-1.0559, 1.4725, 0.0557]
    x3 = [-1.2173, -0.0412, -1.1283]
    x4 = [-1.3493, -0.2611, 0.9535]
    x5 = [0.1286, 0.6565, -1.1678]
    x6 = [-0.4606, -0.2624, -1.2132]
"""
d1 = [0, -0.2788, 0.9604]
d2 = [-0.8837, 0.2019, 0.4223]
d3 = [0.7837, 0.6196, 0.0432]
d4 = [0.7355, 0.6180, -0.2777]
d5 = [-0.2433, -0.1908, -0.9510]
d6 = [-0.5711, 0.2891, 0.7683]
d7 = [0.8960, -0.2179, 0.3869]
d8 = [0.5046, 0.8941, -0.6265]
d9 = [0.2006, 0.2251, -0.9535]
d10 = [-0.5627, 0.8207, -0.0997]


x1 = [-0.9471, -0.3744, -1.1859]
x2 = [-1.0559, 1.4725, 0.0557]
x3 = [-1.2173, -0.0412, -1.1283]
x4 = [-1.3493, -0.2611, 0.9535]
x5 = [0.1286, 0.6565, -1.1678]
x6 = [-0.4606, -0.2624, -1.2132]

D = np.array([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10])     # (10, 3)
X = np.array([x1, x2, x3, x4, x5, x6])                      # (6, 3)
L = 3
#D = np.array([[0.5774, 0.5774, 0.5774], [-0.4083, -0.4083, 0.8165], [0.7071, 0.7071, 0.]])
#X = np.array([[0.5, 0.3, 0.1], [0.8, -0.1, 2.], [1., 2., -1.5]])
#L = 2

cofs = {}

for i, sample in enumerate(X):
    residual = sample.copy()
    a = None
    indices = []
    cof = np.zeros((len(D), 1))

    for time in range(L):
        corr = np.dot(residual, D.T)
#        index = np.argmax(corr)
        index = np.argmax(np.abs(corr)) #<==
        indices.append(int(index))
#        residual = sample - np.max(corr) * D[index]
        residual = sample - corr[index] * D[index] #<==
        a = np.linalg.pinv(
                np.matrix(D.T[:, indices].T) * np.matrix(D.T[:, indices])) * \
                np.matrix(D.T[:, indices].T) * np.matrix(sample).T    
    print(indices)
    cof[indices] += a
    cofs["x" + str(i + 1)] = list(np.round(np.squeeze(cof.T), 4))

for key, val in cofs.items():
    print("{} coefficient is ".format(key), end="[  ")
    for c in val:
        print("{:.4f}".format(c), end="  ")
    print("]")
# 在层次分析法和Topsis法中，权重的获得都是主观得到的，用熵权法就可以更客观地得到权重
# 依据的原理就是指标的变异程度越小，反映的信息量越少，权值越低（信息熵越小，指标离散程度越大，影响越大）
# 基本步骤：1.矩阵正向化,数据标准化，与Topsis法相同，但是如果存在负数，Z(ij)=x-min{x(1j),x(2j),...x(nj)}/max{x(1j)...}-min{x(1j)...}
# 2.计算概率矩阵P:第j项指标下第i个样本所占比重
# 3.计算熵权e(j)=-1/(ln n)∑(i=1->n)p(ij)ln(p(ij)) d(j)=1-e(j) W(j)=d(j)/∑(j=1->m)d(j)
import numpy as np
# 矩阵正向化，矩阵标准化

X = np.array([[9, 0, 0, 0], [8, 3, 0.9, 0.5], [6, 7, 0.2, 1]])
Z = X / np.sqrt(np.sum(X*X, axis=0))

# 计算概率矩阵,第j项指标在第i个样本下的比重,p(ij)=z(ij)/∑(i=1->n)z(ij)
# 计算熵权,由公式易知，p(1j)=p(2j)=...=p(nj)=1/n时，e(j)=1，信息熵最大，信息效用值最小
# 定义信息效用值d(j)=1-e(j),归一化得到熵权

def mylog(p):
    n = len(p)
    lnp = np.zeros(n)
    for i in range(n):
        if p[i] == 0:
            lnp[i] = 0
        else:
            lnp[i] = np.log(p[i])
    return lnp

n,m = Z.shape
D = np.zeros(m) #每个指标的效用值
for i in range(m):
    x = Z[:, i]
    p = x / np.sum(x)
    #使用自定义mylog函数计算p的对数，如果p中含有0,直接用np.log会得到-inf,用自定义log避免问题
    e = -np.sum(p * mylog(p)) / np.log(n)
    D[i] = 1 - e

W = D / np.sum(D)
print(W)
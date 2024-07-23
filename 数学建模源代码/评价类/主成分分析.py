#主成分分析是将原来的变量重新组合成一组新的相互无关的综合变量，从中取出几个少的综合变量来反映原来变量的信息，是一种降维方法
#降维的几何意义是旋转坐标轴。代数意义可以理解成m*n阶原始样本X，与n*k阶矩阵W乘法得到m*k阶降维矩阵Y
#基本步骤，假设有n个样本，p个指标组成的样本矩阵x:
#1.标准化处理,按列算均值xj和标准差Sj，得到标准化数据X(ij)=(x(ij)-xj)/Sj,标准矩阵变为X
#2.计算标准化样本的协方差矩阵/样本相关系数矩阵R
#r(ij)=1/(n-1)*∑(k=1->n)(X(ki)-Xi(列均))(X(kj)-Xj(列均))
#3.计算协方差矩阵R的特征值(λ1>=λ2>=λ3...>=λp>=0)和特征向量(a1,a2,a3...ap)
#4.计算主成分贡献率和累计贡献率，贡献率α(i)=λ(i)/∑(k=1->p)λ(k),累计贡献率就是前缀和
#5.写出主成分，累计贡献率超过80%的作为主成分
#6.解释主成分，指标前面的系数越大，代表该指标对于该主成分的影响越大
import pandas as pd
import numpy as np
from scipy import linalg #线性代数

#标准化
df = pd.read_excel('.xlsx', usecols='C:G') #从C列到G列读取excel
x = df.to_numpy() #转化为numpy数组
X = (x - np.mean(x, axis=0))/np.std(x, ddof=1, axis=0) #std计算标准差

#计算协方差矩阵
R = np.cov(X.T) #X.T转置矩阵

#计算特征值和特征向量
eigenvalues,eigenvectors = linalg.eigh(R) #eigh函数得到的特征值特征向量是从小到大排列的
eigenvalues = eigenvalues[::-1] #::-1代表反向，降序排列
eigenvectors = eigenvectors[:,::-1]

#计算贡献率和累计贡献率
contribution_rate = eigenvalues / sum(eigenvalues)
cum_contribution_rate = np.cumsum(contribution_rate) #cumsum计算前缀和

print("特征值为")
print(eigenvalues)
print("贡献率为")
print(contribution_rate)
print("累计贡献率")
print(cum_contribution_rate)
print("特征矩阵是")
print(eigenvectors)
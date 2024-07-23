#灰色系统理论，信息不完全系统就是灰色系统
#关联分析：系统地分析因素，某个包含多个因素的系统中，那些因素是主要的，那些是次要的，哪些影响大，哪些影响小等
#灰色关联分析，是一种多因素统计分析方法，它对样本量多少和样本有无规律都同样适用，根据序列曲线的几何形状的相似程度来判断关系是否紧密
#基本步骤：1.母序列，反映系统行为特征的数据序列，类似因变量Y=[y1,y2,...,yn]^T
#2.子序列，影响系统行为的因素组成的数据序列，类似自变量X=[x(11),x(12)...,x(1m),x(21).....]
#3.数据预处理，求出每个指标均值，用指标中元素除以均值
#4.计算灰色关联系数：a=min min |x(0k)-x(ik)|,b=max max|x(0k)-x(ik)|,为两极最小差和两级最大差，构造ξ(k)=(a+ρb)/(|x(0k)-x(ik)|+ρb),ρ一般取0.5
#5.计算关联度,r=∑(k=1->n)ξ(k)/n
#例题：给明星K选对象，A,B,C三位候选人，身高165最好，体重90-100最好
#候选人    颜值     脾气(争吵次数)      身高      体重
#A        9       10                175      120
#B        8       7                 164      80
#C        6       3                 157      90
#正向化得到
#9 0  0   0
#8 3 0.9 0.5
#6 7 0.2  1
import numpy as np

n = eval(input("输入参评数目"))
m = eval(input("接受指标数目"))
A = np.zeros(shape=(n, m))
print("输入正向化矩阵")
for i in range(n):
    A[i] = input().split(" ")
    A[i] = list(map(float, A[i]))

#数据预处理，每个指标元素除以该指标平均值

Mean = np.mean(A,axis=0) #求出每列均值
A_norm = A / Mean
#预处理得到：
#1.17  0    0    0
#1.04 0.90 2.45 1.00
#0.78 2.10 0.55 2.00

#母序列
#评价决策问题没有明显因变量，要构造母序列，例子中构造的母序列是每个对象的所有参数中的最大值Y=【1.17 2.45 2.10】
Y = np.max(A_norm, axis=1)

#子序列
X = A_norm

#计算|X0-Xi|矩阵，计算灰色关联系数
absX0_Xi = np.abs(X-np.tile(Y.reshape(-1,1),(1, X.shape[1])))
a = np.min(absX0_Xi) #两级最小差
b = np.max(absX0_Xi)
rho = 0.5 #分辨系数

#计算灰色关联度
gamma = (a + rho * b)/(absX0_Xi + rho * b)
print("灰色关联度:")
print(np.mean(gamma, axis=0))

#根据灰色关联度算权重
weight = np.mean(gamma, axis=0)/np.sum(np.mean(gamma, axis=0))
score = np.sum(X * np.tile(weight, (X.shape[0], 1)), axis=1)
stand_S = score / np.sum(score) #归一化得分
print("得分：")
print(stand_S)
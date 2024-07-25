# https://blog.csdn.net/Westbrook_bo/article/details/130276549?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172190296416800186565755%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=172190296416800186565755&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-3-130276549-null-null.142^v100^pc_search_result_base1&utm_term=GM%281%2C1%29%20python&spm=1018.2226.3001.4187
# https://blog.csdn.net/weixin_51009494/article/details/126384490?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172190293216800186545703%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172190293216800186545703&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-126384490-null-null.142^v100^pc_search_result_base1&utm_term=GM%281%2C1%29&spm=1018.2226.3001.4187
# 灰色系统预测
# 对在一定范围内，与时间有关的灰色过程进行预测
# 适用情况:
# (1)以年份为度量的非负数据(月份或者季度要用时间序列模型)
# (2)数据能经过准指数规律的检验(除了前两期，后面至少90%的期数的光滑比要低于0.5)
# (3)数据期数较短，而且和其他数据之间关联性不强(小于等于10)
#
# GM(1,1)模型:只有一个变量的一阶微分方程模型
#
# 为了确定原始数据是否能进行灰色模型预测，需要级比检验（建模前）
# 累加r次的序列为x^(r),定义级比σ(k)=x^(r)(k)/x^(r)(k-1),k=2.3...n，对于任意k,σ(k)∈[a,b],且b-a<0.5，则序列有准指数规律
# 对于GM(1,1), r=1,序列x^(1)的级比σ(k)= x^(0)(k)/x^(1)(k-1) +1,定义ρ(k)= x^(0)(k)/ x^(1)( k-1)为光滑比，只需要保证任意k,ρ(k)∈(0,0.5)占比越高越好
#
# 级比偏差检验
# 计算原始级比σ(k)= x^(0)(k)/ x^(0)(k-1)
# 根据预测发展系数(-α^)计算级比偏差和平均级比偏差
# η(k)=|1- ((1-0.5*α^)/(1+0.5*α^))*1/σ(k)|
# 平均值=Σ(k=2→n)η(k)/(n-1)
# 如果平均值<0.2，达到要求，<0.1非常不错
#
# 如何用GM(1,1)预测:
# 1.根据原始的离散非负数据列，通过累加等方式削弱随机性，获得有规律离散数据列
# 2.建立相应微分方程模型，得到离散点处解
# 3.通过累减获得原始数据的估计值，而对原始数据预测
#
# 对数据x^(0)进行累加得到新序列x^(1)
# 利用常微分方程
# dx^(1)/dt + a*x^(1) = u
# 对于离散数据 dx就等于△x
# 所以微分方程可以化为x^(0)(t)+a*x^(1)(t)=u
# 为了消除数据随机性，定义z^(1)=(z1, z2...)
# 其中z^(1)(m)=δ*x^(1)(m)+(1-δ)*x^(1)*(m-1)，且δ=0.5，既为前后时刻均值
# 微分方程改为x^(0)(t)=-a*z^(1)(t)+u，利用线性回归求解，最后得到的a和u带入原方程求解
#
# 模型评价:
# 绝对残差ε(k)= x^(0)(k)-x拔^(0)(k)
# 相对残差ε_r(k)=|ε(k)|/ x^(0)(k)*100%
# 相对平均残差1/( n-1) * Σ(k=2→n)|ε_r(k)|
# 如果相对平均残差<20%，拟合达到一般要求,<10%,拟合效果不错
import matplotlib.pyplot  as plt
import numpy as np
import math as mt
# 构建序列进行光滑比检验
X0 = np.array([88898,97242,97990,93363,88093,103852,105475,114183,133762,162214,173977,200973])
X1 = X0.cumsum()
rho = [X0[i] / X1[i - 1] for i in range(1,len(X0))]
rho_ratio = [rho[i + 1] / rho[i] for i in range(len(rho) - 1)]
print("rho:",rho)
print("rho_ratio:",rho_ratio)
flag = True
for i in range(1,len(rho) - 1):
    if rho[i] > 0.5 or rho[i + 1] / rho[i] >= 1:
        flag = False
if rho[-1] > 0.5:
    flag = False
if flag:
    print("数据通过光滑校验")
else:
    print("该数据未通过光滑校验")

# 级比检验
for i in range(len(X0)-1):
    l = X0[i]/X0[i+1]
    if l <= mt.exp(-2/(len(X0)+1)) or l >= mt.exp(2/(len(X0)+1)):
        break
    else:
        pass
if i == len(X0)-2 and l > mt.exp(-2/(len(X0)+1)) and l < mt.exp(2/(len(X0)+1)):
    print('级比检验通过')
else:
    print('级比检验不通过')

# 检验不通过处理

j = 1
while True:
    YO = [k+j for k in X0]
    j += 1
    for m in range(len(YO) - 1):
        l = YO[m] / YO[m + 1]
        if l > mt.exp(-2 / (len(X0) + 1)) and l < mt.exp(2 / (len(X0) + 1)):
            b = True
        else:
            b = False
            break
    if b == True:
        print("新的原始数列为：",YO)
        c = j -1
        print("c的值为：",c)
        break
    else:
        continue

# 构造模型求解


X0 = ['数据']
# 累加数列
X1 = [X0[0]]
add = X0[0] + X0[1]
X1.append(add)
i = 2
while i < len(X0):
    add = add + X0[i]
    X1.append(add)
    i += 1

# 紧邻均值序列
Z = []
j = 1
while j < len(X1):
    num = (X1[j] + X1[j - 1]) / 2
    Z.append(num)
    j = j + 1

# 最小二乘法计算
Y = []
x_i = 0
while x_i < len(X0) - 1:
    x_i += 1
    Y.append(X0[x_i])
Y = np.mat(Y)
Y = Y.reshape(-1, 1)
B = []
b = 0
while b < len(Z):
    B.append(-Z[b])
    b += 1
B = np.mat(B)
B = B.reshape(-1, 1)
c = np.ones((len(B), 1))
B = np.hstack((B, c))
print("B", B)

# 求出参数
alpha = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
a = alpha[0, 0]
b = alpha[1, 0]
print('alpha', alpha)
print("a=", a)
print("b=", b)

# 生成预测模型
GM = []
GM.append(X0[0])
did = b / a
k = 1
while k < len(X0):
    GM.append((X0[0] - did) * mt.exp(-a * k) + did)
    k += 1

# 做差得到预测序列
G = []
G.append(X0[0])
g = 1
while g < len(X0):
    G.append(round(GM[g] - GM[g - 1]))
    g += 1
print("预测数列为：", G)

# 绘图

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

X0 = ['原始数据']
G = ['预测数据']
G = [g-37315 for g in G]
r = range(len(X0))
t = list(r)
plt.plot(t,X0,color='r',linestyle="--",label='true')
plt.plot(t,G,color='b',linestyle="--",label="predict")
plt.legend()
plt.show()

# 检验

X0 = ['原始数据']
G = ['预测数据']
X0 = np.array(X0)
G = np.array(G)
e = X0 - G  #残差
q = e / X0  # 相对误差
w = 0
for q_i in q:
    w += q_i
w = w/len(X0)
print('精度为{}%'.format(round((1-w)*100,2)))

s0 = np.var(X0)
s1 = np.var(e)
S0 = mt.sqrt(s0)
S1 = mt.sqrt(s1)
C = S1 / S0
print('方差比为：',C)

p = 0
for s in range(len(e)):
    if (abs(e[s]-np.mean(e)) < 0.6745 * S1):
        p = p + 1
P = p / len(e)
print('小概率误差为：',P)

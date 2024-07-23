# Topsis法，简称优劣解距离法，引入两个基本概念，理想解（设想的最优解，所有属性都是最佳）负理想解（所有属性都最劣），通过与这两个解的距离来判断最优选择
# 在多目标决策分析中是一种很有效的方法，层次分析法的决策层不多，而且判断矩阵过于主观
# 基本步骤：1.将原始矩阵正向化，将所有指标转成极大型指标
# 2.正向矩阵标准化，有很多种标准化方法，目的是去除量纲的影响，且数据大小排序不变，标准化之后要给个指标加权重
# 3.计算得分并归一化，S(i)=D(i)^- / (D(i)^+ + D(i)^-),其中S(i)为得分，D(i)^+为与最大值距离，D(i)^-为与最小值距离

import numpy as np

print("请输入参评数目")
n = int(input())
print("输入指标数目")
m = int(input())

print("输入类型:1.极大型，2.极小型，3.中间型，4.区间型")
kind = input().split() #将输入字符串按空格分割，形成列表

print("输入矩阵:")
A = np.zeros(shape=(n,m)) # nm全零矩阵
for i in range(n):  #每行接收一次,range(start,stop,step)
    A[i] = input().split(" ")
    A[i] = list(map(float,A[i]))    #接收到的字符串列表转换成浮点数列表
print("输入矩阵为:\n{}",format(A))

# 原始矩阵正向化
# 极小型指标（越小越好） y(i) = max-x(i),max为指标最大值
# 中间型指标（中间最好） M = max{|x(i)-x(best)|},y(i)=1-|x(i)-x(best)|/M,x(best)是最优值
# 区间型指标（最佳区间是[a,b]），M=max{a-min{x(i)},max{x(i)}-b},y=1-(a-x(i))/M,x<a;y=1,a<x<b;y=1-(x-b)/M,x>b

def minTomax(maxx, x):
    x = list(x)
    ans = [[(maxx-e)] for e in x] #列表推导式，用于计算每个值与最大值的差，放入新列表
    return np.array(ans)

def midTomax(bestx, x):
    x = list(x)
    h = [abs(e-bestx) for e in x]
    M = max(h)
    if M == 0:
        M = 1
    ans = [[(1-e/M)] for e in h]
    return np.array(ans)

def regTomax(lowx, highx, x):
    x = list(x)
    M = max(lowx-min(x), max(x)-highx)
    if M == 0:
        M = 1
    ans = []
    for i in range(len(x)):
        if x[i] < lowx:
            ans.append([(1 - (lowx - x[i]) / M)])
        elif x[i] > highx:
            ans.append([(1 - (x[i] - highx) / M)])
        else:
            ans.append([1])
    return np.array(ans)

X = np.zeros(shape=(n, 1))
for i in range(m):
    if kind[i] == "1":
        v = np.array(A[:,i])
    elif kind[i] == "2":
        maxA = max(A[:, i])
        v = minTomax(maxA, A[:, i])
    elif kind[i] == "3":
        print("输入最优值：")
        bestA = eval(input()) #eval执行字符串表达式，并返回值
        v = midTomax(bestA, A[:, i])
    elif kind[i] == "4":
        print("输入区间")
        lowA = eval(input())
        highA = eval(input())
        v = regTomax(lowA, highA, A[:,i])
    if i == 0:
        X = v.reshape(-1,1) #如果是第一个指标，变换成一列型的，直接替换X数组,-1表示自动识别行数
    else:
        X = np.hstack([X, v.reshape(-1, 1)]) #如果不是第一个指标，拼接到X数组上，hstack表示沿着列堆叠数组

# 正向化矩阵标准化
# 及标准化矩阵为Z，z(ij)=每一个元素/√(所在列元素平方和)
# 标准化之后，还要给指标加上权重，权重确定方法很多，有层次分析法，熵权法，Delphi法，对数最小二乘法

X = X.astype('float') #数据类型转为float
for j in range(m):
    X[:, j] = X[:, j]/np.sqrt(sum(X[:, j])**2)

# 计算得分
# 定义最大值Z^+ = 每一列最大值的向量
# 定义最小值Z^- = 每一列最小值的向量
# 则第i个评价对象到最大值的距离为D(i)^+ = √∑(j=1->m)(Z(j)^+ - z(ij))^2 (每一项与该项的最大值的距离）
# 同理的D(i)^-，最后得到每项距离S(i)=D(i)^-/(D(i)^+ + D(i)^-)
# 得到S(i)越大，D(i)^+越小，越接近最大值，最后将得分归一化并转化百分制

x_max = np.max(X, axis=0) #每列最大值
x_min = np.min(X, axis=0)
#计算每个参评对象与最优值距离,square表示每个元素都平方,tile(arr,reps)其中arr表示输入的数组，reps表示返回的数组的形状，用于重复序列
d_z = np.sqrt(np.sum(np.square((X - np.tile(x_max,(n,1)))), axis=1))
d_f = np.sqrt(np.sum(np.square((X - np.tile(x_min,(n,1)))), axis=1))
s = d_f/(d_z + d_f)
Score = 100*s/sum(s)
for i in range(len(Score)):
    print(f"第{i+1}个对象标准化后百分制得分为:{Score[i]}")
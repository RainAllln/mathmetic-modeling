#蒙特卡罗法，称统计模拟法，是使用随机数来解决计算问题的方法，求解的问题和概率模型有关联，就可以用蒙特卡罗法，没有通用算法
#例子：计算圆周率
import numpy as np
import matplotlib.pyplot as plt

p = 10000 #投放10000个点
r = 1 #圆的半径
x0,y0 = 1, 1 #圆心
n = 0 #有n个点在园内

plt.figure() #绘图窗口
plt.title("Monte Carlo calculate PI")
plt.xlabel('x')
plt.ylabel('y')

for i in range(p):
    px = np.random.rand() * 2 #np.random.rand()产生(0,1)随机数
    py = np.random.rand() * 2

    if (px-x0)**2 + (py-y0)**2 < r**2:
        plt.plot(px,py,'.', color='b') #圈内点用蓝色
        n+=1
    else:
        plt.plot(px,py,'.',color='r') #圆外点用红色

plt.axis('equal') #绘图时横纵坐标长度相同，便于观察
plt.show()

PI = ( n / p) * 4
print("近似为：",PI)
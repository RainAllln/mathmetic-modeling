# 模型假设：1.x(t)表示t时刻人口数,且x(t)连续可微
# 2.人口的增长率r是常数（增长率-死亡率）
# 3.人口变化是封闭的（没有流动人口）
# 方程:{dx/dt=rx,x(0)=x_0  解得到x(t)=x_0*e^n

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def malthusiam_model(P,t,r):
    dPdt = r*P
    return dPdt
#初始人口
P0=100
t = np.linspace(0,1000,1000)
r=0.001
P=odeint(malthusiam_model,P0,t,args=(r,))
plt.plot(t,P)
plt.show()

# 其中发现模型与现实不符，r不应该是常数
# 修正r为人口x(t)的函数r(x),且r(x)为x的减函数
# 得到阻滞增长模型
# 模型假设：
# 1.r(x)为线性，r(x)=r-sx
# 2.最大容纳量为x_m

# 定义阻滞增长模型的微分方程
def logistic_growth_model(P, t, r, K):
    dPdt=r*P*(1 -P/K)
    return dPdt
# 定义初始条件和参数
P0=100 #初始人口数量
t=np.linspace(0,1000,1000) #时间范围
r=0.04 #人口增长率
K=1000 #环境容量
# 求解微分方程
P=odeint(logistic_growth_model, P0, t, args=(r,K))
# 绘制人口随时间变化的图像
plt.plot(t, P)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Logistic Growth Model')
plt.show()
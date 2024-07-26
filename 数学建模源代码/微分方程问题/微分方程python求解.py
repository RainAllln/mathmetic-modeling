# odeint函数使用
# func: 微分方程函数或方程组。它必须是一个函数，函数接受两个参数，第一个参数是一个数组，表示当前的因变量值;第二个参数是一个标量，表示当前的自变量值(通常是时间)。
# y0:数组类型，表示初始条件。
# t:数组类型，表示积分的时间点，第个元素必须是初始时间。
from scipy.integrate import odeint  # 微分方程函数
import numpy as np


def model(y, t):
    k = 0.3
    dydt = -k * y
    return dydt  # 初始条件,y'=-0.3*y


y0 = 5
# 时间点
t = np.linspace(0, 20, 100)  # 0-20选100个点
# 求解ODE
result = odeint(model, y0, t)
# 输出结果
print(result)


#solve_ivp函数，更强大的功能
# fun: 微分方程函数，与odeint中的func相似，但它的第一个参数是标量(当前的自变量值)，第二个参数是数组(当前的因变量值)
# t_span:二元组类型，表示积分的时间区间(起始时间和结束时间)
# y0:数组类型，表示初始条件。
# method:字符串类型(可选)，积分方法，例如'RK45'(默认)，'RK23'，'DOP853'，'BDF'等。
from scipy.integrate import solve_ivp
#微分方程函数
def model(t, y):
    k=0.3
    dydt = -k* y
    return dydt
#初始条件
y0 =[5]
#时间区间
t_span =(0,20)
#求解ODE
sol= solve_ivp(model, t_span, y0,t_eval=np.linspace(0,20,100))
#输出结果
print(sol.y)
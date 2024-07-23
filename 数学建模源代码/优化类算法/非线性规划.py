#常见经济增长率，传播率等问题用到1/x形式；空间运动问题，比如避免碰撞，角度调整等，用到三角公式；还有运输类问题，比如已知坐标，运输物品，用到距离公式。
#非线性规划是一种求解的目标函数或者约束函数里面有一个或几个非线性函数的最优化问题
#模型中至少一个变量是非线性，x^2,e^x,1/x,sinx,logx等形式，没有通用解法，求近似解即可
#标准型为min f(x) s.t.={A*x<=b;A_eq*x=b_eq;(线性)c(x)<=0;Ceq(x)=0;(非线性)lb<=x<+ub
#非线性规划对于初始值x0的选取非常重要，一般来说非线性规划取得局部最优解，如果要求全局最优解，有：（1)给定不同的初始值，从中得到最优解（2）蒙特卡罗法得x0
#https://blog.csdn.net/diudiu_aaa/article/details/132635694
import scipy.optimize
from scipy.optimize import brent, fmin_ncg, minimize
import numpy as np

#brent()是求解单变量无约束优化问题最小值的首选方法。这是牛顿法和二分法的混合方法，既能保证稳定性又能快速收敛。
#scipy.optimize.brent(func, args=(), brack=None, tol=1.48e-08, full_output=0, maxiter=500)
# 1. Demo1：单变量无约束优化问题(Scipy.optimize.brent)
def objf(x):  # 目标函数
    fx = x ** 2 - 8 * np.sin(2 * x + np.pi)
    return fx

xIni = -5.0
xOpt = brent(objf, brack=(xIni, 2))
print("xIni={:.4f}\tfxIni={:.4f}".format(xIni, objf(xIni)))
print("xOpt={:.4f}\tfxOpt={:.4f}".format(xOpt, objf(xOpt)))

#fmin() 函数是 SciPy.optimize 模块中求解多变量无约束优化问题（最小值）的首选方法，采用下山单纯性方法。
#下山单纯性方法又称 Nelder-Mead 法，只使用目标函数值，不需要导数或二阶导数值，是最重要的多维无约束优化问题数值方法之一。
#scipy.optimize.fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)
def objf2(x):  # Rosenbrock benchmark function
    fx = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return fx

xIni = np.array([-2, -2])
xOpt = scipy.optimize.fmin(objf2, xIni)
print("xIni={:.4f},{:.4f}\tfxIni={:.4f}".format(xIni[0], xIni[1], objf2(xIni)))

#minimize() 函数是 SciPy.optimize 模块中求解多变量优化问题的通用方法，可以调用多种算法，支持约束优化和无约束优化。
#scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

# 定义目标函数
def objf3(x):  # Rosenbrock 测试函数
    fx = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return fx

# 定义边界约束（优化变量的上下限）
b0 = (0.0, None)  # 0.0 <= x[0] <= Inf
b1 = (0.0, 10.0)  # 0.0 <= x[1] <= 10.0
b2 = (-5.0, 100.)  # -5.0 <= x[2] <= 100.0
bnds = (b0, b1, b2)  # 边界约束

# 优化计算
xIni = np.array([1., 2., 3.])
resRosen = minimize(objf3, xIni, method='SLSQP', bounds=bnds)
xOpt = resRosen.x

print("xOpt = {:.4f}, {:.4f}, {:.4f}".format(xOpt[0], xOpt[1], xOpt[2]))
print("min f(x) = {:.4f}".format(objf3(xOpt)))

#例题：
#目标函数:min f(x)=a*x1^2+b*x2^2+c*x3^2+d
#s.t.={x1^2-x2+x3^3>=0;x1+x2^2+x3^2<=20;-x1-x2^2+2=0;x2+2*x3^2=3;x1,x2,x3>=0
#转成标准形式{x1^2-x2+x3^2>=0;-(x1+x2^2+x3^2-20)>=0;-x1-x2^2+2=0;x2+2*x3^2-3=0;x1,x2,x3>=0


def objF4(x):  # 定义目标函数
    a, b, c, d = 1, 2, 3, 8
    fx = a*x[0]**2 + b*x[1]**2 + c*x[2]**2 + d
    return fx


# def objF5(args):  #比上面的目标函数更优化，args作为参数
#     a,b,c,d = args
#     fx = lambda x: a*x[0]**2 + b*x[1]**2 + c*x[2]**2 + d
#     return fx

# 定义约束条件函数

def constraint1(x):  # 不等式约束 f(x)>=0
    return x[0]** 2 - x[1] + x[2]**2
def constraint2(x):  # 不等式约束 转换为标准形式
    return -(x[0] + x[1]**2 + x[2]**3 - 20)
def constraint3(x):  # 等式约束
    return -x[0] - x[1]**2 + 2
def constraint4(x):  # 等式约束
    return x[1] + 2*x[2]**2 -3

# def constraint1():  # 定义约束条件函数,更优化版本
#     cons = ({'type': 'ineq', 'fun': lambda x: (x[0]**2 - x[1] + x[2]**2)},  # 不等式约束 f(x)>=0
#             {'type': 'ineq', 'fun': lambda x: -(x[0] + x[1]**2 + x[2]**3 - 20)},  # 不等式约束 转换为标准形式
#             {'type': 'eq', 'fun': lambda x: (-x[0] - x[1]**2 + 2)},  # 等式约束
#             {'type': 'eq', 'fun': lambda x: (x[1] + 2*x[2]**2 - 3)})  # 等式约束
#     return cons
# cons = constraint1()
# args1 = (1,2,3,8)  # 定义目标函数中的参数

# 定义边界约束
b = (0.0, None)
bnds = (b, b, b)

# 定义约束条件
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': constraint3}
con4 = {'type': 'eq', 'fun': constraint4}
cons = ([con1, con2, con3,con4])  # 3个约束条

# 求解优化问题
x0 = np.array([1., 2., 3.])  # 定义搜索的初值
res = minimize(objF4, x0, method='SLSQP', bounds=bnds, constraints=cons)
print("Optimization problem (res):\t{}".format(res.message))  # 优化是否成功
print("xOpt = {}".format(res.x))  # 自变量的优化值
print("min f(x) = {:.4f}".format(res.fun))  # 目标函数的优化值
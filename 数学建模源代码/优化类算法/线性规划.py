#线性规划解决问题，在有限的资源下，得到最优解
#线性规划三要素，1.决策变量，要确定的未知量,2.目标函数,决策变量的函数，3.约束条件，决策变量的约束和限制，用含有决策变量的等式或不等式表示
#基本步骤：1.找到决策变量，2.找到目标函数，3.找出约束条件
#矩阵表现形式，max(min)z=c^T*x
#s.t.={Ax<=(或者=,>=)b,x>=0
#使用linprog函数，result = linprog(c,A_ub,b_ub,A_eq,b_eq,bounds,method)
#其中，min cx,s.t.={A_ub * x <=b_ub,A_eq * x = b_eq,x∈bounds
#c是目标函数决策变量对应的系数向量，行列都行，A_ub,B_ub是不等式约束条件，必须是小于等于，A_eq,b_eq是等式约束，bounds表示决策变量定义域，None表示无穷
#result有多个参数，x是最优解,fun为最小值，nit迭代次数,一般调用result.x

#例题：明星KK喜欢玩游戏，想要通过一种最快的方式升级
#游戏有100体力，反复通关ABC三张地图升级，A图20经验，消耗4体力，B图30经验，消耗8体力，C图45经验，消耗15体力，同时ABC加在一起最多通关20次
#决策变量：设通关三张地图的次数分别是x1,x2,x3，
#目标函数:max y = 20*x1 + 30*x2 + 45*x3
#约束函数 4*x1 + 8*x2 + 15*x3 <=100;x1+x2+x3<=20;x1,x2,x3>=0
from scipy.optimize import linprog

c = [-20, -30, -45] #标准形式是min,这里是max，取反
A_ub = [[4, 8, 15], [1, 1, 1]]
b_ub = [100, 20]
bounds = [[0,None], [0, None], [0, None]]
result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print(result)
print("三图通关次数分别为:")
print(result.x)
print("最终获得经验:")
print(-result.fun)
#此例题是整数线性规划，严格来说并不能用这个方法求解，只是在此处做一个演示
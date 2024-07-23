#在规划问题中，某些问题的解必须是整数，如果所有的变量都限制是整数，那就是纯整数规划，一部分就是混合整数规划，特殊形式是01规划
#线性整数规划可以求解，非线性整数规划只能用近似算法，比如蒙特卡罗和智能算法

#例子1：min z=3*x1+4*x2+x3,s.t.={x1+6*x2+2*x3>=5;2x1>=3;x1,x2,x3>=0且都为整数
import pulp
# 参数设置
c = [3,4,1]        #目标函数未知数前的系数
A_gq = [[1,6,2],[2,0,0]]   # 大于等于式子 未知数前的系数集合 二维数组
b_gq = [5,3]         # 大于等于式子右边的数值 一维数组
# 确定最大最小化问题，当前确定的是最小化问题
m = pulp.LpProblem(sense=pulp.LpMinimize)
# 定义三个变量放到列表中 生成x1 x2 x3
x = [pulp.LpVariable(f'x{i}',lowBound=0,cat='Integer') for i in [1,2,3]]
# 定义目标函数，并将目标函数加入求解的问题中
m += pulp.lpDot(c,x) # lpDot 用于计算点积
# 设置比较条件
for i in range(len(A_gq)):# 大于等于
    m += (pulp.lpDot(A_gq[i],x) >= b_gq[i])
# 求解
m.solve()
# 输出结果
print(f'优化结果：{pulp.value(m.objective)}')
print(f'参数取值：{[pulp.value(var) for var in x]}')

#例子2

# 创建一个线性整数规划问题实例
lp = pulp.LpProblem("Integer Linear Programming", pulp.LpMinimize)

# 定义决策变量，设置为整数类型
x1 = pulp.LpVariable('x1', cat='Integer')
x2 = pulp.LpVariable('x2', cat='Integer')

# 设置目标函数
lp += x1 + 2 * x2, "Z"

# 设置约束条件
lp += 0 <= x1 <= 5, "Constraint 1"
lp += 0 <= x2 <= 5, "Constraint 2"
lp += x1 + x2 >= 1, "Constraint 3"

# 求解问题
lp.solve()

# 输出结果
for variable in lp.variables():
    print(f"{variable.name} = {variable.varValue}")
print(f"Objective = {pulp.value(lp.objective)}")

#0-1规划仍然可以使用pulp求解，在设置约束条件的时候的时候，改成0<=x<=1即可
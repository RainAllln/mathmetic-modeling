#一个实际问题中往往不止一个指标，比如导弹需要射程远，耗燃料少，命中率高
#一般形式为：min f(x) = [f1(x),f2(x),...fn(x)]^T,s.t.={Gi(x)<=0,i=1,2...p;Hj(x)=0,j=1,2...q
#记Ω={x|Gi(x)<=0,Hj(x)=0}(即满足约束条件）为可行域（决策空间）f(Ω)={f(x)|x∈Ω}为多目标规划问题的像集（目标空间）
#多目标规划有三种解：最优解，有效解，满意解
#最优解定义：设x^(x拔)∈Ω,对于任意i=1,2...以及任意x∈Ω均有fi(x^)<=fi(x),则x^为最优解，一般来说最优解不常见，需要有效解
#有效解定义：设x^∈Ω,若不存在x∈Ω,使得fi(x)<=fi(x^),i=1,2...m,且至少有一个fj(x)<fj(x^),则x^为有效解,f(x^)有有效点
#满足解定义：根据决策者给出的m个阈值αi,当x^∈Ω满足fi(x^)<=αi时，就认为x^是可接受的，则x^是满意解
#一般来说目标函数之间有不可公度性，需要对目标函数去量纲，归一化
#有四种求有效解的方法：
# 1.线性加权法，该方法的基本思想是根据目标的重要性确定一个权重，以目标函数的加权平均值为评价函数，使其达到最优。权重的确定一般由决策者给出，因而具有较大的主观性，不同的决策者给的权重可能不同，从而会使计算的结果不同。
# 2.ε约束法,根据决策者的偏好，选择一个主要关注的参考目标，而将其他目标函数放到约束条件中。约束法也称主要目标法或参考目标法，参数ε是决策者对变为约束条件的目标函数的容许接受阈值。
# 3.理想点法,该方法的基本思想是:以每个单目标最优值为该目标的理想值，使每个目标函数值与理想值的差的加权平方和最小。
# 4.优先级法,该方法的基本思想是根据目标重要性分成不同优先级，先求优先级高的目标函数的最优值，在确保优先级高的目标获得不低于最优值的条件下，再求优先级低的目标函数
# 对于线性加权法：
# 1、要将多个目标函数统一为最大化和最小化问题(不同的加“-”号)才可以进行加权组合
# 2、如果目标函数量纲不同，则需要对其进行标准化再进行加权，标准化的方法一般是目标函数除以某一个常量，该常量是这个目标函数的某个取值，具体取何值可根据经验确定
# 3、对多目标函数进行加权求和，权重一般由该领域专家给定，实际比赛中，若无特殊说明，我们可令权重相同

#线性加权法：
#例题：max 2*x1+3*x2;max 4*x1-2*x2;s.t.={x1+x2<=10;2*x1+x2<=15;x1,x2>=0
#设权重α,目标函数变为max α(2*x1+3*x2)+(1-α)(4*x1-2*x2)
#最后要敏感性分析，敏感性分析是指从定量分析的角度研究有关因素发生某种变化对某一个或一组关键指标影响程度的一种不确定分析技术。其实质是通过逐一改变相关变量数值的方法来解释关键指标受这些因素变动影响大小的规律。
#线性加权法一般可以通过改变权重来看指标变化

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']
import numpy as np
import pandas as pd
import pulp

#  定义步长
stepSize = 0.01

#  初始化空的DataFrame以存储优化结果
solutionTable = pd.DataFrame(columns=["alpha","x1_opt","x2_opt","obj_value"])

x1 = pulp.LpVariable('x1')
x2 = pulp.LpVariable('x2')

#  使用stepSize迭代从0到1的alpha值，并将PuLP解决方案写入solutionTable
for i in range(0,101,int(stepSize*100)):
        #  再次声明问题
        linearProblem = pulp.LpProblem("多目标线性最大化",pulp.LpMaximize)
        #  在采样的alpha处添加目标函数
        linearProblem += (i/100)*(2*x1+3*x2) + (1-i/100)*(4*x1-2*x2)
        #  添加约束
        linearProblem += x1 + x2 <= 10
        linearProblem += 2*x1 + x2 <= 15
        #  解决这个问题
        solution = linearProblem.solve()
        #  将解决方案写入DataFrame
        solutionTable.loc[int(i/(stepSize*100))] = [i/100,pulp.value(x1),pulp.value(x2),pulp.value(linearProblem.objective)]

#  使用matplotlib.pyplot可视化优化结果
# --  创建线图
plt.plot(solutionTable["alpha"],solutionTable["obj_value"],color="red")
# --  添加轴标签
plt.xlabel("alpha")
plt.ylabel("obj_value")
# --  添加剧情标题
plt.title(" 最佳组合目标函数值作为alpha的函数 ")
# -- show plot
plt.show()
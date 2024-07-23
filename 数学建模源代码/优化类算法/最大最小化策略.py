#最大最小策略是一种保守博弈策略，使得博弈者最小收入最大化策略，是一种找出失败的最大可能性中最小值的算法
#现实中很多问题需要该思想，比如急救中心选址，需要其到所有地点的最大距离最小
#基本模型：min{max[f1(x)],max[f2(x)],...,max[fm(x)]},s.t.={Ax<=b;Aeq*x=beq;C(x)<=0;Ceq(x)=0;VLB<=X<=VUB

#例子：
import numpy as np
from scipy.optimize import minimize

# 假设有一组需求点的坐标
demand_points = np.array([
    [1, 2],
    [3, 1],
    [2, 4]
])


# 求解急救中心选址，使得离最远需求点的距离最小
def calculate_distance(center_coordinates):
    # 计算到所有需求点的距离
    distances = np.linalg.norm(demand_points - center_coordinates, axis=1)
    # 返回最远需求点的距离
    return max(distances)


# 初始猜测的急救中心选址
initial_guess = np.array([0, 0])  # 例如，(0, 0) 是一个猜测的选址点

# 使用scipy.optimize.minimize进行优化
result = minimize(calculate_distance, initial_guess, method='Powell')

# 输出最优选址
print("最优急救中心选址:", result.x)
print("最小距离:", calculate_distance(result.x))
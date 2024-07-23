#模拟退火算法属于启发式算法，一般用于解决一些np-hard问题，在有限的时间和空间资源下得到全局最优解，不一定是最优解，是一个还不错的解
#模拟退火算法的思想是:为了不被局部最优解困住，需要以一定概率跳出当前位置，暂时接受一个不太好的解。在搜索最优解的过程中逐渐降温，初期跳出去的概率比较大，进行广泛搜索;后期跳出去的概率比较小，尽量收敛到较优解
#算法步骤如下：1.设定初始解X，设定初始温度T
#2.在上一次解的基础上做调整，生成新解X',对比旧解f(X)和f(X'),如果新解更好了就接受，如果新解不够好，那么就以e^((f(X)-f(X'))/T)的概率接受
#3.降温，并重复以上步骤，直到迭代到一定次数为止
#可以先生成几个解，利用贪心算法选定初始解
#降温方法：没有固定方法，只能尝试，比如初始温度为1，每次迭代降低99%，降低到10^(-30)为止，也不一定要没代都降温，可以间隔几代
#如何生成新解是最重要的部分，要确保新解是有效的，
#https://blog.csdn.net/Chandler_river/article/details/132029417 代码示例
#https://blog.csdn.net/qq_42912425/article/details/138601036 代码示例

# import numpy as np
# import matplotlib.pyplot as plt
#
# def objective_function(x):
#     return x[0] ** 2 + 2 * x[0] - 15 + 4 * 4 * 2 * x[1] + 4 * x[1] ** 2 #目标函数
#
#
# def simulated_annealing(objective_func, initial_solution=np.array([0, 0]),
#                     temperature=100, min_temperature=0.1,
#                     cooling_rate=0.90, iter_max=100, seed=0):
#     np.random.seed(seed) #随机数种子
#     current_solution = initial_solution
#     best_solution = current_solution
#     scores = []
#
#     while temperature > min_temperature:
#         for j in range(iter_max):
#             # 生成新的解
#             new_solution = current_solution + np.random.uniform(-1, 1, len(current_solution))
#             # 函数原型：  numpy.random.uniform(low, high, size)
#             # 功能：从一个均匀分布[low, high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
#             # 参数介绍:
#             # low: 采样下界，float类型，默认值为0；
#             # high: 采样上界，float类型，默认值为1；
#             # size: 输出样本数目，为int或元组(tuple)
#             # 类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个值。
#
#             # 计算新解与当前解之间的目标函数值差异
#             current_cost = objective_func(current_solution)
#             new_cost = objective_func(new_solution)
#             cost_diff = new_cost - current_cost
#
#             # 判断是否接受新解
#             if cost_diff < 0 or np.exp(-cost_diff / temperature) > np.random.random():
#                 current_solution = new_solution
#
#             # 更新最优解
#             if objective_func(current_solution) < objective_func(best_solution):
#                 best_solution = current_solution
#
#             scores.append(best_solution)
#
#         # 降低温度
#         temperature *= cooling_rate
#
#     return best_solution,scores
#
# # 调用退火算法求解最小值
# best_solution, scores = simulated_annealing(objective_function)
#
# print(f"函数最小值： {objective_function(best_solution)} 自变量取值：{best_solution}")
#
# plt.plot(scores)
# plt.xlabel("Iteration")
# plt.ylabel("Best Score")
# plt.title("Simulated Annealing Optimization Process (Himmelblau)")
# plt.show()

#TSP:
import random
import math
import matplotlib.pyplot as plt


# 计算两城市之间的距离
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# 计算总路径长度
def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[i + 1]]) for i in range(len(tour) - 1)) + distance(cities[tour[0]],
                                                                                                        cities[tour[-1]]) #-1表示最后一行


# 邻域解生成函数
def random_neighbor_tsp(tour):
    new_tour = tour[:]
    i, j = random.sample(range(len(tour)), 2) #随机选两个城市交换位置
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


# 模拟退火算法
def simulated_annealing_tsp(cities, T0=1000, Tmin=1e-5, alpha=0.99, max_iter=1000):
    tour = list(range(len(cities)))
    random.shuffle(tour) #打乱数组顺序,洗牌
    best_solution = tour
    best_score = total_distance(tour, cities)
    current_solution = tour
    current_score = best_score
    T = T0
    scores = []

    for _ in range(max_iter):
        if T < Tmin:
            break

        # 生成新邻域解
        neighbor = random_neighbor_tsp(current_solution)
        neighbor_score = total_distance(neighbor, cities)

        # 接受概率计算
        if neighbor_score < current_score:
            current_solution = neighbor
            current_score = neighbor_score
        else:
            p = math.exp((current_score - neighbor_score) / T)
            if random.random() < p:
                current_solution = neighbor
                current_score = neighbor_score

        # 更新最优解
        if current_score < best_score:
            best_solution = current_solution
            best_score = current_score

        scores.append(best_score)

        # 降低温度
        T *= alpha

    return best_solution, best_score, scores


# 随机生成城市坐标
random.seed(0)
num_cities = 20
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

# 使用模拟退火算法解决 TSP
best_solution, best_score, scores = simulated_annealing_tsp(cities)

print("Best solution:", best_solution)
print("Best score:", best_score)

# 绘制优化过程中的得分变化
plt.plot(scores)
plt.xlabel("Iteration")
plt.ylabel("Best Score")
plt.title("Simulated Annealing Optimization Process (TSP)")
plt.show()

# 绘制 TSP 最优路径
x = [cities[i][0] for i in best_solution] + [cities[best_solution[0]][0]]
y = [cities[i][1] for i in best_solution] + [cities[best_solution[0]][1]]
plt.plot(x, y, marker='o')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Optimal Tour (TSP)")
plt.show()
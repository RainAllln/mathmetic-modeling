# https://blog.csdn.net/qq_38048756/article/details/108945267
# 粒子群算法基本思想：每个寻优问题的解都被想象成一只鸟，称为‘粒子’，所有的粒子都在一个D维空间进行搜索。
# 2.所有的粒子都由一个fitness function确定适应值以判断目前位置的好坏。
# 3.每一个粒子都被赋予记忆功能，能记住所搜寻到的最佳位置
# 4.每一个粒子还有一个速度以决定飞行的距离和方向。这个速度根据本身的飞行经验及同伴的飞行经验进行动态调整。
# D维空间中，有N个粒子；
# 粒子i位置：xi=(xi1,xi2,…xiD)，将xi代入适应函数f(xi)求适应值；
# 粒子i速度：vi=(vi1,vi2,…viD)
# 粒子i个体经历过的最好位置：pbesti=(pi1,pi2,…piD)
# 种群所经历过的最好位置：gbest=(g1,g2,…gD)
# 每一维位置和速度都有一个限制大小，
# 粒子i的第d维更新公式:
# v_id^k=w*v_id^(k-1)+c1*r1*(pbest-x_id^(k-1))+c2*r2*(gbest-x_id^(k-1))
# 其中，w是惯性权重，c1,c2是加速度常数，三个都是超参数，r1,r2是概率（0,1）
# w*v_id^(k-1)是惯性，保留一点粒子在第k-1次迭代的速度
# c1*r1*(pbest-x_id^(k-1))是粒子往自己最高适应度的地方移动
# c2*r2*(gbesst-x_id^(k-1))是粒子往群体的最好位置移动
# 三者相互结合就产生了新一轮迭代的粒子速度
# 粒子位置更新为:x_id^k = x_id^(k-1) + v_id^k
# 算法流程：
# 初始化粒子群体（群体规模为n），包括随机位置和速度。
# 根据fitness function ，评价每个粒子的适应度。
# 对每个粒子，将其当前适应值与其个体历史最佳位置（pbest）对应的适应值做比较，如果当前的适应值更高，则将用当前位置更新历史最佳位置pbest。
# 对每个粒子，将其当前适应值与全局最佳位置（gbest）对应的适应值做比较，如果当前的适应值更高，则将用当前粒子的位置更新全局最佳位置gbest。
# 根据公式更新每个粒子的速度与位置。
# 如未满足结束条件，则返回步骤2
# 通常算法达到最大迭代次数或者最佳适应度值的增量小于某个给定的阈值时算法停止。
# 示例代码解决的题目是 min f(x)=∑(i=1->3)[100*(x_(i+1)-x_i^2)^2+(x_i-1)^2],x_i∈[-30,30]
# 取种群大小为5，粒子维度是4，设最大速度Vmax=60

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fit_fun(x):  # 适应函数
    return sum(100.0 * (x[0][1:] - x[0][:-1] ** 2.0) ** 2.0 + (1 - x[0][:-1]) ** 2.0)


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = np.random.uniform(-x_max, x_max, (1, dim))  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截至条件
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros((1, dim))  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                    + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel #超过边界需要固定位边界值
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        part.set_pos(pos_value)
        value = fit_fun(part.get_pos())
        if value < part.get_fitness_value(): #本题中适应度越小越好,minf(x)
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)

    def update_ndim(self):

        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
            if self.get_bestFitnessValue() < self.tol:
                break

        return self.fitness_val_list, self.get_bestPosition()

if __name__ == '__main__':
    # test 香蕉函数
    pso = PSO(4, 5, 10000, 30, 60, 1e-4, C1=2, C2=2, W=1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5)


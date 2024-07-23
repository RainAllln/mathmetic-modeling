# 突变和基因重组是进化的原因，遗传算法是通过群体搜索技术，根据适者生存的原则逐代进化
# 最终得到准最优解操作包括:初始群体的产生、求每一个体的适应度、根据适者生存的原则选择优良个体被选出的优良个体两两配对，通过随机交叉其染色体的基因并随机变异某些染色体的基因生成下一代群体，按此方法使群体逐代进化，直到满足进化终止条件
# 编码与解码：比如说区间为[1,10]，生成2位串，只能表示四个数字，[1,4,7,10],精度为3，对应[00,01,10,11],解码时就是找对应的十进制数值
# 复制与交叉：复制有：1.轮盘赌法，将个体适应度映射成概率，适应度高的复制概率高；2.适应度前N/4的个体进行复制，替换掉后N/4的个体，精英产生精英;还有很多别的方法
# 交叉也有很多种方法，两两顺序交叉，或者多个交叉，可以是适应度前N/4的个体交叉，还能多段交叉
# 变异是为了跳出局部最优，前面好的解可以不动，可以选择适应度后N/4的解变异，或者是适应度大小映射变异概率
# 遗传算法的步骤如下:
# 产生M个初始解，构成初始种群
# 每对父母以一定概率生成一个新解(交配产生后代)
# 每个个体以一定概率发生突变(即将自己的解变变换产生新解)
# 编码和解码，将数值转化为二进制串，一个十进制数对应一个二进制的串
# 父代和子代合在一起，留下M个最好的个体进入下一轮，其余淘汰(进行自然选择)
# 重复以上迭代，最后输出最好的个体
# 如何选择参数:
# 遗传算法中要选择的参数很多:种群数量M、变异概率、生成子代的数量、迭代次数
# 种群数量M越大、迭代次数越多、生成的子代越多，当然更有希望找到最优解，但相应的计算资源消耗也会增大，只能在可接受范围内进行选择
# 如何选择初始种群:
# 其实可以随便初始化，但是较好的初始种群可以帮助更快收敛
# 例如随机生成若干个选最好的、贪心算法等
# 如何交配产生子代:
# 交配方法是最能体现creativity的地方，应该尽量继承父代，但也要进行足够的调整例如:
# 选择父亲的一个标号t，在母亲那里找到它后面的全部数字，并依序取出
# 把父亲标号t后面的部分接到母亲后面
# 把母亲取出来的数字接到父亲后面
# https://blog.csdn.net/RSociopath/article/details/124137755 代码示例
# 适应度选择：https://blog.csdn.net/weixin_30239361/article/details/101540896

import numpy as np
from numpy.ma import cos
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # 建模，不必需
import datetime  # 统计时间，不必需

DNA_SIZE = 24  # 编码长度
POP_SIZE = 100  # 种群大小
CROSS_RATE = 0.5  # 交叉率
MUTA_RATE = 0.15  # 变异率
Iterations = 50  # 迭代次数
X_BOUND = [0, 10]  # X区间
Y_BOUND = [0, 10]  # Y区间


def F(x, y):  # 函数
    return (6.452 * (x + 0.125 * y) * (cos(x) - cos(2 * y)) ** 2) / (
                0.8 + (x - 4.2) ** 2 + 2 * (y - 7) ** 2) + 3.226 * y


def getfitness(pop):  # 适应度函数
    x, y = decodeDNA(pop)
    temp = F(x, y)
    return (temp - np.min(temp)) + 0.0001


def decodeDNA(pop):  # 二进制转坐标，解码
    x_pop = pop[:, 1::2] #奇数索引
    y_pop = pop[:, ::2] #偶数索引
    # .dot()用于矩阵相乘
    # 2**np.arange(DNA_SIZE)[::-1]生成了2^0,2^1...2^(DNA_SIZE-1)的序列，与x_pop相乘，就是二进制转十进制，转完后除以精度
    # 1/(float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]))就是精度
    # 最后+首位
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


def select(pop, fitness):  # 选择
    temp = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / (fitness.sum()))
    return pop[temp]


# mutation函数以及crossmuta函数均为编码过程


def mutation(temp, MUTA_RATE):  # 变异
    if np.random.rand() < MUTA_RATE:
        mutate_point = np.random.randint(0, DNA_SIZE)
        temp[mutate_point] = temp[mutate_point] ^ 1  # ^为异或运算


def crossmuta(pop, CROSS_RATE):  # 交叉
    new_pop = []
    for i in pop:
        temp = i
        if np.random.rand() < CROSS_RATE:
            j = pop[np.random.randint(POP_SIZE)]
            cpoints1 = np.random.randint(0, DNA_SIZE * 2 - 1)
            cpoints2 = np.random.randint(cpoints1, DNA_SIZE * 2)
            temp[cpoints1:cpoints2] = j[cpoints1:cpoints2]
            mutation(temp, MUTA_RATE)
        new_pop.append(temp)
    return new_pop


def print_info(pop):  # 输出最优解等
    fitness = getfitness(pop)
    maxfitness = np.argmax(fitness)
    print("max_fitness", fitness[maxfitness])
    x, y = decodeDNA(pop)
    print("最优的基因型:", pop[maxfitness])
    print("(x,y):", (x[maxfitness], y[maxfitness]))
    print("F(x,y)_max=", F(x[maxfitness], y[maxfitness]))


def plot_3d(ax):  # 建模
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-20, 160)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(0.5)
    plt.show()


if __name__ == "__main__":  # 主函数
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # 如果出现程序跑通但不显示图片问题请使用这两行代码，注释掉ax=Axes3D(fig)
    plt.ion()
    plot_3d(ax)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    start_t = datetime.datetime.now()
    for i in range(Iterations):
        print("i:", i)
        x, y = decodeDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c="black", marker='o')
        plt.show()
        plt.pause(0.1)
        pop = np.array(crossmuta(pop, CROSS_RATE))
        fitness = getfitness(pop)
        pop = select(pop, fitness)
    end_t = datetime.datetime.now()
    print((end_t - start_t).seconds)
    print_info(pop)
    plt.ioff()
    plot_3d(ax)
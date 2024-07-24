# 一元线性回归模型是由y=ax+b构成的模型
# 利用最小二乘法可以得到a和b，最小二乘法是使y与估计值之间的离差平方和Q=∑(y-y^)^2最小来求解的
# 公式a=(n*∑(x_i*y_i)-∑x_i*∑y_i)/(n*∑x_i^2-(∑x_i)^2)
# b=y^- - a*x^-,(y^-指y的平均数)
# scikit-learn 是用于机器学习的最佳Python库之一，适用于拟合和预测。它为用户提供了不同的数值计算和统计建模选项。
# 它最重要的线性回归子模块是LinearRegression， 使用最小二乘法作为最小化标准来寻找线性回归的参数。
# https://blog.csdn.net/csdn1561168266/article/details/129214694?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E8%BF%9B%E8%A1%8C%E4%B8%80%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-8-129214694.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.array([20, 30, 33, 40, 15, 13, 26, 38, 35, 43])
    y = np.array([7, 9, 8, 11, 5, 4, 8, 10, 9, 10])
    # 转换成numpy的ndarray数据格式，n行1列,LinearRegression需要列格式数据，如下：
    X_train = np.array(x).reshape((len(x), 1))
    Y_train = np.array(y).reshape((len(y), 1))

    # 新建一个线性回归模型，并把数据放进去对模型进行训练
    lineModel = LinearRegression()
    lineModel.fit(X_train, Y_train)

    # 用训练后的模型，进行预测
    Y_predict = lineModel.predict(X_train)

    # coef_是系数，intercept_是截距
    a1 = lineModel.coef_[0][0]
    b = lineModel.intercept_[0]
    print("y=%.4f*x+%.4f" % (a1, b))

    # 对回归模型进行评分，一般拿别的数据评分
    print("得分", lineModel.score(X_train, Y_train))

    # 简单画图显示
    plt.scatter(x, y, c="blue")
    plt.plot(X_train, Y_predict, c="red")
    plt.show()

# 由y=β_0+β_1*x_1+β_2*x_2+...β_k*x_k(回归平面方程)确定的:
# {Y=X*β+ε，E(ε)=0，Cov(ε,ε)=σ^2*I_n
# 其中，ε是误差项，误差项的期望为0，其协方差矩阵=σ^2乘以n阶单位矩阵I_n
# 这样的模型是k元线性回归模型，简记为(Y,X*β,σ^2*I_n)
#
# 基本操作:
# 对β和σ^2做点估计，建立y与x_1,x_2...的线性关系
# 对模型参数和模型结果检验
# 对y值做区间预测
#
# 多元回归模型检验:
# (1)F检验
# 当H_0成立时，F=(U/k) / Q_e/(n-k-1) ~F(k,n-k-1)
# 如果F>F_1-α(k,n-k-1)，则拒绝H_0,认为有显著线性关系
# 其中U=回归平方和,Q_e=残差平方和
# (2)R检验
# 定义R=√U/(U+Qe)
# 称为多元相关系数/复相关系数
# 由于F=((n-k-1)/k)*(R^2/1-R^2)
# 所以F检验和R检验等效

#https://blog.csdn.net/weixin_46277779/article/details/135667638?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E5%A4%9A%E5%85%83%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-135667638.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
from sklearn.linear_model import LinearRegression
import numpy as np
import random

if __name__ == '__main__':
    # 随机创建X1,X2,X3,Y。使Y=4*X1-3*X2+2X3+5
    X1 = [random.randint(0, 100) for i in range(0, 50)]
    X2 = [random.randint(0, 60) for i in range(0, 50)]
    X3 = [random.randint(0, 35) for i in range(0, 50)]
    Y = [4 * x1 - 3 * x2 + 2 * x3 + 5 for x1, x2, x3 in zip(X1, X2, X3)]

    # 组合X1，X2成n行2列数据
    X_train = np.array(X1 + X2 + X3).reshape((len(X1), 3), order="F")
    Y_train = np.array(Y).reshape((len(Y), 1))

    # 加入噪声干扰
    noise = np.random.randn(50, 1)
    noise = noise - np.mean(noise)

    Y_train = Y_train + noise

    # 新建一个线性回归模型，并把数据放进去对模型进行训练
    lineModel = LinearRegression()
    lineModel.fit(X_train, Y_train)

    # 用训练后的模型，进行预测
    Y_predict = lineModel.predict(X_train)
    f = ""
    # coef_是系数，intercept_是截距
    a_arr = lineModel.coef_[0]
    b = lineModel.intercept_[0]

    for i in range(0, len(a_arr)):
        ai = a_arr[i]
        if ai >= 0:
            ai = "+%.4f" % (ai)
        else:
            ai = "%.4f" % (ai)
        f = f + "%s*x%s" % (ai, str(i + 1))
    f = "y=%s+%.4f" % (f[1:], b)

    print("拟合方程", f)
    # 对回归模型进行评分，这里简单使用训练集进行评分，实际很多时候用其他的测试集进行评分
    print("得分", lineModel.score(X_train, Y_train))

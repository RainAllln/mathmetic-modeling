# https://blog.csdn.net/Yellow_python/article/details/104504629?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172182081716800184157642%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=172182081716800184157642&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-3-104504629-null-null.142^v100^pc_search_result_base1&utm_term=python%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90&spm=1018.2226.3001.4187
# 各种机器学习样例↑
# https://blog.csdn.net/qq_35591253/article/details/130938485?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172183593616800211552547%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172183593616800211552547&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-130938485-null-null.142^v100^pc_search_result_base1&utm_term=python%E5%81%9A%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90&spm=1018.2226.3001.4187
# GAM非线性回归↑
# https://blog.csdn.net/zbh13859825167/article/details/138203057?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E5%81%9A%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-138203057.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
# 非线性回归及流程
# https://blog.csdn.net/weixin_71894495/article/details/132035479?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172183593616800211552547%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172183593616800211552547&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-132035479-null-null.142^v100^pc_search_result_base1&utm_term=python%E5%81%9A%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90&spm=1018.2226.3001.4187
# 多项式非线性回归

import random
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义非线性模型函数
def nonlinear_model(xy, a, b, c):
    x, y = xy
    return (a * x - b) * y + c

# 指定Excel文件路径并读取
excel_file_path = 'E:\zbh.xlsx'
df = pd.read_excel(excel_file_path)

x = 'R'
y = 'SOS'
z = 'MEAN'
x_data = np.array(df[x])
y_data = np.array(df[y])
z_data = np.array(df[z])


# 利用 curve_fit 进行非线性回归
popt, pcov = curve_fit(nonlinear_model, (x_data, y_data), z_data)
a_fit, b_fit, c_fit = popt
z_fit = nonlinear_model((x_data, y_data), a_fit, b_fit, c_fit)

# 计算指标
ss_total = np.sum((z_data - np.mean(z_data)) ** 2)
ss_reg = np.sum((z_fit - np.mean(z_data)) ** 2)
r_squared = ss_reg / ss_total
rmse = np.sqrt(np.mean((z_data - z_fit) ** 2))
print("R方:", r_squared)
print("RMSE:", rmse)
# 拟合公式
formula = "{} = ({:.2f} * {} + ({:.2f})) * {} + {:.2f}".format(z, a_fit, x, b_fit, y, c_fit)
print(formula)

# 绘制三维散点图和拟合曲面
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# 散点图
ax.scatter(x_data, y_data, z_data, color='blue', label='Data Points')
# 曲面图
X, Y = np.meshgrid(np.linspace(min(x_data), max(x_data), 30),
                   np.linspace(min(y_data), max(y_data), 30))
Z = nonlinear_model((X.flatten(), Y.flatten()), a_fit, b_fit, c_fit).reshape(X.shape)
ax.plot_surface(X, Y, Z, color='r', alpha=0.6, label='Fitted Surface')

ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)
plt.title(x +"-"+ y + "-" + z + ":" + formula)
plt.show()

# 原理:通过将线性回归模型的输出通过一个逻辑函数进行转换，将连续的预测值映射到0和之间，表示属于某一类的概率。逻辑回归的模型假设因变量服从伯努利分布，通过最大似然估计来拟合模型参数。
# 解决二分类问题
# https://blog.csdn.net/m0_47256162/article/details/129776507 原理实现
# https://blog.csdn.net/guan1843036360/article/details/129441586?ops_request_misc=&request_id=&biz_id=102&utm_term=python%20%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-129441586.nonecase&spm=1018.2226.3001.4187 项目实战
# 1.获取数据
# 2.基本数据处理
# 2.1 缺失值处理
# 2.2 确定特征值,目标值
# 2.3 分割数据
# 3.特征工程(标准化)
# 4.机器学习(逻辑回归)
# 5.模型评估
# sklearn.linear_model.LogisticRegression(solver=‘liblinear’, penalty =‘l2’, C = 1.0)
# solver可选参数: {‘liblinear’, ‘sag’, ‘saga’, ‘newton - cg’, ‘lbfgs’}，
# 默认: ‘liblinear’；用于优化问题的算法。
# 对于小数据集来说，“liblinear”是个不错的选择，而“sag”和’saga’对于大型数据集会更快。
# 对于多类问题，只有’newton - cg’， ‘sag’， 'saga’和’lbfgs’可以处理多项损失;“liblinear”仅限于“one-versus-rest”分类。
# penalty：正则化的种类
# C：正则化力度

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


import ssl
ssl._create_default_https_context = ssl._create_unverified_context  #关闭ssl验证

# 1.获取数据
#names 表示设置的列索引字段：与数据源字段一致
names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                  names=names)  #将列索引注入
data.head()
# 2.基本数据处理
# 2.1 缺失值处理
data = data.replace(to_replace="?", value=np.NaN)  #数据源中有16个缺失值，这里对缺失值进行处理
data = data.dropna()
# 2.2 确定特征值,目标值
x = data.iloc[:, 1:10]
x.head()
y = data["Class"]
y.head()
# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
# 3.特征工程(标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4.机器学习(逻辑回归)
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
# 5.模型评估
y_predict = estimator.predict(x_test)
y_predict
estimator.score(x_test, y_test)

# 分类评估报告
# sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None)
# y_true：真实目标值
# y_pred：估计器预测目标值
# labels: 指定类别对应的数字
# target_names：目标类别名称
# return：每个类别精确率与召回率

# 5.模型评估
# 5.1 准确率
ret = estimator.score(x_test, y_test)
print("准确率为:\n", ret)

# 5.2 预测值
y_pre = estimator.predict(x_test)
print("预测值为:\n", y_pre)

# 5.3 精确率\召回率指标评价
ret = classification_report(y_test, y_pre, labels=(2, 4), target_names=("良性", "恶性"))
print(ret)

roc_auc_score(y_test, y_pre)


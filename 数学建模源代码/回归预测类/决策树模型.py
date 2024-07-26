# 原理:决策树通过对数据集进行递归地划分，以建立一个树形结构，每个内部节点表示一个属性上的测试，每个分支代表一个测试输出，每个叶节点代表一个类别标签或者一个连续值。
# 在分类问题中，决策树的目标是创建一个能够对实例进行准确分类的模型。
# 在建立决策树的过程中，通常使用信息增益(03算法)、基尼不纯度(CART算法)或者增益率等指标来选择最佳划分属性。
# 优先选择决策树，决策树不行就使用随机森林，再不行朴素贝叶斯或者其他模型
# https://blog.csdn.net/qq_34160248/article/details/127170221?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%86%B3%E7%AD%96%E6%A0%91python&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-127170221.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187
# 分类决策树模型DecisionTreeClassifier()的常用超参数
# criterion：特征选择标准，取值为’entropy’（信息熵）和’gini’（基尼系数），默认值为’gini’。
# splitter：取值为’best’和’random’。'best’指在特征的所有划分点中找出最优的划分点，适合样本量不大的情况；‘random’指随机地在部分划分点中寻找局部最优的划分点，适合样本量非常大的情况；默认值为’best’。
# max_depth**：决策树最大深度，取值为int型数据或None，默认值为None。一般数据或特征较少时可以不设置，如果数据或特征较多，可以设置最大深度进行限制。
# min_samples_split：子节点往下分裂所需的最小样本数，默认值为2。如果子节点中的样本数小于该值则停止分裂。
# min_samples_leaf：叶子节点的最小样本数，默认值为1。如果叶子节点中的样本数小于该值，该叶子节点会和兄弟节点一起被剪枝，即剔除该叶子节点和其兄弟节点，并停止分裂。
# min_weight_fraction_leaf：叶子节点最小的样本权重和，默认值为0，即不考虑权重问题。如果小于该值，该叶子节点会和兄弟节点一起被剪枝。如果较多样本有缺失值或者样本的分布类别偏差很大，则需考虑样本权重问题。
# max_features：在划分节点时所考虑的特征值数量的最大值，默认值为None，可以传入int型或float型数据，如果传入的是float型数据，则表示百分数。
# max_leaf_nodes：最大叶子节点数，默认值为None，可以传入int型数据。
# class_weight：指定类别权重，默认值为None，可以取’balanced’，代表样本量少的类别所对应的样本权重更高，也可以传入字典来指定权重。该参数主要是为防止训练集中某些类别的样本过多，导致训练的决策树过于偏向这些类别。除了指定该参数，还可以使用过采样和欠采样的方法处理样本类别不平衡的问题。
# random_state：当数据量较大或特征变量较多，可能在某个节点划分时，会遇到两个特征变量的信息增益或基尼系数下降值相同的情况，此时决策树模型默认会从中随机选择一个特征变量进行划分，这样可能会导致每次运行程序后生成的决策树不一致。设置random_state参数（如设置为123）可以保证每次运行程序后各节点的分裂结果都是一致的，这在特征变量较多、树的深度较深时较为重要。

# 分类决策树
from sklearn.tree import DecisionTreeClassifier

# X是特征变量，共有5个训练数据，每个数据有2个特征，如数据[1，2]，它的第1个特征的数值为1，第2个特征的数值为2。
X = [[1,2],[3,4],[5,6],[7,8],[9,10]]

# 目标变量，共有2个类别——0和1。
y = [1,0,0,1,1]

# 第4行代码引入模型并设置随机状态参数random_state为0
# 这里的0没有特殊含义，可换成其他数字。它是一个种子参数，可使每次运行结果一致。
model = DecisionTreeClassifier(random_state=0)

model.fit(X,y)
y_predict = model.predict([[5,5]])
print(y_predict)

# 输出结果
# array([0])

# 回归决策树
from sklearn.tree import DecisionTreeRegressor
X = [[1,2],[3,4],[5,6],[7,8],[9,10]]
y = [1,2,3,4,5]
model = DecisionTreeRegressor(max_depth=2,random_state=0)
model.fit(X,y)
print(model.predict([[9,9]]))

# 输出
# array([4.5])

# 例题，员工离职问题
# 1.数据读取与预处理
import pandas as pd
df = pd.read_excel('员工离职预测模型.xlsx')
df = df.replace({'工资':{'低':0,'中':1,'高':2}})

# 2.提取特征变量与目标变量
X = df.drop(columns=['离职'])
y = df['离职']

# 3.划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

# 4.模型训练与拟合
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3,random_state=123)
model.fit(X_train,y_train)
model.score(X_test,y_test) #查看预测准确度

# 模型效果评估
y_pred = model.predict(X_test)
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_pred)
a.head()
model.score(X_test,y_test) #查看整体准确度

# 预测离职和不离职概率
y_pred_proba = model.predict_proba(X_test) #二维数组
b = pd.DataFrame(y_pred_proba,columns=['离职概率','不离职概率'])

# ROC曲线评估
from sklearn.metrics import roc_curve
#此时获得的tpr、tpr、thres均为一维数组
fpr, tpr, thres = roc_curve(y_test,y_pred_proba[:,1])
a = pd.DataFrame()
a['阈值'] = list(thres)
a['假警报率']= list(fpr)
a['命中率']= list(tpr)

import matplotlib.pyplot as plt
#设置正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt. plot(fpr, tpr)
plt.xlabel('假警报率')
plt.ylabel('命中率')
plt.title('ROc曲线')
plt.show()
# 得到AUC值(ROC面积)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba[:,1])

# 特征重要性评估
features = X.columns
importances = model.feature_importances_
df = pd.DataFrame
df['特征名称'] = features
df['特征重要性'] = importances
df.sort_values('特征重要性',ascending=False)

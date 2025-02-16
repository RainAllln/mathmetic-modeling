# 原理:KNN算法的核心思想是通过计算待预测样本与训练集中各个样本的距离，然后选取距离最近的K个样本,根据这K个样本的类别进行投票决定待预测样本的类别。
# 属于非监督式学习，数据可以没有标签，让模型自动得到有K类的结果
# 计算各个点的距离，根据设定的阈值归类到不同类别
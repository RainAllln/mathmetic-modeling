# 原理:随机森林由多棵决策树组成，每棵树都是使用随机选择的特征和随机选择的样本进行训练。
# 在进行预测时，随机森林对所有树的预测结果进行平均或多数投票，以得到最终的预测结果。
# 通过引入随机性，随机森林能够降低模型的方差，提高模型的鲁棒性。
# 随机森林缓解过拟合问题（训练模型样本太特殊，导致新样本进来时表现差）
# https://blog.csdn.net/qq_42912425/article/details/138666703?ops_request_misc=&request_id=&biz_id=102&utm_term=%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97python&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-138666703.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187

# 示例一鸢尾花分类
# 导入所需库

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}

# 使用 GridSearchCV 进行超参数调优
# n_estimators：树的数量
# max_depth：树的最大深度
# min_samples_split：节点分裂所需的最小样本数
# max_features：寻找最佳分裂时的特征数量
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数组合进行分类
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 输出分类报告和混淆矩阵
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 绘制特征重要性
# feature_importances = rf.feature_importances_
# features = iris.feature_names
# sns.barplot(x=feature_importances, y=features)
# plt.xlabel("Feature Importance Score")
# plt.ylabel("Features")
# plt.title("Feature Importance in Random Forest")
# plt.show()


# # 示例2，加州房价回归
# # 导入所需库
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 加载加州房价数据集
# california = fetch_california_housing()
# X = california.data
# y = california.target
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # 使用随机森林进行回归
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
#
# # 预测测试集
# y_pred = rf.predict(X_test)
#
# # 输出性能指标
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: {mse:.2f}")
# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"R-squared Score: {r2:.2f}")
#
# # 绘制预测值与实际值的散点图
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted Values (Random Forest Regression)")
# plt.show()
#
# # 绘制特征重要性
# feature_importances = rf.feature_importances_
# features = california.feature_names
# sns.barplot(x=feature_importances, y=features)
# plt.xlabel("Feature Importance Score")
# plt.ylabel("Features")
# plt.title("Feature Importance in Random Forest Regression")
# plt.show()


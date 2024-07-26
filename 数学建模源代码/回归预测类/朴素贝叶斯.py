# 原理:朴素贝叶斯模型基于贝叶斯定理，它假设特征之间相互独立，即给定类别下特征之间是条件独立的。
# 基于这个假设，可以利用训练数据计算出每个特征在各个类别下的条件概率，并结合贝叶斯定理计算出给定特征条件下各个类别的概率，最终选择概率最大的类别作为预测结果。
# https://blog.csdn.net/weixin_66845445/article/details/138135601?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172200796716800207025251%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172200796716800207025251&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-138135601-null-null.142^v100^pc_search_result_base1&utm_term=%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%A8%A1%E5%9E%8Bpython&spm=1018.2226.3001.4187
import numpy as np
import math
import pandas as pd


def loadDataSet():
    dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
               ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
               ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
               ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
               ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
               ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
               ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],

               ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
               ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
               ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
               ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
               ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']]
    testSet = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]  # 待测集
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']  # 特征

    return dataSet, testSet, labels


# 计算(不同类别中指定连续特征的)均值、标准差
def mean_std(feature, cla):
    dataSet, testSet, labels = loadDataSet()
    lst = [item[labels.index(feature)] for item in dataSet if item[-1] == cla]  # 类别为cla中指定特征feature组成的列表
    mean = round(np.mean(lst), 3)  # 均值
    std = round(np.std(lst), 3)  # 标准差

    return mean, std


# ? 计算先验概率P(c)
def prior():
    dataSet = loadDataSet()[0]  # 载入数据集
    countG = 0  # 初始化好瓜数量
    countB = 0  # 初始化坏瓜数量
    countAll = len(dataSet)

    for item in dataSet:  # 统计好瓜个数
        if item[-1] == "好瓜":
            countG += 1
    for item in dataSet:  # 统计坏瓜个数
        if item[-1] == "坏瓜":
            countB += 1

    # 计算先验概率P(c)
    P_G = round(countG / countAll, 3)
    P_B = round(countB / countAll, 3)

    return P_G, P_B


# ? 计算离散属性的条件概率P(xi|c)
def P(index, cla):
    dataSet, testSet, labels = loadDataSet()  # 载入数据集
    countG = 0  # 初始化好瓜数量
    countB = 0  # 初始化坏瓜数量

    for item in dataSet:  # 统计好瓜个数
        if item[-1] == "好瓜":
            countG += 1
    for item in dataSet:  # 统计坏瓜个数
        if item[-1] == "坏瓜":
            countB += 1

    lst = [item for item in dataSet if
           (item[-1] == cla) & (item[index] == testSet[index])]  # lst为cla类中第index个属性上取值为xi的样本组成的集合
    P = round(len(lst) / (countG if cla == "好瓜" else countB), 3)  # 计算条件概率

    return P


# ? 计算连续属性的条件概率p(xi|c)
def p():
    dataSet, testSet, labels = loadDataSet()  # 载入数据集
    denG_mean, denG_std = mean_std("密度", "好瓜")  # 好瓜密度的均值、标准差
    denB_mean, denB_std = mean_std("密度", "坏瓜")  # 坏瓜密度的均值、标准差
    sugG_mean, sugG_std = mean_std("含糖率", "好瓜")  # 好瓜含糖率的均值、标准差
    sugB_mean, sugB_std = mean_std("含糖率", "坏瓜")  # 坏瓜含糖率的均值、标准差
    # p(密度|好瓜)
    p_density_G = (1 / (math.sqrt(2 * math.pi) * denG_std)) * np.exp(
        -(((testSet[labels.index("密度")] - denG_mean) ** 2) / (2 * (denG_std ** 2))))
    p_density_G = round(p_density_G, 3)
    # p(密度|坏瓜)
    p_density_B = (1 / (math.sqrt(2 * math.pi) * denB_std)) * np.exp(
        -(((testSet[labels.index("密度")] - denB_mean) ** 2) / (2 * (denB_std ** 2))))
    p_density_B = round(p_density_B, 3)
    # p(含糖率|好瓜)
    p_sugar_G = (1 / (math.sqrt(2 * math.pi) * sugG_std)) * np.exp(
        -(((testSet[labels.index("含糖率")] - sugG_mean) ** 2) / (2 * (sugG_std ** 2))))
    p_sugar_G = round(p_sugar_G, 3)
    # p(含糖率|坏瓜)
    p_sugar_B = (1 / (math.sqrt(2 * math.pi) * sugB_std)) * np.exp(
        -(((testSet[labels.index("含糖率")] - sugB_mean) ** 2) / (2 * (sugB_std ** 2))))
    p_sugar_B = round(p_sugar_B, 3)

    return p_density_G, p_density_B, p_sugar_G, p_sugar_B


# ? 预测后验概率P(c|xi)
def bayes():
    # ? 计算类先验概率
    P_G, P_B = prior()
    # ? 计算离散属性的条件概率
    P0_G = P(0, "好瓜")  # P(青绿|好瓜)
    P0_B = P(0, "坏瓜")  # P(青绿|坏瓜)
    P1_G = P(1, "好瓜")  # P(蜷缩|好瓜)
    P1_B = P(1, "坏瓜")  # P(蜷缩|好瓜)
    P2_G = P(2, "好瓜")  # P(浊响|好瓜)
    P2_B = P(2, "坏瓜")  # P(浊响|好瓜)
    P3_G = P(3, "好瓜")  # P(清晰|好瓜)
    P3_B = P(3, "坏瓜")  # P(清晰|好瓜)
    P4_G = P(4, "好瓜")  # P(凹陷|好瓜)
    P4_B = P(4, "坏瓜")  # P(凹陷|好瓜)
    P5_G = P(5, "好瓜")  # P(硬滑|好瓜)
    P5_B = P(5, "坏瓜")  # P(硬滑|好瓜)
    # ? 计算连续属性的条件概率
    p_density_G, p_density_B, p_sugar_G, p_sugar_B = p()

    # ? 计算后验概率
    isGood = P_G * P0_G * P1_G * P2_G * P3_G * P4_G * P5_G * p_density_G * p_sugar_G  # 计算是好瓜的后验概率
    isBad = P_B * P0_B * P1_B * P2_B * P3_B * P4_B * P5_B * p_density_B * p_sugar_B  # 计算是坏瓜的后验概率

    return isGood, isBad


if __name__ == '__main__':
    dataSet, testSet, labels = loadDataSet()
    testSet = [testSet]
    df = pd.DataFrame(testSet, columns=labels, index=[1])
    print("=======================待测样本========================")
    print(f"待测集:\n{df}")

    isGood, isBad = bayes()
    print("=======================后验概率========================")
    print("后验概率:")
    print(f"P(好瓜|xi) = {isGood}")
    print(f"P(好瓜|xi) = {isBad}")
    print("=======================预测结果========================")
    print("predict ---> 好瓜" if (isGood > isBad) else "predict ---> 坏瓜")


# 例题2：鸢尾花
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 因为鸢尾花数据集中各特征为连续特征值，所以需要导入sklearn库中的高斯朴素贝叶斯分类器GaussianNB；

#? 定义朴素贝叶斯分类器
class bayes_model():
    def __int__(self):
        pass

    # ? 加载数据
    def load_data(self):
        data = load_iris()
        iris_data = data.data
        iris_target = data.target
        iris_featureNames = data.feature_names

        iris_features = pd.DataFrame(data=iris_data, columns=iris_featureNames)
        train_x, test_x, train_y, test_y = train_test_split(iris_features, iris_target, test_size=0.3, random_state=123)

        return train_x, test_x, train_y, test_y

    #? 训练高斯朴素贝叶斯模型
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf

    # ? 处理预测的数据
    def proba_data(self, clf, test_x, test_y):
        y_predict = clf.predict(test_x)  # 返回待预测样本的预测结果(所属类别)
        y_proba = clf.predict_proba(test_x)  # 返回预测样本属于各标签的概率
        accuracy = accuracy_score(test_y, y_predict) * 100  # 计算predict预测的准确率

        return y_predict, y_proba, accuracy

    # ? 训练数据
    def exc_p(self):
        train_x, test_x, train_y, test_y = self.load_data()  # 加载数据
        clf = self.train_model(train_x, train_y)  # 训练 高斯朴素贝叶斯 模型clf
        y_predict, y_proba, accuracy = self.proba_data(clf, test_x, test_y)  # 利用训练好的模型clf对测试集test_x进行结果预测分析

        return train_x, test_x, train_y, test_y, y_predict, y_proba, accuracy


if __name__ == '__main__':
    train_x, test_x, train_y, test_y, y_predict, y_proba, accuracy = bayes_model().exc_p()

    # 训练集与其标签 df1
    df1_1 = pd.DataFrame(train_x).reset_index(drop=True)
    df1_2 = pd.DataFrame(train_y)
    df1 = pd.merge(df1_1, df1_2, left_index=True, right_index=True)
    df1.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'train classify']
    print("=============================================训练集==============================================")
    print(f'The train dataSet is:\n{df1}\n')
    # 测试集与其标签 df2
    df2_1 = pd.DataFrame(test_x).reset_index(drop=True)
    df2_2 = pd.DataFrame(test_y)
    df2 = pd.merge(df2_1, df2_2, left_index=True, right_index=True)
    df2.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'test classify']
    print("=============================================测试集==============================================")
    print(f'The test dataSet is:\n{df2}\n')
    # 预测结果
    tot1 = pd.DataFrame([test_y, y_predict]).T
    tot2 = pd.DataFrame(y_proba).applymap(lambda x: '%.2f' % x)
    tot = pd.merge(tot1, tot2, left_index=True, right_index=True)
    tot.columns = ['y_true', 'y_predict', 'predict_0', 'predict_1', 'predict_2']
    print("============================================预测结果==============================================")
    print('The result of predict is: \n', tot)
    print("=============================================准确率==============================================")
    print(f'The accuracy of Testset is: {accuracy:.2f}%')
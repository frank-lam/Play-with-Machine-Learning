# coding: utf-8

import numpy as np
from math import sqrt # math包里面只用了sqrt这一个函数
from collections import Counter
from .metrics import accuracy_score # 为什么要加个小数点呢？


class KNNClassifier: 
    # 为什么都要加入self, 类的成员函数才需要加入吧
    def __init__(self, k): # 凡是有参数的地方都需要进行断言
        """初始化kNN分类器模型"""
        assert k >= 1
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集也就是kNN模型训练kNN分类器模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 别忘了之前的k
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self # 为什么返回self？遵循scikit-learn的习惯
        # 因为可能要传给其他模型？

    def predict(self, X_predict):
        """给定待预测数据集X_predict,利用训练出来的kNN分类器模型预测数据，返回表示X_predict的结果向量"""
        # 之前是忘记了考虑k的断言，现在又忘了步骤性的考虑
        # 如果要求当前函数在某个函数之后执行，则必须有内置数据不为空
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before execute predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must equal to X_train"
        # 调用内部函数都需要加self.
        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        # return y_predict
        return np.array(y_predict) # 用小写字母吧

    def score(self, X_test, y_test):
        # 断言，如果参数都传入了其他函数，没有之间使用参数，则不需要使用断言，因为出问题也是其他函数的事情
        assert X_test.shape[0] == y_test.shape[0], \
            "the size of X_test must be equal to the size of y_test"
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    # 最难的部分之前kNN函数已经写过了
    def _predict(self, x_predict):
        """给定单个待预测数据x，返回x的预测结果值"""
        distances = [sqrt(np.sum((x_train-x_predict)**2)) \
            for x_train in self._X_train] # 可以理解为1个横向的列表，不过1维的时候看做列向量
        sortdistances = np.argsort(distances)
        # 投票统计
        # votes = Counters(sortdistances[:self.k])
        # 统计的应该是得到的标签而不是距离
        nearest = np.argsort(distances)
        # 求nearest对应的标签向量
        topk_y = [self._y_train[i] for i in  nearest[:self.k]]
        # votes = Counter[topk_y] # 注意[]表示取向量元素的意思
        votes = Counter(topk_y)

        # return distances
        # 返回的应该是输入样本的预测值
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k # return self
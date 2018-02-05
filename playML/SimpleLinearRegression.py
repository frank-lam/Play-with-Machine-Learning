# coding: utf-8
import numpy as np 

class SimpleLinearRegression1: # 后续还会实现一个2
    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集计算线性规划模型的参数，使用最优化原理的方法"""
        # 进行断言保证传进来的数据时合法的
        # 维度方面的考虑
        # 看到咖啡店里座下写东西，旁边6个女人在说话，我还有什么理由瞻左顾右呢
        assert x_train.ndim == 1, \
            "SimpleLinearRegressor only solve with one dimetional"
        # 考虑变量之间的关系
        assert len(x_train) == len(y_train), \
            "the size of x_train must equal to the size of y_train"
        # 还记得y_hat = ax + b吗？我们要计算其中的a和b
        # a = sigma[(x_i-x_mean)*(y_i-y_mean)]/(x_i-x_mean)**2
        # b = y_mean - a*x_mean
        # 计算分子
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = 0.0
        d = 0.0
        # for x_i in x_train:
        for x_i,y_i in zip(x_train, y_train):
            # num += (x_i - x_mean)*(y_i - y_mean)/(x_i-x_mean)**2
            num += (x_i-x_mean)*(y_i-y_mean)
            d += (x_i-x_mean)**2
        # a = num / d
        # b = y_mean - a * x_mean
        self.a_ = num / d
        # self.b_ = y_mean - a * x_mean
        self.b_ = y_mean - self.a_ * x_mean
        return self # sklearn对fit的规范
    
    def predict(self, x_predict):
        # "the dimentioal of x_prediction must be one"
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        # 
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict" # 如果你忘了使用self.a_这里就会报错
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x): # 用单个输入的预测函数处理输入数据集
    # def _predict(self, x_single)
        # y_hat = a * x + b
        y_hat = self.a_ * x + self.b_
        return y_hat

    def __repr__(self): # 字符串输出的算法
        return "SimpleLinearRegression()"
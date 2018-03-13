# coding: utf-8
import numpy as np 
from math import sqrt
# from __future__ import division

def accuracy_score(y_true, y_predict): # 精确度打分
    """计算y_true和y_predict之间的准确度"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    # from __future__ import division
    return sum(y_true == y_predict) / len(y_true)

# 添加线性回归的度量函数
def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    # 断言是针对计算而言的，而不是针对外部变量而言的
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true-y_predict)**2) / len(y_true) # 先差值平方求和，然后在平均

def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""
    # return sqrt(mean_squared_error)
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true-y_predict)) / len(y_true)    

def r2_score(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return 1-mean_squared_error(y_true, y_predict) / np.var(y_true)
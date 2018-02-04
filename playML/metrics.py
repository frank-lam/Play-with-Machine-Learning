# coding: utf-8
import numpy as np 
# from __future__ import division

def accuracy_score(y_true, y_predict): # 精确度打分
    """计算y_true和y_predict之间的准确度"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    # from __future__ import division
    return sum(y_true == y_predict) / len(y_true)
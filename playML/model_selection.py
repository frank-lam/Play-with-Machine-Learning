# coding: utf-8
# 为什么要加 # coding: utf-8 之前文件里有中文也没有在意啊
import numpy as np 

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """"""
    assert X.shape[0]==y.shape[0], \
        "the size of X_train must be equal to y"
    # assert 0<=test_ratio<=1, \
    # 注意是否加小数点
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"

    # np.random.seed(seed)
    if seed: # 用于debug，希望得到的随机数一样
        np.random.seed(seed) # 种子

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio) # 因为乘以小数不能保证得到整数
    # test_size
    train_size = len(X) - test_size # 这种一逼就出来的就不需要了，有test_size就够了
    # print(shuffle_indexes.shape[0])
    # print(test_size)
    # print(train_size)
    # train_indexes = shuffled_indexes[:train_size]
    # test_indexes = shuffled_indexes[train_size:]

    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    # 怎么把shuffle_indexes套进去呢？
    X_train = X[train_indexes]
    X_test = X[test_indexes] # 可以把训练数据的X_train和y_train放在一起啊

    y_train = y[train_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test

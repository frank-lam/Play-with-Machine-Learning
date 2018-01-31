import numpy as np
from math import sqrt # 可以理解为函数声明
from collections import Counter

# 把原来Jupyter Notebook里的代码都挪到这里面来了
def kNN_classify(k, X_train, y_train, x): # 大杂烩，全扔进去，代码没有突出处理流程
    assert 1<=k<=X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train" # 不要把.shape忘记，\后面不能有空格
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to the feature number of X_train"

    # 求距离，这里我们只求了一个测试数据x而不是一组测试数据
    distances = [sqrt(np.sum(x_train-x)**2) for x_train in X_train] # distances表示一个列向量
    # 确定选民
    nearest = np.argsort(distances)
    topk_y = [y_train[i] for i in nearest[:k]]
    # 统计得票
    votes = Counter(topk_y)

    return votes.most_common(1)[0][0] 
    # 第1个1表示取得票前1多的统计数据，第2个0表示其中第0个元组，第3个0表示取元组的第0列也就是键名了
    # 元组中只包含一个元素时，需要在元素后面添加逗号
    # tup3 = tup1 + tup2;
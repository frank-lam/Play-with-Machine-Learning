# coding: utf-8
import numpy as np 

class StandardScaler: # 类的首字母大写
    """"""
    def __init__(self):
        # 把后面遇到的私有变量初始化一遍？
        # self._mean = None
        self.mean_ = None
        # self._std = None
        self.scale_ = None # 拼写不要错了

    # 魅的，都在聊天，有啥好聊的
    def fit(self, X_train):
    # def fit(self, X):
        # assert 这里有个维度限制，但是我不知道是否有大于2维的存在？
        assert X_train.ndim == 2, "The dimension of X_train must be 2"

        # self._mean = np.mean(X_train) 
        # self._std = np.std(X_train)
        # 生成表达式的方式，注意for循环是在range(X_train.shape[1])里面的
        # self._mean = [np.mean(X_train[:,i] for i in X_train.shape[1])]
        # self._std = [np.std(X_train[:,i]) for i in X_train.shpae[1]]
        # 得到的是一个行向量
        # self._mean = [np.mean(X_train[:,i] for i in range(X_train.shape[1]))]
        # self._std = [np.std(X_train[:,i]) for i in range(X_train.shpae[1])]
        # 列表我们利用了python生成表达式的优势，同时类型退化到了python的列表类型，所以还要类型转换一下
        self.mean_ = np.array([np.mean(X_train[:,i]) for i in range(X_train.shape[1])])
        # self._std = np.array([np.std(X_train[:,i]) for i in range(X_train.shpae[1])])
        self.scale_ = np.array([np.std(X_train[:,i]) for i in range(X_train.shape[1])]) # shape别拼写错了
        return self # 遵守sklearn规范
    
    def transform(self, X_train):
    # def transform(self, X):
        assert X_train.ndim == 2, "The dimension of X_train must be 2"
        # 过好这一生太难了，总是要有前提做assert
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        # return (X_train - self.mean_) / self.scale_
        # 不能像上面这一行这么搞，因为是对每一列操作的，不要纠结这么搞是什么意思，反正这么搞很难理解
        resX = np.empty(shape=X_train.shape, dtype=float) # 因为均值方差都是float，所以需要往精度高的地方靠
        # 对于表达式长的用for循环而不用生成表达式
        for col in range(X_train.shape[1]):
            resX[:,col] = (X_train[:,col] - self.mean_[col]) / self.scale_[col]
        return resX

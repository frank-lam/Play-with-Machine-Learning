# self.intercept_
# self.coef_
# self._theta
# self.hstack()
from .metrics import r2_score
import numpy as np 

class LinearRegression:
    def __init__(self): # 构造函数
        # 初始化变量
        self.interception_ = None
        self.coef_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 得到正规化的theta值
        # np.ones((h, v))
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train);

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self
    
    # def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):
    # def fit_gd(X_train, y_train, initial_theta, eta, n_iters = 1e4):
    # def fit_gd(self, X_train, y_train, initial_theta, eta = 0.01, n_iters = 1e4):
    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
            
        # 应该把所有的函数定义都封装进去
        def J(theta, X_b, y): # y是实际值，X_b.dot(theta)得到的是线性回归模型的值
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(X_b)
            except OverflowError:
                return float('inf')

        def dJ(theta, X_b, y): # 为什么之前不要X_b和y呢，因为之前就没有模型啊，现在的话每一步都依赖特征啊
            # res = np.empty(len(theta)) # 开辟空间
            # res[0] = np.sum(X_b.dot(theta)-y) # 对每一个样本而言，np.sum对每一个样本行求和
            # # for i in range(len(y)):
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta)-y).dot(X_b[:, i]) # 取出第i个特征列
            # return res * 2 / len(X_b) # len(X_b)==len(y)
            return 2 / len(X_b) * X_b.T.dot(X_b.dot(theta)-y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):
            # 梯度下降法就是从一个初值出发
            theta = initial_theta # 这里的参数是一个向量了
            # 因为theta现在是高维向量了，无法绘制，所以跟踪theta_history也没意义了
            # 循环变化并比较是否到达极值点
            # while true: # 由于可能参数不恰当以及有限的计算资源，我们不应该使用while true，除非肯定可以结束
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                # 计算新的theta
                # theta = theta - eta * dJ(theta, X_b, y)
                theta = theta - eta * gradient
                # 比较新的值
                if np.abs(J(theta, X_b, y)-J(last_theta, X_b, y)) < epsilon:
                    break
                i_iter += 1
            return theta   

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = initial_theta # 这里的参数是一个向量了
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        # i_iter = 0
        # while i_iter < n_iters:
        #     gradient = dJ(self._theta, X_b, y)
        #     last_theta = self._theta
        #     self._theta = self._theta - eta * gradient
        #     if np.abs(J(self._theta, X_b, y)-J(last_theta, X_b, y)) < epsilon:
        #         break
        #     i_iter += 1
        return self

    # n_iters表示整体对我们的样本要看几轮
    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """根据训练数据集X_train，y_train，使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def J(theta, X_b, y): # 计算导数才是真正困难的地方
            try: # 这个不会出现inf的现象，但是会有OverflowError的可能
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except OverflowError:
                return float('inf')

        def dJ_sgd(theta, X_b_i, y_i): # 每次只要取一个样本
            # 返回的是随机梯度函数，使用的输入样本是准备好的
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.
        # 随机梯度下降法
        # initial_theta不是超参数，所以不要再fit_sgd那里传入，也就是不需要外部修改
        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            
            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b) # 得到了样本数

            # for cur_iter in range(n_iters): # 从0开始索引
            # for cur_iter in range(n_iters*m):
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                # indexes = range(m) # m是样本的个数，旁边一对小年轻相亲的，开始女方还拘谨，后来也健谈了
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    # rand_i = np.random.randint(m)
                    # gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
                    # theta = theta - learning_rate(cur_iter) * gradient
                    gradient = dJ_sgd(theta, X_b[i], y[i]) # 现在不要随机索引了
                    theta = theta - learning_rate(cur_iter * m + i) * gradient
                    # print(J(theta, X_b, y)) # 计算随机梯度下降法的值

            return theta
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1]) # 为什么随机地取theta
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self

    # def predict(self, X_test, y_test):
    def predict(self, X_predict):
        """"""
        # must fit_normal before predict
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict" # "must fit_normal before predict"
        # the size of X_test must be equal to the size of y_test
        assert X_predict.shape[1] == len(self.coef_), \
            "the size of X_test must be equal to the size of y_test"
        
        # X_b = np.hstack(np.ones(len(X_test), 1))
        # 应该传递列表进去
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        y_predict = X_b.dot(self._theta)
        return y_predict

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict) # 调用 metrics 里的函数

    def __repr__(self):
        return "LinearRegression()"

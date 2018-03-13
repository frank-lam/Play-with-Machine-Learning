import numpy as np 

class PCA:
    def __init__(self, n_components): # 为什么没有传入输入数据集你？
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None # 不是外界传来的，但是需要给外界使用的
    # 算出主成分，其他函数就能利用它做降维处理
    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_componnets must not be greater than the feature number of X"
        
        def demean(X):
            return X - np.mean(X, axis=0)
        
        def f(w, X): # 目标函数
            return np.sum((X.dot(w) ** 2)) / len(X)
        
        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8): # 冒号
           
            # 因为initial_w并不是一个单位向量啊
            w = direction(initial_w) # 得到单位向量
            cur_iter = 0
           
            # 梯度上升法求得目标函数的最值问题
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient # 梯度上升
                w = direction(w) # 总之呢要使用w作为参数的时候要单位化
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break 

                cur_iter += 1

            return w

        X_pca = demean(X) # 首先是均值归一化
        # 新坐标系矩阵：主成分矩阵
        self.components_ = np.empty(shape=(self.n_components, X.shape[1])) # 元组，坐标轴的表示依然是X.shape[1]
        for i in range(self.n_components):
            # w是一个二维的向量
            initial_w = np.random.random(X_pca.shape[1]) # 应该和数据集的特征维度一致
            w = first_component(X_pca, initial_w, eta, n_iters)

            # self.n_components[i, :] = w # 一个主成分方向向量占据一行
            self.components_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        
        return self

    def transform(self, X): # 从高维数据映射回低维数据
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1] # 样本个数是随意的，样本特征和成分特征也是一致的

        # return X.dot(self.n_components.T)
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
    
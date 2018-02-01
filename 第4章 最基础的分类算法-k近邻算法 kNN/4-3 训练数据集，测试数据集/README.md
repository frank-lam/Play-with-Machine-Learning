## 参考资料
[A sample Python project](https://github.com/pypa/sampleproject)
[python 怎么引入上上级目录的文件啊？](https://www.v2ex.com/t/163653)
[How can I see function arguments in IPython Notebook Server 3?](https://stackoverflow.com/questions/30855169/how-can-i-see-function-arguments-in-ipython-notebook-server-3)


## 学习目标
判断机器学习算法的性能

## 学习笔记
全部训练数据都用来计算生产用的模型

问题：
模型很差怎么办？会造成真实损失，在投放真实环境前无法验证有效性。
难以拿到label？就没办法用KNN来改进了，比如信用评级是用在没有label的客户身上的，这是有风险的
训练和测试数据集的分离
测试数据可以用来验证模型的训练效果如何？
可以改进投放到真实环境的损失？
name到底测试数据分几批呢？
train test split
训练数据-->模型
测试数据-------验证模型的性能或者是有效性
问题？后续分解

测试机器学习算法的目的其实是帮助选择更好的模型，如果不够好就重新选择其他的模型
所以放到一个model_selection.py的模块中


[ What is the difference between ndarray and array in numpy?](https://stackoverflow.com/questions/15879315/what-is-the-difference-between-ndarray-and-array-in-numpy)

## 学习状态
写代码写的比较急躁，因为连贯的代码变成了好多表达式的片段，感觉浪费了很多时间上，思路不连贯了，不得不重新再看视频，而已经花了很多时间了
注意转移注意力就不需要考虑思维分散问题
视频看3遍，第1遍理解，第2遍代码回忆，第3遍代码补充

## 学习方法
[decision tree用了cython，svm直接wrap了libsvm，但后期的代码基本原则都是优先Python实现，只有Python实在太慢的时候才会考虑用cython加速。
如果真是初学者为了学机器学习的话，与其看别人代码不如自己尝试实现，不一定要效率高，但至少得能用。这样比你一知半解的看完scikit-learn都有用。](https://www.zhihu.com/question/37217348#answer-23879869)

## 编程方法
无论是乱序还是提取都在索引上操作，从而得到最终数据的索引，不要中途对数据的次序进行变换，很容易就变成list类型了
train_indexes
test_indexes
shuffle_indexes
显示变量print(a)
sum(y_predict == y_test)

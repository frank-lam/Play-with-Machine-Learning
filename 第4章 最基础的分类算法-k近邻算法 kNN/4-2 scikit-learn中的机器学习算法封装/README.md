# 4-2 scikit-learn中的机器学习算法封装

## 学习目标

## 学习笔记


## 问答
问题：
私有成员变量名前应该加两条横线？
回答：
严格来说Python没有真正的私有成员变量。__variable的形式也可以通过_ClassName__variable的形式被外界调用。__variable比_variable多做了name mangling。

很多文章都会称_variable为“私有”。这个称呼不够严谨。Python官方称为internal use，并且特别强调，Python是没有“私有”概念的。或者说，_variable和__variable都可以叫“私有”。事实上，Python官方对这二者的区别强调是name mangling，而不是“私有”和“非私有”。这一点，可以参见Python官方PEP 8: https://www.python.org/dev/peps/pep-0008/#descriptive-naming-styles

习惯上，如果没有特殊的理由，应该使用_variable作为其他OO语言中的“私有”的概念，在Python语言中更多的是一个提示：这是一个不应该被外界触碰的变量。为什么没有特殊情况不应该使用__variable？这是一篇很好的文章说明这一点：http://python.net/~goodger/projects/pycon/2007/idiomatic/handout.html#naming 其中还指向了两个stackoverflow的链接，里面都有很好的讨论。

在这个课程中，我没有特别纠结这个称呼，毕竟不是Python课程。按照习俗，我称_variable为“私有”。随便找一个称_variable为私有的例子：https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc

最后，这个课程实现的算法代码风格和scikit-learn保持一致，可以参见scikit-learn的代码，其中对于“私有”变量（既不需要被外界调用的变量），统一使用_variable。随便给一个例子：https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/base.py 可以搜索看一下其中诸如：_fit_X，_tree等“私有”变量；_fit等“私有”方法。可以再搜索一下__，除了魔法方法，不会有自定义的__variable或者__method：）

问题：
关于本节课程开始的第三个assert 我认为应该是X_train.shape[1]==x.shape[1]
回答：
在这一小节这里的小x传进去的是一个向量也就是只代表一个样本。这个函数是通过X_train和y_train的信息判断一个样本x所属的类别。因此x的shape只有一个元素所以是x.shape[0]。

不过在这个课程的后面我们会按照sklearn的接口定义改成传入的是一个大X

在这个课程中编码规范上小写字母变量均表示一维向量大写字母变量均表示二维矩阵。所以我们叫大X_train因为是一个矩阵小y_train因为y是一个向量
## 学习方法
再看一遍
代码敲一遍
代码和讲解在笔记本里连线一下
用提交commit来表示学习进度，提交学习进度
有了想法就立即做，不要沾沾自喜
有了问题就问出来，问也是一种回答，因为要梳理了才能把要问的说清楚，所以有什么不明白的就问出来不管有没有答案
按时睡觉比量更重要
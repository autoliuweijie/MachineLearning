特征缩减技术(shrinkage): lasso和岭回归
===

作者：刘伟杰 日期：2015-12-01
参考：
    [1]《机器学习实战》 Peter Harrington  
    [2] scikit-learn官方手册


##1. 理论

1. 概述:

    通过对损失函数(即优化目标)加入惩罚项，使得训练求解参数过程中会考虑到系数的大小，通过设置缩减系数(惩罚系数)，会使得影响较小的特征的系数衰减到0，只保留重要的特征。常用的缩减系数方法有lasso(L1正则化)，岭回归(L2正则化)。

2. 缩减系数的目的

    2.1 消除噪声特征:

        如果模型考虑了一些不必要的特征，那么这些特征就算是噪声。噪声是没必要的，使得模型复杂，降低模型准确性，需要剔除。

    2.2 消除关联的特征:

        如果模型的特征空间中存在关联的特征，这会使得模型不适定，即模型参数会有多解。训练得到的只是其中一个解，这个解往往不能反映模型的真实情况，会误导模型的分析与理解。训练求解的模型参数受样本影响特别大，样本变化一点点，参数解就跳到另一组解去了。总之，模型是不稳定的。

3. 正则化：

    什么是正则化：

        对损失函数(目标函数)加入一个惩罚项，使得模型由多解变为更倾向其中一个解。

        在最小二乘法中，可以这样理解。XTX可能是不可逆的，通过加上正则项，迫使弱的特征的系数缩减为0.

    ![shrinkage](http://latex.codecogs.com/svg.latex?W=\(X^{T}X+\alphaI\)^{-1}X^{T}Y)

    3.1 lasso(L1正则化):

    与普通的线性回归相同，只是把损失函数换成:

    ![shrinkage](http://scikit-learn.org/stable/_images/math/5ff15825a85204658e3e5aa6e3b5952b8f709c27.png)

    这样，最优化求解参数过程，就会倾向选择系数比较小的解。当alpha增大，哪些无关的系数会率先缩减为0。但是L1正则化对相关特征的消除无能为力，仍然是不稳定的。

    3.2 岭回归(L2正则化):

    与普通的线性回归相同，只是把损失函数换成:

    ![shrinkage](http://scikit-learn.org/stable/_images/math/11f0787a645f4b5f2b810c0d00618785b58ff574.png)

    与lasso相比，这里的惩罚项加上了平方，这样就对于相关的两个特征，倾向与一个特别大，另一个接近0，起到了消除相关特征的效果。下图为随着alpha变化，系数的缩减效果。

    ![shrinkage](http://scikit-learn.org/stable/_images/plot_ridge_path_0011.png)



##2. 实现

1. 我的实现：

    包含L1与L2:

        https://github.com/autoliuweijie/MachineLearning/blob/master/regression/regression.py

2. scikit-learn:

    岭回归:

        >>> from sklearn import linear_model
        >>> clf = linear_model.Ridge (alpha = .5)
        >>> clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
        Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
              normalize=False, random_state=None, solver='auto', tol=0.001)
        >>> clf.coef_
        array([ 0.34545455,  0.34545455])
        >>> clf.intercept_ 
        0.13636...

    通过交叉验证寻找最优的alpha:

        >>> from sklearn import linear_model
        >>> clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        >>> clf.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
        RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
            normalize=False)
        >>> clf.alpha_                                      
        0.1

    lasso:

        >>> from sklearn import linear_model
        >>> clf = linear_model.Lasso(alpha = 0.1)
        >>> clf.fit([[0, 0], [1, 1]], [0, 1])
        Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
           normalize=False, positive=False, precompute=False, random_state=None,
           selection='cyclic', tol=0.0001, warm_start=False)
        >>> clf.predict([[1, 1]])
        array([ 0.8])

####扩展：

更多关于特征选择: http://dataunion.org/14072.html

scikit-learn: http://scikit-learn.org/stable/modules/linear_model.html#lars-lasso














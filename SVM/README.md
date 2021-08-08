### 支持向量机(Support Vector Machine, SVM)

SVM是一种二元分类模型，它依托的原理是如果能找到一条线，能够划分样本，并且训练点到线的间隔尽可能远，也就是所有点到超平面的最小值要最大，这样的分割线(一般叫超平面, Hyperplane)就是最好的分类方式，间隔叫作Margin。因此SVM的两个关键点就是：完全分类正确，及所有点到超平面的最小值要最大。

<img src="assets/20180214224342909.png" alt="img" style="zoom:50%;" />

### 线性SVM

设定超平面为$w^{T} x+b=0$, 在超平面上方我们定义$y=1$，下方定义为$y=-1$，则点$x$到超平面的**几何间隔**为:
$$
\frac{1}{\|w\|}\left|w^{T} x+b\right|
$$
我们需要判断分类是否正确，可以通过观察$y$(实际值)和$w^{T} x+b$(预测值)是否同号($y$只能为$1$或$-1$)，即我们可以定义**函数间隔**:
$$
\gamma^{\prime}=y\left(w^{T} x+b\right)
$$
现在我们可以初步定义优化函数为:
$$
\max \gamma=\frac{y\left(w^{T} x+b\right)}{\|w\|} \\
\text { s.t } y_{i}\left(w^{T} x_{i}+b\right)=\gamma^{\prime(i)} \geq \gamma^{\prime}(i=1,2, \ldots m)
$$
由于无论$w$和$b$怎么放缩，超平面的几何位置不变，所以我们可以设定最小的函数间隔$\gamma^{\prime}=1$，这样优化函数进一步简化为：
$$
\max \frac{1}{\|w\|} \\
\text { s.t } y_{i}\left(w^{T} x_{i}+b\right) \geq 1(i=1,2, \ldots m)
$$
由于$\frac{1}{\|w\|}$等同于$\frac{1}{2}\|w\|^{2}$的最小化，所以最终的优化函数可以表达为：
$$
\min \frac{1}{2}\|w\|^{2} \\
\text { s.t } y_{i}\left(w^{T} x_{i}+b\right) \geq 1(i=1,2, \ldots m)
$$
这样其实就是一个标准的二次规划(QP)问题，可以代入别的工具求解

### 对偶SVM



## Refs

台大林轩田机器学习技法

https://www.cnblogs.com/pinard/p/6097604.html
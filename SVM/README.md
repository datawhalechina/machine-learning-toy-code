## 支持向量机(Support Vector Machine, SVM)

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

我们通常会将线性SVM转化为对偶问题，主要解决在非线性转换之后维度变高二次规划问题求解困难的问题。在非线性分类问题中，我们一般作非线性转换$\mathbf{z}_{n}=\phi\left(\mathbf{x}_{n}\right)$，从而在$\mathbf{z}$空间里面进行线性分类，在非线性转换过程一般是一个升维的过程，将$\mathbf{z}$的维度一般设为$\tilde{d}$, 一般情况$\tilde{d}>>d$，(比如2维变5维，3维变19维等是一个非线性升高)，这样在高维空间内部，在二次规划问题中的就会有$\tilde{d}+1$个变量，那么权值维度同样也为$\tilde{d}+1$维，以及$m$个约束，此外$Q$ 矩阵维度会十分大，达到$\tilde{d}^2$，因此，一旦$\tilde{d}$变大了， 二次规划问题求解就会变得很困难。

我们引入拉格朗日乘子来转化为对偶问题，优化函数变为：
$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{m} \alpha_{i}\left[y_{i}\left(w^{T} x_{i}+b\right)-1\right] \text { 满足 } \alpha_{i} \geq 0
$$
优化目标为：
$$
\underbrace{\min }_{w, b} \underbrace{\max }_{\alpha_{i} \geq 0} L(w, b, \alpha)
$$
由于这个优化函数满足KTT条件，因此通过拉格朗日对偶将优化目标转为：
$$
\underbrace{\max }_{\alpha_{i} \geq 0} \underbrace{\min }_{w, b} L(w, b, \alpha)
$$
根据上式，我们可以先求出优化函数对于$w,b$的极小值，然后求拉格朗日乘子$\alpha$的最大值。

我们先求出优化函数对于$w,b$的极小值，即$\underbrace{\min }_{w, b} L(w, b, \alpha)$，只需要对$w,b$求偏导：
$$
\begin{gathered}
\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i} \\
\frac{\partial L}{\partial b}=0 \Rightarrow \sum_{i=1}^{m} \alpha_{i} y_{i}=0
\end{gathered}
$$
这样我们就知道了$w$和$\alpha$的关系，然后代入$\underbrace{\min }_{w, b} L(w, b, \alpha)$就可以消去$w$，定义：
$$
\psi(\alpha)=\underbrace{\min }_{w, b} L(w, b, \alpha)
$$
代入可得：
$$
\begin{aligned}
\psi(\alpha) &=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{m} \alpha_{i}\left[y_{i}\left(w^{T} x_{i}+b\right)-1\right] \\
&=\frac{1}{2} w^{T} w-\sum_{i=1}^{m} \alpha_{i} y_{i} w^{T} x_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i} b+\sum_{i=1}^{m} \alpha_{i} \\
&=\frac{1}{2} w^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i} w^{T} x_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i} b+\sum_{i=1}^{m} \alpha_{i} \\
&=\frac{1}{2} w^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}-w^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i} b+\sum_{i=1}^{m} \alpha_{i} \\
&=-\frac{1}{2} w^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}-\sum_{i=1}^{m} \alpha_{i} y_{i} b+\sum_{i=1}^{m} \alpha_{i} \\
&=-\frac{1}{2} w^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}-b \sum_{i=1}^{m} \alpha_{i} y_{i}+\sum_{i=1}^{m} \alpha_{i} \\
&=-\frac{1}{2}\left(\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}\right)^{T}\left(\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}\right)-b \sum_{i=1}^{m} \alpha_{i} y_{i}+\sum_{i=1}^{m} \alpha_{i} \\
&=-\frac{1}{2} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}-b \sum_{i=1}^{m} \alpha_{i} y_{i}+\sum_{i=1}^{m} \alpha_{i} \\
&=-\frac{1}{2} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}^{T} \sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}+\sum_{i=1}^{m} \alpha_{i} \\
&=-\frac{1}{2} \sum_{i=1, j=1}^{m} \alpha_{i} y_{i} x_{i}^{T} \alpha_{j} y_{j} x_{j}+\sum_{i=1}^{m} \alpha_{i} \\
&=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1, j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}
\end{aligned}
$$
我们再代入到优化目标中：
$$
\begin{gathered}
\underbrace{\max }_{\alpha}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{m} \alpha_{i} \\
\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\
\alpha_{i} \geq 0 i=1,2, \ldots m
\end{gathered}
$$
去掉负号转为最小值:
$$
\begin{gathered}
\underbrace{\min }_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{m} \alpha_{i} \\
\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\
\alpha_{i} \geq 0 i=1,2, \ldots m
\end{gathered}
$$
此时一般用SMO算法求解$\alpha$对应的极小值$\alpha^*$，求得之后，我们就可以根据$w=\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}$得出$w^*$。

求$b$则麻烦一些，但可根据对于任意的支持向量$\left(x_{x}, y_{s}\right)$，都有:
$$
y_{s}\left(w^{T} x_{s}+b\right)=y_{s}\left(\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i}^{T} x_{s}+b\right)=1
$$
即只需求出支持向量代入即可。

### 软间隔SVM

前面讲到线性SVM一个关键点就是超平面必须完全分类所有点，但实际上一方面有些数据混入了异常点，导致本来线性可分变成了不可分，如下图一个橙色和蓝色的异常点导致我们无法用线性SVM。

![img](assets/1042406-20161125104106409-1177897648.png)

但还有一种情况是没有糟糕到那么不可分，如下图，本来如果不考虑异常点，SVM的超平面应该是如红色线所示，但由于有一个蓝色的异常点，导致我们学到的超平面是如粗虚线所示。

![img](assets/1042406-20161125104737206-364720074.png)

为解决这个问题，SVM引入了软间隔的方法。

回顾硬间隔的优化函数:
$$
\min \frac{1}{2}\|w\|^{2} \\
\text { s.t } y_{i}\left(w^{T} x_{i}+b\right) \geq 1(i=1,2, \ldots m)
$$
现在对每个样本引入一个**松弛变量**$\xi_{i} \geq 0$，使函数间隔加上松弛变量大于等于1，即:
$$
y_{i}\left(w \bullet x_{i}+b\right) \geq 1-\xi_{i}
$$


可以看到我们对样本到超平面的函数距离的要求放松了，之前是一定要大于等于1，现在只需要加上一个大于等于0的松弛变量能大于等于1就可以了。当然，松弛变量不能白加，这是有成本的，每一个松弛变量$\xi_{i}$, 对应了一个代价$\xi_{i}$，因此优化函数变为:
$$
\begin{gathered}
\min \frac{1}{2}\|w\|_{2}^{2}+C \sum_{i=1}^{m} \xi_{i} \\
\text { s.t. } \quad y_{i}\left(w^{T} x_{i}+b\right) \geq 1-\xi_{i} \quad(i=1,2, \ldots m) \\
\xi_{i} \geq 0 \quad(i=1,2, \ldots m)
\end{gathered}
$$
这里$C>0$为惩罚参数，$C$越大，对误分类的惩罚越大，$C$越小，对误分类的惩罚越小。

下面我们需要优化软间隔SVM优化函数，与线性SVM类似，引入拉格朗日乘子转为无约束问题：
$$
L(w, b, \xi, \alpha, \mu)=\frac{1}{2}\|w\|_{2}^{2}+C \sum_{i=1}^{m} \xi_{i}-\sum_{i=1}^{m} \alpha_{i}\left[y_{i}\left(w^{T} x_{i}+b\right)-1+\xi_{i}\right]-\sum_{i=1}^{m} \mu_{i} \xi_{i}
$$
需要优化的目标函数，根据KKT条件再经过拉格朗日对偶转为:
$$
\underbrace{\max }_{\alpha_{i} \geq 0, \mu_{i} \geq 0} \underbrace{\min }_{w, b, \xi} L(w, b, \alpha, \xi, \mu)
$$
(以下推导略过，具体可见[支持向量机原理(二) 线性支持向量机的软间隔最大化模型](https://www.cnblogs.com/pinard/p/6100722.html)，类似地首先求出$w, b, \xi$的极小值:
$$
\begin{gathered}
\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{m} \alpha_{i} y_{i} x_{i} \\
\frac{\partial L}{\partial b}=0 \Rightarrow \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\
\frac{\partial L}{\partial \xi}=0 \Rightarrow C-\alpha_{i}-\mu_{i}=0
\end{gathered}
$$


最后导出软间隔SVM的优化函数为:
$$
\begin{gathered}
\underbrace{\min }_{\alpha} \frac{1}{2} \sum_{i=1, j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}-\sum_{i=1}^{m} \alpha_{i} \\
\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\
0 \leq \alpha_{i} \leq C
\end{gathered}
$$
可以看到与硬间隔SVM相比，只是多了一个约束条件$0 \leq \alpha_{i} \leq C$。

### 核函数SVM

对于完全线性不可分的情况，我们可以将数据映射到高维，从而线性可分。回顾线性可分的SVM优化函数：
$$
 \begin{gathered}\underbrace{\min }_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{m} \alpha_{i} \\\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\\alpha_{i} \geq 0 i=1,2, \ldots m\end{gathered}
$$
上式低维特征仅仅以内积$𝑥_𝑖∙𝑥_𝑗$的形式出现，如果我们定义一个低维特征空间到高维特征空间的映射$\phi$，将所有特征映射到一个更高的维度，让数据线性可分，从而按照前面的方法求出超平面，即：
$$
\begin{gathered}
\underbrace{\min }_{\alpha} \frac{1}{2} \sum_{i=1, j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \phi\left(x_{i}\right) \cdot \phi\left(x_{j}\right)-\sum_{i=1}^{m} \alpha_{i} \\
\text { s.t. } \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\
0 \leq \alpha_{i} \leq C
\end{gathered}
$$
但是这样我们需要求出内积$\phi\left(x_{i}\right) \cdot \phi\left(x_{j}\right)$，这样类似于二次规划那样会引入一个$\tilde{d}$维空间，从而发生维度爆炸影响计算速度。

为此我们可以引入核函数，设$\phi$是一个从低维的输入空间$\chi$（欧式空间的子集或者离散集合）到高维的希尔伯特空间的$\mathcal{H}$映射，如果存在函数$K(x, x')$，对于任意$x, x' \in \chi $，都有：
$$
K(x, x')=\phi(x) \bullet \phi(x')
$$
咋看跟上面没什么区别，但实际上核函数计算都是在低维空间下进行的，例如对于
$$
\Phi(\mathbf{x})=\left(1, x_{1}, x_{2}, \ldots, x_{d}, x_{1}^{2}, x_{1} x_{2}, \ldots, x_{1} x_{d}, x_{2} x_{1}, x_{2}^{2}, \ldots, x_{2} x_{d}, \ldots, x_{d}^{2}\right)
$$
我们得到:
$$
K_{\Phi}\left(x, x^{\prime}\right)=1+\left(x^{T} x^{\prime}\right)+\left(x^{T} x^{\prime}\right)^{2}
$$
这样看只需要计算低维空间的内积就行了。

常见的核函数有四种：

|    核函数     |                       公式                       | 备注                                    |
| :-----------: | :----------------------------------------------: | --------------------------------------- |
|  线性核函数   |             $K(x, x')=x \bullet x'$              | 其实就是线性可分的SVM                   |
| 多项式核函数  |      $K(x, x')=(\gamma x \bullet x'+r)^{d}$      | 其中$\gamma,r,d$都需要自己调参定义      |
|  高斯核函数   | $K(x, x')=\exp \left(-\gamma||x-x'||^{2}\right)$ | 最主流的核函数，在SVM中也叫径向基核函数 |
| Sigmoid核函数 |      $K(x, x')=\tanh (\gamma x \cdot x'+r)$      | 也是线性不可分SVM常用的核函数之一       |

下图是高斯核函数在不同参数下的分类效果:

<img src="assets/image-20210809104104109.png" alt="image-20210809104104109" style="zoom:50%;" />

可以看到原来线性SVM下超平面是一条直线，映射到高维可以较自由地定义位置形状。

## Refs

西瓜书

台大林轩田机器学习技法

[支持向量机原理(二) 线性支持向量机的软间隔最大化模型](https://www.cnblogs.com/pinard/p/6100722.html)

https://shomy.top/2017/02/17/svm-02-dual/



[林轩田机器学习笔记](https://wizardforcel.gitbooks.io/ntu-hsuantienlin-ml/content/21.html)
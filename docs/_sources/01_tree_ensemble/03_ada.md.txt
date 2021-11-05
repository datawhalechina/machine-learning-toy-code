---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Part C: 自适应提升法

## 1. Adaboost概述

Adaboost的全称是Adaptive Boosting，其含义为自适应提升算法。其中，自适应是指Adaboost会根据本轮样本的误差结果来分配下一轮模型训练时样本在模型中的相对权重，即对错误的或偏差大的样本适度“重视”，对正确的或偏差小的样本适度“放松”，这里的“重视”和“放松”具体体现在了Adaboost的损失函数设计以及样本权重的更新策略。本课我们将介绍Adaboost处理分类和回归任务的算法原理，包括SAMME算法、SAMME.R算法和Adaboost.R2算法。

## 2. 分类损失

对于$K$分类问题而言，当样本标签$\mathbf{y}=[y_1,...,y_K]^T$的类别$S(\mathbf{y})$为第$k$类（$k=1,...,K$）时，标签$\mathbf{y}$的第$i$个（$i=1,...,K$）元素$y_i$满足

$$
y_i=\left\{
\begin{aligned}
&1,\quad &{\rm if}\ i=k\\
&-\frac{1}{K-1},\quad &{\rm if}\ i\neq k
\end{aligned}
\right.
$$

````{margin}
【练习】假设有一个3分类问题，标签类别为第2类，模型输出的类别标签为[-0.1,-0.3,0.4]，请计算对应的指数损失。
````

设模型的输出结果为$\mathbf{f}=[f_1,...,f_K]^T$，则记损失函数为

$$
L(\mathbf{y},\mathbf{f})=\exp(-\frac{\mathbf{y}^T\mathbf{f}}{K})
$$

由于对任意的向量$a\mathbb{1},a\in \mathbb{R}$有

$$
L(\mathbf{y}, \mathbf{f}+a\mathbb{1})=\exp(-\frac{\mathbf{y}^T\mathbf{f}}{K}-\frac{a\mathbf{y}^T\mathbb{1}}{K})=\exp(-\frac{\mathbf{y}^T\mathbf{f}}{K})=L(\mathbf{y}, \mathbf{f})
$$

因此为了保证$\mathbf{f}$的可估性，我们需要作出约束假设，此处选择对称约束条件

$$
f_1+f_2+...+f_K=0
$$

当模型训练完毕后，等价于我们估计出了给定特征$\mathbf{x}$情况下$\mathbf{y}$属于各个类别的后验概率，在预测时对于一个新的$\mathbf{x}$我们总是希望$L(\mathbf{y},\mathbf{f})$越小越好，但此时$y$是随机变量，因此我们希望损失的后验期望$\mathbb{E}_{\mathbf{y}\vert\mathbf{x}}$较小。从概率角度而言，一个设计良好的分类问题损失函数应当保证模型在损失后验期望达到最小时的输出结果$k^*$是使得后验概率达到最大的类别$\mathop{\arg\max}_k P(S(\mathbf{y})=k\vert \mathbf{x})$，这个条件被称为贝叶斯最优决策条件。在本问题下，$k^* =\mathop{\arg\max}_kf_k^*(\mathbf{x})$，即选取$\mathbf{f}$中分量最大的位置对应类别输出。

想要验证指数损失是否满足贝叶斯最优决策条件，我们只需证明当损失后验期望达到最小时有

$$\mathop{\arg\max}_kf_k^*(\mathbf{x})=\mathop{\arg\max}_k P(S(\mathbf{y})=k\vert \mathbf{x})$$

由于此处约束为$\mathbf{f}$分量和为$0$，因此可以通过拉格朗日乘子法求解。写出拉格朗日函数为

$$
\begin{aligned}
Lag(\mathbf{f}, \lambda) &= \mathbb{E}_{\mathbf{y}\vert\mathbf{x}}\exp(-\frac{\mathbf{y}^T\mathbf{f}}{K}) + \lambda \sum_{i=1}^K f_i\\
&= \sum_{i=1}^K \left.\exp(-\frac{1}{K}\sum_{t=1}^Ky_tf_t)\right|_{S(\mathbf{y})=i}P(S(\mathbf{y})=i\vert \mathbf{x}) + \lambda \sum_{i=1}^K f_i\\
\end{aligned}
$$

当$S(\mathbf{y})=i$时，指数内的式子满足

$$
\begin{aligned}
-\frac{1}{K}\sum_{t=1}^Ky_tf_t &=-\frac{1}{K}[ f_i+\sum_{t\neq i}(-\frac{1}{K-1})f_t]\\
&=-\frac{1}{K}[f_i -\frac{-f_i}{K-1}]\\
&=-\frac{f_i}{K-1}
\end{aligned}
$$

此时有

$$
Lag(\mathbf{f}, \lambda) = \sum_{i=1}^K\exp(-\frac{f_i}{K-1}) P(S(\mathbf{y})=i\vert \mathbf{x}) + \lambda \sum_{i=1}^K f_i
$$

对各变量$f_1,...,f_K,\lambda$求偏导数结果式置0后，联立可解得

$$
f_k^*=(K-1)[\log P(S(\mathbf{y})=k\vert \mathbf{x})-\frac{1}{K}\sum_{i=1}^K\log P(S(\mathbf{y})=i\vert \mathbf{x})],k=1,...,K
$$

此时有$\mathop{\arg\max}_kf_k^*(\mathbf{x})=\mathop{\arg\max}_k P(S(\mathbf{y})=k\vert \mathbf{x})$，故选择指数损失能够满足贝叶斯最优决策条件。

## 3. SAMME

SAMME算法的全称是$\rm{\textbf{S}tagewise\,\textbf{A}dditive\,\textbf{M}odeling\, using\, a\,\textbf{M}ulticlass\,\textbf{E}xponential\, loss\, function}$，它假定模型的总输出$\mathbf{f}$具有$\mathbf{f}^{(M)}(\mathbf{x})=\sum_{m=1}^M \beta^{(m)} \mathbf{b}^{(m)}(\mathbf{x})$的形式。其中，$M$是模型的总迭代轮数，$\beta^{(m)}\in \mathbb{R^+}$是每轮模型的加权系数，$\mathbf{b}^{(m)}(\mathbf{x}) \in\mathbb{R}^K$是基模型$G$输出类别的标签向量。设样本的标签类别为$k$，当基模型预测的样本类别结果为$k'$时，记

$$
b^{(m)}_{k'}=\left\{
\begin{aligned}
&1,\quad &{\rm if}\ k'=k\\
&-\frac{1}{k-1},\quad &{\rm if}\ k'\neq k
\end{aligned}
\right.
$$

对于第$m$轮迭代而言，上一轮的模型输出为$\mathbf{f}^{(m-1)}(\mathbf{x})$，本轮需要优化得到的$\beta^{*(m)}$和$\mathbf{b}^{*(m)}$满足

$$
(\beta^{*(m)}, \mathbf{b}^{*(m)})=\mathop{\arg\min}_{\beta^{(m)}, \mathbf{b}^{(m)}}\sum_{i=1}^n L(\mathbf{y}_i, \mathbf{f}^{(m-1)}(\mathbf{x}_i)+\beta^{(m)}\mathbf{b}^{(m)}(\mathbf{x}_i))
$$

由于$\mathbf{f}^{(m-1)}(\mathbf{x}_i)$在第$m$轮为常数，记

$$
w_i=\exp(-\frac{1}{K}\mathbf{y}_i^T\mathbf{f}^{(m-1)}(\mathbf{x}_i))
$$

此时有

$$
(\beta^{*(m)}, \mathbf{b}^{*(m)})=\mathop{\arg\min}_{\beta^{(m)}, \mathbf{b}^{(m)}}\sum_{i=1}^n w_i\exp(-\frac{1}{K}\beta^{(m)}\mathbf{y}_i^T\mathbf{b}^{(m)}(\mathbf{x}_i))
$$

````{margin}
【练习】左侧公式的第二个等号是由于当样本分类正确时，$\mathbf{y}^T\mathbf{b}=\frac{K}{K-1}$，当样本分类错误时，$\mathbf{y}^T\mathbf{b}=-\frac{K}{(K-1)^2}$，请说明原因。
````

设当轮预测正确的样本索引集合为$T$，则损失可表示为

$$
\begin{aligned}
\tilde{L}(\beta^{(m)}, \mathbf{b}^{(m)})&=\sum_{i=1}^n w_i\exp(-\frac{1}{K}\beta^{(m)}\mathbf{y}_i^T\mathbf{b}^{(m)}(\mathbf{x}_i)) \\
&= \sum_{i\in T}w_i\exp[-\frac{\beta^{m}}{K-1}]+\sum_{i \notin T}w_i\exp[\frac{\beta^{(m)}}{(K-1)^2}] \\
&= \sum_{i\in T}w_i\exp[-\frac{\beta^{m}}{K-1}] +\sum_{i\notin T}w_i\exp[-\frac{\beta^{m}}{K-1}] \\
&\qquad - \sum_{i\notin T}w_i\exp[-\frac{\beta^{m}}{K-1}] +\sum_{i \notin T}w_i\exp[\frac{\beta^{(m)}}{(K-1)^2}]\\
&=\exp[-\frac{\beta^{(m)}}{K-1}]\sum_{i=1}^nw_i + \{ \exp[\frac{\beta^{(m)}}{(K-1)^2}]-\exp[-\frac{\beta^{(m)}}{K-1}] \}\sum_{i=1}^nw_i\mathbb{I}_{\{i\notin T\}}
\end{aligned}
$$

注意到$\mathbf{b}^{(m)}$仅与$\sum_{i=1}w_i\mathbb{I}_{\{i\notin T\}}$有关（因为基学习器的好坏控制了样本是否能够正确预测），且此项前的系数非负（因为$\beta^{(m)}$非负），因此得到

$$
\mathbf{b}^{*(m)}=\mathop{\arg\min}_{\mathbf{b}^{(m)}}\sum_{i=1}^n w_i\mathbb{I}_{\{i\notin T\}}
$$

在得到$\mathbf{b}^{*(m)}$后，通过求$\tilde{L}$关于$\beta^{(m)}$的导数并令之为0可解得

$$
\beta^{*(m)}=\frac{(K-1)^2}{K}[\log\frac{1-err^{(m)}}{err^{(m)}}+\log(K-1)]
$$

其中，样本的加权错误率为

$$
err^{(m)}=\sum_{i=1}^n\frac{w_i}{\sum_{i=1}^nw_i}\mathbb{I}_{\{i\notin T\}}
$$

样本$\mathbf{x}_i$在第$m$轮的预测类别为$k_i^*=\mathop{\arg\max}_{k} \mathbf{f}^{(m)}(\mathbf{x}_i)$，其中

$$
\mathbf{f}^{(m)}(\mathbf{x}_i)=\mathbf{f}^{(m-1)}(\mathbf{x}_i)+\beta^{*(m)}\mathbf{b}^{*(m)}(\mathbf{x}_i)
$$

````{margin}
【练习】对公式进行化简，写出$K=2$时的SAMME算法流程，并与李航《统计学习方法》一书中所述的Adaboost二分类算法对比是否一致。
````

将上述算法过程总结伪代码如下：

```{figure} ../_static/ada_algo1.png
---
width: 440px
align: center
---
```

事实上，我们还能通过一些多分类的性质来改写算法的局部实现，使得一些变量前的系数得到简化。记

$$
\alpha^{*(m)}=\log\frac{1-err^{(m)}}{err^{(m)}}+\log(K-1)
$$

此时，$w_i$每轮会被更新为

$$
\begin{aligned}
w^{new}_i&=w_i\exp(-\frac{1}{K}\beta^{*(m)}\mathbf{y}_i^T\mathbf{b}^{*(m)}(\mathbf{x}_i))\\
&=w_i\exp(-\frac{(K-1)^2}{K^2}\alpha^{*(m)}\mathbf{y}_i^T\mathbf{b}^{*(m)}(\mathbf{x}_i))
\end{aligned}
$$

当样本分类正确时，$\mathbf{y}_i^T\mathbf{b}^{*(m)}(\mathbf{x}_i)=\frac{K}{K-1}$，即

$$
w^{new}_i=w_i\cdot\exp[\frac{1-K}{K}\alpha^{*(m)}]
$$

当样本分类错误时，$\mathbf{y}_i^T\mathbf{b}^{*(m)}(\mathbf{x}_i)=-\frac{K}{(K-1)^2}$，即

$$
w^{new}_i=w_i\cdot\exp[\frac{1}{K}\alpha^{*(m)}]
$$

从而可以利用示性函数$\mathbb{1}_{\{i\notin T\}}$来统一表示$w_i$更新的两类结果：

$$
w^{new}_i=w_i\cdot\exp[\frac{1-K}{K}\alpha^{*(m)}]\exp(\alpha^{*(m)}\mathbb{1}_{\{i\notin T\}})
$$

````{margin}
【练习】在sklearn源码中找出算法流程中每一行对应的处理代码。
````

````{margin}
【练习】算法2第12行中给出了$\mathbf{f}$输出的迭代方案，但在sklearn包的实现中使用了$\mathbb{I}_{\{G^*(\mathbf{x})=S(\mathbf{y})\}}$来代替$\mathbf{b}^{*(m)}(\mathbf{x})$。请根据本文的实现，对sklearn包的源码进行修改并构造一个例子来比较它们的输出是否会不同。（提示：修改AdaboostClassifier类中的decision\_function函数）
````

````{margin}
【练习】请解释将$\beta^{*(m)}$替换为$\alpha^{*(m)}$不会改变输出类别的原因。
````

对$\mathbf{w}$进行归一化操作后，不会对下一轮算法1中$G^*$和$err^{(m)}$的结果产生任何影响。同时，如果把算法1第12行的$\beta^{*(m)}$替换为$\alpha^{*(m)}$，最后的预测结果（即取$\arg\max$产生的类别）也不会产生任何变化。

由于$\exp[\frac{1-K}{K}\alpha^{*(m)}]$是样本公共项，故我们可以每次都利用

$$
\tilde{w}_i = w_i\cdot\exp(\alpha^{*(m)}\mathbb{1}_{\{i\notin T\}})
$$

来更新，而不影响归一化结果。此时，算法1的迭代循环可进行如下重写：

```{figure} ../_static/ada_algo2.png
---
width: 380px
align: center
---
```

## 4. SAMME.R

许多分类器都能够输出预测样本所属某一类别的概率，但是SAMME算法只能利用分类的标签信息，而不能利用这样的概率信息。SAMME.R算法通过损失近似的思想，将加权分类模型的概率输出信息与boosting方法相结合。SAMME.R中的字母“R”代表“Real”，意味着模型每轮迭代的输出为实数。

不同于SAMME在第$m$轮需要同时考虑得到最优的$\beta^{(m)}$和$\mathbf{b}^{(m)}$，SAMME.R将其统一为$\mathbf{h}^{(m)}\in \mathbb{R}^K$，它需要满足对称约束条件$\sum_{i=1}^K h_k=0$以保证可估性。此时，损失函数为

$$
L(\mathbf{h}^{(m)})=\exp[-\frac{1}{K}\mathbf{y}^T(\mathbf{f}^{(m-1)}(\mathbf{x})+\mathbf{h}^{(m)}(\mathbf{x}))]
$$

为了与概率联系，我们需对损失$L$的后验概率进行最小化，即

$$
\begin{aligned}
\mathbf{h}^{*(m)}&=\mathop{\arg\min}_{\mathbf{h}^{(m)}} \mathbb{E}_{\mathbf{y}\vert\mathbf{x}} [L\vert \mathbf{x}]\\
&=\mathop{\arg\min}_{\mathbf{h}^{(m)}} \mathbb{E}_{\mathbf{y}\vert\mathbf{x}}[ \exp[-\frac{1}{K}\mathbf{y}^T(\mathbf{f}^{(m-1)}(\mathbf{x})+\mathbf{h}^{(m)}(\mathbf{x}))]\vert \mathbf{x}]
\end{aligned}
$$

````{margin}
【练习】请说明左式第三个等号为何成立。
````

设样本$\mathbf{y}$对应的标签为$S(\mathbf{y})$，则

$$
\begin{aligned}
\mathbb{E}_{\mathbf{y}\vert\mathbf{x}} [L\vert \mathbf{x}] &= \mathbb{E}_{\mathbf{y}\vert\mathbf{x}}[ \exp[-\frac{1}{K}\mathbf{y}^T\mathbf{f}^{(m-1)}(\mathbf{x})]
\exp[-\frac{1}{K}\mathbf{y}^T\mathbf{h}^{(m)}(\mathbf{x})]]\vert \mathbf{x}]\\
&=\sum_{k=1}^K\left. [\exp[-\frac{1}{K}\mathbf{y}^T\mathbf{f}^{(m-1)}(\mathbf{x})]\exp[-\frac{1}{K}\mathbf{y}^T\mathbf{h}^{(m)}(\mathbf{x})]]\right|_{S(\mathbf{y})=k}P(S(\mathbf{y})=k\vert \mathbf{x})\\
&=\sum_{k=1}^K \left.[\exp[-\frac{1}{K}\mathbf{y}^T\mathbf{f}^{(m-1)}(\mathbf{x})]\right|_{S(\mathbf{y})=k}P(S(\mathbf{y})=k\vert \mathbf{x})]\exp(-\frac{h^{(m)}_k(\mathbf{x})}{K-1})
\end{aligned}
$$

记$w=\exp[-\frac{1}{K}\mathbf{y}^T\mathbf{f}^{(m-1)}(\mathbf{x})]$，则

$$
\mathbb{E}_{\mathbf{y}\vert\mathbf{x}} [L\vert \mathbf{x}] = \sum_{k=1}^K \left.w\right|_{S(\mathbf{y})=k}\cdot P(S(\mathbf{y})=k)\exp(-\frac{h^{(m)}_k(\mathbf{x})}{K-1})
$$

不难发现对于样本$\mathbf{y}$而言，越大的$w$意味着上一轮的模型结果越糟糕，此时负责预测$P(S(\mathbf{y})=k)$的基模型就要加大对该样本的重视程度以获得较小的损失。

但是，此时基模型本身是不带权重的，SAMME.R采用的近似方法是，考虑以$w$为权重的基模型$G$，用其输出$P_w(s(\mathbf{y})=k\vert \mathbf{x})$的概率值来代替$\left.w\right|_{S(\mathbf{y})=k}\cdot P(S(\mathbf{y})=k\vert \mathbf{x})$，这种行为合法的原因在于权重对于总体损失的惩罚方向是一致的，$G$通过权重$w$将原本作用于$L$的损失近似地“分配”给了基分类器的损失。

此时，损失函数近似为

$$
\mathbb{E}_{\mathbf{y}\vert\mathbf{x}} [L\vert \mathbf{x}] = \sum_{k=1}^K P_w(s(\mathbf{y})=k\vert \mathbf{x})\exp(-\frac{h^{(m)}_k(\mathbf{x})}{K-1})
$$

````{margin}
【练习】验证$h^*_{k'}$的求解结果。
````

由对称约束条件，结合拉格朗日乘子法可得

$$
h^{*(m)}_{k'}=(K-1)[\log P_w(S(\mathbf{y})=k'\vert \mathbf{x})-\frac{1}{K}\sum_{k=1}^K\log P(S(\mathbf{y})=k\vert \mathbf{x})]
$$

````{margin}
【练习】算法3的第14行给出了$w_i$的更新策略，请说明其合理性。
````

将上述算法过程总结伪代码如下：

```{figure} ../_static/ada_algo3.png
---
width: 580px
align: center
---
```

## 5. Adaboost.R2

利用权重重分配的思想，Adaboost还可以应用于处理回归问题。其中，Adaboost.R2算法是一种最常使用的实现。

设训练集特征和目标分别为$\mathbf{X}=(\mathbf{x}_1, ..., \mathbf{x}_n)$和$\mathbf{y}=(y_1,...,y_n)$，权重$\mathbf{w}$初始化为$(w_1,...,w_n)$。在第$m$轮时，根据权重训练基预测器得到$G^*$，计算每个样本的相对误差

$$
e_{i}=\frac{\vert y_i-G^*(\mathbf{x}_i)\vert}{\max_i \vert y_i-G^*(\mathbf{x}_i)\vert}
$$

设样本的加权相对误差率为$E^{(m)}=\sum_{i=1}^n w_ie_i$，则相对误差率与正确率的比值为$\beta^{(m)}=\frac{E^{(m)}}{1-E^{(m)}}$，即预测器权重$\alpha^{(m)}=\log \frac{1}{\beta^{(m)}}$。

更新权重$w_i$为$w_{i}[\beta^{(m)}]^{1-e_{i}}$，权重在归一化后进入下一轮训练，由此可如下写出训练算法：

```{figure} ../_static/ada_algo4.png
---
width: 380px
align: center
---
```

````{margin}
【练习】请结合加权中位数的定义解决以下问题：
  - 当满足什么条件时，Adaboost.R2的输出结果恰为每个基预测器输出值的中位数？
  - Adaboost.R2模型对测试样本的预测输出值是否一定会属于$M$个分类器中的一个输出结果？若是请说明理由，若不一定请给出反例。
  - 设$k\in \{y_1,...,y_M\}$，记$k$两侧（即大于或小于$k$）的样本集合对应的权重集合为$W^+$和$W^-$，证明使这两个集合元素之和差值最小的$k$就是Adaboost.R2输出的$y$。
  - 相对于普通中位数，加权中位数的输出结果鲁棒性更强，请结合公式说明理由。
````

在预测阶段，Adaboost.R2使用的是加权中位数算法。设每个基模型对某一个新测试样本的预测输出为$y_1,...,y_M$，基模型对应的预测器权重为$\alpha^{(1)},...,\alpha^{(M)}$，则Adaboost.R2的输出值为

$$
y=\inf \{ y\big| \sum_{m\in \{m\vert y_m\leq y\}}\alpha^{(m)} \geq 0.5 \sum_{m=1}^M\alpha^{(m)}\}
$$

## 知识回顾

1. 二分类问题下，Adaboost算法如何调节样本的权重？
2. 样本A在当轮分类错误，且样本B在当轮分类正确，请问在权重调整后，样本A的权重一定大于样本B吗？
3. 在处理分类问题时，Adaboost的损失函数是什么？请叙述其设计的合理性。
4. Adaboost如何处理回归问题？
5. 用已训练的Adaboost分类模型和回归模型来预测新样本的标签，请分别具体描述样本从输入到标签输出的流程。
6. 观看周志华老师的讲座视频[《Boosting 25年》](https://www.bilibili.com/video/BV1Cs411c7Zt)并谈谈体会。
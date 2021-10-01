#!/usr/bin/env python
# coding: utf-8

# # Part D: 梯度提升树（上）
# 
# ## 1. 用于回归的GBDT
# 
# 设数据集为$D=\{(X_1,y_1),...,(X_N,y_N)\}$，模型的损失函数为$L(y,\hat{y})$，现希望利用多棵回归决策树来进行模型集成：设第$m$轮时，已知前$m-1$轮中对第$i$个样本的集成输出为$F_{m-1}(X_i)$，则本轮的集成输出$\hat{y}_i$为
# 
# $$
# F_{m}(X_i)=F_{m-1}(X_i)+h_m(X_i)
# $$
# 
# 其中，$h_m$是使得当前轮损失$\sum_{i=1}^N L(y_i,\hat{y}_i)$达到最小的决策树模型。
# 
# ````{margin}
# 【练习】对于均方损失函数和绝对值损失函数，请分别求出模型的初始预测$F_{0}$。
# ````
# 
# 特别地，当$m=0$时，$F_{0}(X_i)=\arg\min_{\hat{y}} \sum_{i=1}^N L(y_i,\hat{y})$。
# 
# 
# 记第$m$轮的损失函数为
# 
# $$
# G(h_m) = \sum_{i=1}^NL(y_i, F_{m-1}(X_i)+h_m(X_i))
# $$
# 
# 令上述损失最小化不同于一般的参数优化问题，我们需要优化的并不是某一组参数，而是要在所有决策树模型组成的函数空间中，找到一个$h^*$使得$G(h^*)$最小。因此我们不妨这样思考：学习一个决策树模型等价于对数据集$\tilde{{D}}=\{(X_1,h^*(X_1)),...,(X_N,h^*(X_N))\}$进行拟合，设$w_i=h^*(X_I)$，$\textbf{w}=[w_1,...,w_N]$，此时的损失函数可改记为
# 
# $$
# G(\textbf{w})=\sum_{i=1}^NL(y_i, F_{m-1}(X_i)+w_i)
# $$
# 
# 由于只要我们获得最优的$\textbf{w}$，就能拟合出第$m$轮相应的回归树，此时一个函数空间的优化问题已经被转换为了参数空间的优化问题，即对于样本$i$而言，最优参数为
# 
# $$
# w_i=\arg\min_{w}L(y_i,F_{m-1}(X_i)+w)
# $$
# 
# 对于可微的损失函数$L$，由于当$\textbf{w}=\textbf{0}$时的损失就是第$m-1$轮预测产生的损失，因此我们只需要在$w_i=0$处进行一步梯度下降（若能保证合适的学习率大小）就能够获得使损失更小的$w^*_i$，而这个值正是我们决策树需要拟合的$h^*(X_i)$。
# 以损失函数$L(y,\hat{y})=\sqrt{\vert y-\hat{y}\vert}$为例，记残差为
# 
# $$
# r_i = y_i-F_{m-1}(X_i)
# $$
# 
# 则实际损失为
# 
# $$
# L(w_i)=\sqrt{\vert r_i-w_i\vert }
# $$
# 
# ````{margin}
# 【练习】给定了上一轮的预测结果$F_{m-1}(X_i)$和样本标签$y_i$，请计算使用平方损失时需要拟合的$w^*_i$。
# ````
# ````{margin}
# 【练习】当样本$i$计算得到的残差$r_i=0$时，本例中的函数在$w=0$处不可导，请问当前轮应当如何处理模型输出？
# ````
# 
# 根据在零点处的梯度下降可知：
# 
# $$
# \begin{aligned}
# w^*_i &= 0 - \left.\frac{\partial L}{\partial w} \right|_{w=0}\\
# &= -\frac{1}{2\sqrt{r_i}}sign(r_i)
# \end{aligned}
# $$
# 
# 为了缓解模型的过拟合现象，我们需要引入学习率参数$\eta$来控制每轮的学习速度，即获得了由$\textbf{w}^*$拟合的第m棵树$h^*$后，当前轮的输出结果为
# 
# $$
# \hat{y}_i=F_{m-1}(X_i)+\eta h^*_m(X_i)
# $$
# 
# 对于上述的梯度下降过程，还可以从另一个等价的角度来观察：若设当前轮模型预测的输出值为$\tilde{w}_i= F_{m-1}(X_i)+w_i$，求解的问题即为
# 
# $$
# \tilde{w}_i=\arg\min_{\tilde{w}} L(y_i, \tilde{w})
# $$
# 
# 由于当$\tilde{w}=F_{m-1}(X_i)$时，损失函数的值就是上一轮预测结果的损失值，因此只需将$L$在$\tilde{w}$在$\tilde{w}=F_{m-1}(X_i)$的位置进行梯度下降，此时当前轮的预测值应为
# 
# $$
# \tilde{w}^*_i=F_{m-1}(X_i)-\left.\frac{\partial L}{\partial \tilde{w}} \right|_{\tilde{w}=F_{m-1}(X_i)}
# $$
# 
# ````{margin}
# 【练习】除了梯度下降法之外，还可以使用[牛顿法](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)来逼近最值点。请叙述基于牛顿法的GBDT回归算法。
# ````
# 
# 从而当前轮学习器$h$需要拟合的目标值$w^*_i$为
# 
# $$
# \begin{aligned}
# w^*_i &= \tilde{w}_i-F_{m-1}(X_i)\\
# &=0-\frac{\partial L}{\partial w} \left.\frac{\partial w}{\partial \tilde{w}} \right|_{\tilde{w}=F_{m-1}(X_i)} \\
# &= 0-\left.\frac{\partial L}{\partial w} \right|_{\tilde{w}=F_{m-1}(X_i)} \\
# &=  0 - \left.\frac{\partial L}{\partial w} \right|_{w=0}
# \end{aligned}
# $$
# 
# 上述的结果与先前的梯度下降结果完全一致，事实上这两种观点在本质上没有任何区别，只是损失函数本身进行了平移，下图展示了它们之间的联系。
# 
# 
# ```{figure} ../_static/gbdt_pic1.png
# ---
# width: 600px
# align: center
# ---
# ```
# 
# ```{admonition} GBDT的特征重要性
# 在sklearn实现的GBDT中，特征重要性的计算方式与随机森林相同，即利用相对信息增益来度量单棵树上的各特征特征重要性，再通过对所有树产出的重要性得分进行简单平均来作为最终的特征重要性。
# ```
# 
# ## 2. 用于分类的GBDT
# 
# CART树能够同时处理分类问题和回归问题，但是对于多棵CART进行分类任务的集成时，我们并不能将树的预测结果直接进行类别加和。在GBDT中，我们仍然使用回归树来处理分类问题，那此时拟合的对象和流程又是什么呢？
# 
# 对于$K$分类问题，我们假设得到了$K$个得分$F_{1i},...,F_{Ki}$来代表样本$i$属于对应类别的相对可能性，那么在进行Softmax归一化后，就能够得到该样本属于这些类别的概率大小。其中，属于类别k的概率即为$\frac{e^{F_{ki}}}{\sum_{c=1}^Ke^{F_{ci}}}$。此时，我们就能够使用多分类的交叉熵函数来计算模型损失，设$\textbf{y}_i=[y_{1i},...,y_{Ki}]$为第$i$个样本的类别独热编码，记$\textbf{F}_i=[F_{1i},...,F_{Ki}]$，则该样本的损失为
# 
# $$
# L(\textbf{y}_i,\textbf{F}_i)=-\sum_{c=1}^K y_{ci}\log \frac{e^{F_{ci}}}{\sum_{\tilde{c}=1}^Ke^{F_{\tilde{c}i}}}
# $$
# 
# 上述的$K$个得分可以由$K$棵回归树通过集成学习得到，树的生长目标正是使得上述的损失最小化。记第$m$轮中$K$棵树对第$i$个样本输出的得分为$\textbf{h}^{(m)}_i=[h^{(m)}_{1i},...,h^{(m)}_{Ki}]$，则此时$\textbf{F}^{(m)}_i=\textbf{F}^{(m-1)}_i+\textbf{h}^{(m)}_i$。与GBDT处理回归问题的思路同理，只需要令损失函数$L(\textbf{y}_i,\textbf{F}_i)$在$\textbf{F}_i=\textbf{F}_i^{(m-1)}$处梯度下降即可：
# 
# $$
# \textbf{F}_i^{*(m)} = \textbf{F}_i^{(m-1)} - \left.\frac{\partial L}{\partial \textbf{F}_i} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}}
# $$
# 
# 我们需要计算第二项中每一个梯度元素，即
# 
# $$
# -\left.\frac{\partial L}{\partial \textbf{F}_i} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}}=[-\left.\frac{\partial L}{\partial F_{1i}} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}},...,-\left.\frac{\partial L}{\partial F_{Ki}} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}}]
# $$
# 
# 对于第$k$个元素有
# 
# $$
# \begin{aligned}
# -\left.\frac{\partial L}{\partial F_{ki}} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} &= \left.\frac{\partial}{\partial F_{ki}} \sum_{c=1}^K y_{ci}\log \frac{e^{F_{ci}}}{\sum_{\tilde{c}=1}^Ke^{F_{\tilde{c}i}}} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} \\
# &= \left.\frac{\partial}{\partial F_{ki}} \sum_{c=1}^K y_{ci} F_{ki} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} -  \left.\frac{\partial}{\partial F_{ki}} \sum_{c=1}^K y_{ci} \log [\sum_{\tilde{c}=1}^K e^{F_{\tilde{c}i}}] \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} \\
# &= y_{ki} -  \left.\frac{\partial}{\partial F_{ki}} \sum_{c=1}^K y_{ci} \log [\sum_{\tilde{c}=1}^K e^{F_{\tilde{c}i}}] \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}}
# \end{aligned}
# $$
# 
# 由于在上式的第二项里，$K$个$y_{ci}$中只有一个为$1$，且其余为$0$，从而得到
# 
# $$
# \begin{aligned}
# -\left.\frac{\partial L}{\partial F_{ki}} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} &= y_{ki} -  \left.\frac{\partial}{\partial F_{ki}} \log [\sum_{\tilde{c}=1}^K e^{F_{\tilde{c}i}}] \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} \\
# &= y_{ki} - \frac{e^{F^{(m-1)}_{ki}}}{\sum_{c=1}^K e^{F^{(m-1)}_{ci}}}
# \end{aligned}
# $$
# 
# 此时，$K$棵回归树的学习目标为：
# 
# $$
# \begin{aligned}
# \textbf{h}_i^{*(m)} &= \textbf{F}_i^{*(m)} - \textbf{F}_i^{(m-1)}\\
# &= - \left.\frac{\partial L}{\partial \textbf{F}_i} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} \\
# &= [y_{1i} - \frac{e^{F^{(m-1)}_{1i}}}{\sum_{c=1}^K e^{F^{(m-1)}_{ci}}},...,y_{Ki} - \frac{e^{F^{(m-1)}_{Ki}}}{\sum_{c=1}^K e^{F^{(m-1)}_{ci}}}]
# \end{aligned}
# $$
# 
# 同时，为了减缓模型的过拟合现象，模型在第$m$轮实际的$\textbf{F}^{*(m)}_i$为$\textbf{F}_i^{(m-1)}+\eta \textbf{h}_i^{*(m)}$。
# 
# 由于每一轮都需要进行$K$棵树的拟合，因此GBDT在处理多分类时的速度较慢。事实上，我们可以利用概率和为$1$的性质，将$K$次拟合减少至$K-1$次拟合，这在处理类别数较少的分类问题时，特别是在处理二分类问题时，是非常有用的。
# 
# 具体来说，此时我们需要$K-1$个得分，记为$F_{1i},...,F_{(K-1)i}$，则样本相应属于$K$个类别的概率值可表示为
# 
# $$
# [\frac{e^{F_{1i}}}{1+\sum_{c=1}^{K-1}e^{F_{ci}}},...,\frac{e^{F_{(K-1)i}}}{1+\sum_{c=1}^{K-1}e^{F_{ci}}},\frac{1}{1+\sum_{c=1}^{K-1}e^{F_{ci}}}]
# $$
# 
# 当$K\geq3$时，仍然使用独热编码来写出损失函数：
# 
# $$
# L(F_{1i},...,F_{(K-1)i})= y_{Ki}\log [1+\sum_{c=1}^{K-1}e^{F_{ci}}] -\sum_{c=1}^{K-1} y_{ci}\log \frac{e^{F_{ci}}}{\sum_{c=1}^Ke^{F_{ci}}} 
# $$
# 
# ````{margin}
# 【练习】请验证多分类负梯度的结果。
# ````
# 
# 类似地记$\textbf{F}_i=[F_{1i},...,F_{(K-1)i}]$，我们可以求出负梯度：
# 
# $$
# -\left.\frac{\partial L}{\partial F_{ki}} \right|_{\textbf{F}_i=\textbf{F}_i^{(m-1)}} = \left\{
# \begin{aligned}
# -\frac{e^{F^{(m-1)}_{ki}}}{\sum_{c=1}^{K-1} e^{F^{(m-1)}_{ci}}}  &\qquad y_{Ki}=1 \\
# y_{ki} - \frac{e^{F^{(m-1)}_{ki}}}{\sum_{c=1}^{K-1} e^{F^{(m-1)}_{ci}}} & \qquad y_{Ki}=0 \\
# \end{aligned}
# \right.
# $$
# 
# 当$K=2$时，不妨规定$y_i\in \{0,1\}$，此时损失函数可简化为
# 
# $$
# L(F_i) = - y_i\log \frac{e^{F_i}}{1+e^{F_i}} - (1-y_i)\log \frac{1}{1+e^{F_i}}
# $$
# 
# ````{margin}
# 【练习】请验证二分类负梯度的结果。
# ````
# 
# 负梯度为
# 
# $$
# -\left.\frac{\partial L}{\partial F_{i}} \right|_{F_i=F^{(m-1)}_i}=y_i-\frac{e^{F_i}}{1+e^{F_i}} 
# $$
# 
# 最后，我们可以使用各个类别在数据中的占比情况来初始化$\textbf{F}^{(0)}$。具体地说，设各类别比例为$p_1,...,p_K$（$K\geq3$），我们希望初始模型的参数$F^{(0)}_1,...,F^{(0)}_{K-1}$满足
# 
# $$
# [\frac{e^{F^{(0)}_{1i}}}{1+\sum_{c=1}^{K-1}e^{F^{(0)}_{ci}}},...,\frac{e^{F^{(0)}_{(K-1)i}}}{1+\sum_{c=1}^{K-1}e^{F^{(0)}_{ci}}},\frac{1}{1+\sum_{c=1}^{K-1}e^{F^{(0)}_{ci}}}] = [p_1,...,p_{K-1},p_K]
# $$
# 
# ````{margin}
# 【练习】设二分类数据集中正样本比例为$10\%$，请计算模型的初始参数$F^{(0)}$。
# ````
# 
# 对二分类（0-1分类）而言，设正负样本占比分别为$p_1$和$p_0$，则初始模型参数$F^{(0)}$应当满足
# 
# $$
# [ \frac{1}{1+e^{F^{(0)}_i}},\frac{e^{F^{(0)}_i}}{1+e^{F^{(0)}_i}}]=[p_0,p_1]
# $$
# 
# ## 3. GBDT中的并行策略
# 
# ## 代码实践
# 
# ## 算法实现
# 
# ## 知识回顾

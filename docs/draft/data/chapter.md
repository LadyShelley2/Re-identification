
前言
====

相关工作
========

稀疏表示
--------

信号稀疏表示的最初目的是为了用比香农定理更低的采样率来表示和压缩信号[@de2011embedding],就像离散余弦变换和小波变换等。在过去几年里，稀疏表示已经被用于很多信号处理，模式识别的实际应用中。例如，在信号的图片处理领域，稀疏表示被用于信号压缩和编码[@marcellin2000overview],图片去噪[@elad2006image],图片超像素图片超像素[@yang2008image]。在模式识别领域，稀疏表示被用于目标识别和分类任务[@huang2006sparse]
[@davenport2007smashed]。一些研究表明基于稀疏表示的分类器非常有效，在一些人脸数据库上的识别率甚至是目前的最高水平。

正如在离散余弦变换和小波变换中，原始信号可以用基函数表示，稀疏表示则是用一个称之为字典的超完备冗余的函数系统来代替原始基函数，并利用字典$\mathbf{X}$中最少的原子来表示一个原始信号$b$。

稀疏表示的思想如下[@殷俊2013核稀疏保持投影及生物特征识别应用]：假设样本集$X$有可以分为$k$类的共$m$个训练样本，$X_{k} = [x_{k1},x_{k2},\ldots,x_{kl_{k}}]\in R^{n\times l_{k}}$表示属于第$k$类的l个样本集,如果训练样本足够多，则属于第$k$类的任意一个样本$y\in R^{n}$可以近似地表示成此类的所有训练样本的线性组合
$$\mathbf{y}=h_{k1}\mathbf {x}_{k1}+h_{k2}\mathbf {x}_{k2}+\ldots+h_{kl_{k}}\mathbf {x}_{kl_{k}}$$
目标函数如下： $$\min_{s}\Vert{s}\Vert_{0}$$ $$s.t. \quad b=Xs$$
假如有$\mathbf{N}$

鉴别稀疏保持投影
----------------

稀疏保持投影[@qiao2010sparsity]将所有样本作为字典原子进行稀疏重构，鉴别稀疏保持投影则是增强了同类数据的稀疏表示权重[@马小虎2014基于鉴别稀疏保持嵌入的人脸识别算法],在保持局部结构的同时保持样本之间的总体信息，通过局部合并，以整体的方式发现和重建数据集的内在规律。并通过求解最小二乘问题来更新SPP中的稀疏权重矩阵，得到了一个更能真实反映鉴别信息的鉴别稀疏权重。将无监督的SPP转换成有监督的DSPE，充分利用训练样本中的标签信息。
样本训练集$X=[X_{1},X_{2},\ldots,X_{c}]$，其中，$X_{i}=[x_{i1}\ldots,X_{ik}]\in \mathbf{R}^{m\times k}$是类别标记为$label(i)$的样本集合，即第$i$类样本。通过最小二乘法获得最能反映类别鉴别信息的权重，目标函数如下：
$$\min_{t}\quad\Vert x_{ij} - \mathbf{X}_{i}t\Vert_{2}$$
$$s.t.\quad It = 1$$
其中$\mathbf{t}=[t_{1},t_{2},\ldots,t_{j-1},0,t_{j+1},\ldots,t_{k}]^{\mathbf{T}}$,第$j$个分量为0，即将任一向量利用同类中除自身以外的向量进行表示。经证明[@马小虎2014基于鉴别稀疏保持嵌入的人脸识别算法]，该方法具有良好的鲁棒性，具有旋转、平移、尺度不变的特性。
为了考虑样本集的全局几何结构，DSPE进一步考虑异类样本对原样本重构的影响。将$\hat{t}\in \mathbf{R}^k$扩展到$n$个分量$\hat{h}=[\vec{0},\ldots,\vec{0},\hat{t},\vec{0},\ldots,\vec{0}]\in \mathbf{R}^{n}$,即将除鉴别信息权重以外的系数设置为0.令$er=x_{ij}-X\hat{t}=x_{ij}-X\hat{h}$,对$er$进行在超完备字典$\hat{X}=[X_{1},X_{2},\ldots,X_{c}$上的稀疏表示，其中$X_{i}=\vec{0}$,将人脸样本由鉴别信息权重重构后得到的残差有其他类的所有样本进行稀疏重构：
$$\min_{s}\quad\Vert{s}\Vert_{1}$$
$$s.t.\quad\Vert{er-\hat{X}s}\Vert_{1}<\epsilon$$ $$Is = 0$$
作如下变换： $$s^{+} = \left\{ 
    \begin{array}{ll}
        s, & s>0\\
        0, & s\le{0}
    \end{array} \right.$$ $$s^{-} = \left\{ \begin{array}{ll}
        -s,&s<0\\
        0, &s\ge{0}
        \end{array}\right.$$
由于$s^{+},s^{-}$非负，$s=s^{+}+s^{-}$,$\Vert{s}\Vert_{1} = \Sigma_{i}(s_{i}^{+}+s_{i}^{-})$，目标函数转化为
$$\min_{s^{+}+s^{-}}\quad\sum_{i}(s_{i}^{+}+s_{i}^{-})$$
$$s.t.\quad \Vert{er-[\hat{X},-\hat{X}][s^{+},s^{-}]^{T}}\Vert < \epsilon$$
$$[I,-I][s^{+},s^{-}]^{T}=0$$
将优化结果$\hat{s}=[\hat{s_{1}},\ldots,\hat{s_{i-1}},\vec{0},\hat{s_{i+1},\ldots,\hat{s_{c}}}]$与$\hat{t}$串联，得鉴别稀疏权重$\hat{d}=\hat{s}+\hat{h}=[\hat{s_{1}},\ldots,\hat{s_{i-1}},\hat{t},\hat{s_{i+1}},\ldots,\hat{s_{c}}]\in \mathbf{R}^{n}$。由于$I\hat{t}=1$,$I\hat{s}=1$,故$I\hat{d}=1$,该鉴别稀疏权重能够保持旋转，平移，尺度不变的特性[@马小虎2014基于鉴别稀疏保持嵌入的人脸识别算法]与SPP相比较，DSPE中最小化同类样本重构数更加趋近于0，既增强了同类费劲林样本在重构中的作用，又削弱了异类伪近邻样本对原样本重构的影响，有利于保持同类样本相互靠近，异类样本相互远离的本质。当$er=0$时，鉴别稀疏权重便退化成了类似于NPE保持低维流形上同类数据近邻关系的重构权重，表明DSPE能很好地反映数据在低维流形上的分布，与此同时，对$er$的稀疏表示有利于加强算法的鲁棒性
DSPE的目标在于将数据间的鉴别稀疏重构特征进行保持并嵌入到低维流形上，因此，数学模型如下：
$$\min_{W}\quad\sum_{i=1}^{n}\Vert{W^{T}x_{i}-W^{T}X\hat{d_{i}}}\Vert^{2}$$
$$s.t. W^{T}XX^{T}W=1$$ 对目标函数进行变换，目标函数转换成
$$\max_{W}\quad W^{T}XMX^{T}W$$ $$s.t.\quad W^{T}XX^{T}W=1$$
其中，$M=D+D^{T}-D^{T}D$,目标函数可以通过求解广义特征值得解
$$XMX^{T}W=\lambda XX^{T}W$$
选取最大的$d$个特征值所对应的特征向量$\mathbf{a}_{i}$构成特征子空间，即可得到DSPE的线性降维映射$W_{DSPE}=[a_{1},a_{2},\ldots,a_{d}].$
DSPE既反映了样本间的鉴别关系，又能很好地排除伪类样本在重构人脸数据时带来的负面影响，经实验，在多个人脸库上效果超过PCA，LDA，NPE，LPP，SPP等算法。

核技巧
------
### 非线性分类问题

非线性分类问题是指通过利用非线性模型才能很好地进行分类的问题[@李航2012统计学习方法7.3]。如下图所示，左图为一个分类问题，图中“.”表示正实例点，“$\times$”表示负实例点，由图可知，这些实例无法通过一条直线进行分割，而可以通过一条椭圆曲线进行正确区分，像这种需要一个超曲面才能将正负实例区分开来的问题成为非线性可分问题。

\begin{figure}[H]
  \centering
  \subfloat{\includegraphics[scale=0.4]{pic/ellipse.png}}\qquad
  \subfloat{\includegraphics[scale=0.4]{pic/ellipse.png}}
  \caption{含两个子图形的图形}
\end{figure}
非线性问题求解困难，因此将非线性求解转化成线性求解是一种有效的方式，通过非线性变换将非线性分类问题转换成线性分类问题。

以以上分类问题为例，设原空间为 $\mathcal{X}\subset\mathbf{R}^{2}$，$x=(x^{(1)}$，$x^{(2)})^{T}\in\mathcal{X}$，新空间为$\mathcal{Z}\subset\mathbf{R}^{2}$，$z=(z^{(1)},z^{(2)})^{T}\in \mathcal{Z}$，定义从原空间到新空间的映射:
$$z=\phi (x) = ((x^{(1)})^{2},(x^{(2)})^{2})^{T}$$
经过变换$z=\phi (x)$，原空间 $\mathcal{X}\subset\mathbf{R}^{2}$变换为新空间$\mathcal{Z}\subset \mathbf{R}^2$，原空间中的点相应地变换为新空间中的点，原空间中的椭圆
$$w_{1}(z^{(1)})^{2}+w_{2}(z^{(2)})^{2}+b=0$$
变换成新空间中的直线
$$w_{1}z^{(1)}+w_{2}z^{(2)}+b=0$$
在变换后的新空间中，直线$w_{1}z^{(1)}+w_{2}z^{(2)}+b=0$可以将变换后的正负实例点正确分开。这样，原空间的非线性可分问题就变成了新空间的线性可分问题。因此用线性分类方法求解非线性分类问题分为两步：首先使用一个变换将原空间的数据映射到新空间，然后在新空间里用线性可分学习方法从训练数据中学习分类模型。核函数就是将原空间数据映射到新空间的一种方法。

### 核函数的定义
\newtheorem{kernel}{定义}
\begin{kernel}[核函数]
设$\mathcal{X}$是输入空间(欧式空间$\mathbf{R}^{n}$的子集或离散集合)，又设$\mathcal{H}$为特征空间（希尔伯特空间），如果存在一个从$\mathcal{X}$到$\mathcal{H}$的映射
$$\phi(x):\mathcal{X}\to\mathcal{H}$$
使得对所有$x,z\in\mathcal{X}$，函数$\mathbf{K}(x,z)$满足条件
$$\mathbf{K}(x,z) = \phi(x).\phi(z)$$
则称$\mathbf{K}(x,z)$为核函数，$\phi(x)$为映射函数，式中$\phi(x)\cdot\phi(z)$为$\phi(x)$$\phi(z)$的内积。
\end{kernel}

在学习和预测中，我们只需要定义核函数$\mathbf{K}(x,z)$[@李航2012统计学习方法7.3]，而不显式地定义映射函数$\phi$。通常，直接计算$\mathbf{K}(x,z)$比较容易，而通过$\phi(x)$和$\phi(z)$并不容易。对于给定的核$\mathbf{K}(x,z)$，特征空间$\mathcal{H}$和映射函数$\phi$的取法并不唯一，可以取不同的特征空间，即便是同一特征空间里也可以去不同的映射。

### 常用核函数

通常所说的核函数是正定核函数（positive definite kernel function）[@李航2012统计学习方法7.3]，由Mercer定理可以得到Mercer核（Mercer Kernel）[@邓2004支持向量机]，正定核比Mercer核更具一般性。在实际问题中往往使用已有的核函数，下面介绍一些常用的核函数：

1. 多项式核函数
   $$\mathbf{K}(x,z)=(x\cdot z+1)^{p}$$
2. 高斯核函数
   $$\mathbf{K}(x,z) = \exp{(-\frac{\Vert{x-z}\Vert^{2}}{2\sigma^{2}})}$$


核函数的思想可以和传统的方法相结合，形成多种不同的基于很函数技术的方法，可以为不同的应用选择不同的核函数和算法。目前基于核函数的传统方法改造已取得一些成果，核主成分分析(kernel PCA)[@scholkopf1998nonlinear]、核部分最小二乘法(kernel PLS)[@rosipal2002kernel]和核$Fisher$鉴别分析(Kernel Fisher Discriminator,KFD)[@roth1999nonlinear]是核函数的典型用例，在应用中都取得了不错的效果。


核鉴别稀疏保持嵌入
==================

核鉴别信息权重
--------------

将样本信息$X_{i}=[x_{i1},\ldots,x_{ik}]$映射到高维空间为$B_{i}=[\phi(x_{i1}),\ldots,\phi(x_{ik})]$，只要证明$\Vert x_{ij}-X_{i}t\Vert$越小，则$\Vert \phi(x_{ij})-X_{i}t\Vert$越小最小二乘问题变为
$$\min_{t}\Vert{\phi{(x_{ij})}-B_{i}t}\Vert_{2}$$ $$s.t.\quad It=1$$
因为$B$和$\phi{x}$是未知的，所以目标函数不能直接求解，所以将问题转化成以下约束：
$$\min_{t}\Vert{B^{T}\phi{(x_{ij})}-B^{T}B_{i}t}\Vert_{2}$$
$$s.t.\quad It=1$$ 求证
$\Vert x_{ij}-X_{i}t\Vert$，$$\min_{t}\Vert{\phi{(x_{ij})}-B_{i}t}\Vert_{2}$$，$$\min_{t}\Vert{B^{T}\phi{(x_{ij})}-B^{T}B_{i}t}\Vert_{2}$$同时达到最小值

$$B_{i}^{T}\phi{(x_{ij})}=
        \left[
            \begin{array}{ccc}
                \phi{(x_{i1})}\\
                \phi{(x_{i2})}\\
                \vdots\\
                \phi{(x_{ik})}
            \end{array}
        \right]\phi{(x_{ij})}=
        \left[
            \begin{array}{ccc}
                k(x_{i1},x_{ij})\\
                k(x_{i2},x_{ij})\\
                \vdots\\
                k(x_{ik},x_{ij})
            \end{array}
        \right]$$

$$B_{i}^{T}B_{i}=
    \left[
        \begin{array}{ccc}
            \phi{(x_{i1})}\\
            \phi{(x_{i2})}\\
            \vdots\\
            \phi{(x_{ik})}
        \end{array}
    \right]
    \left[
        \begin{array}{cccc}
            \phi{(x_{i1})} & \phi{(x_{i2})} & \ldots & \phi{(x_{ik})}
        \end{array}
    \right]=
    \left[
        \begin{array}{cccc}
            k(x_{i1},x_{i1}) & k(x_{i1},x_{i2}) & \ldots & k(x_{i1},x_{ij})\\
            k(x_{i2},x_{i1}) & k(x_{i2},x_{i2}) & \ldots & k(x_{i2},x_{ij})\\
            \vdots & \vdots & \ddots & \vdots \\
            k(x_{i3},x_{i1}) & k(x_{i3},x_{i2}) & \ldots & k(x_{i3},x_{ij})
        \end{array}
    \right]$$

目标函数转换成线性规划问题，可以求解。

核稀疏重构权重
--------------

令$C=[\phi{(X_{1})},\phi{(X_{2})},\ldots,\phi{(X_{c})}]$,令$C_{i}=[\phi{(X_{1})},\ldots,\phi{(X_{i-1})},\phi{(X_{i+1})},\ldots,\phi{(X_{c}})]$即从超完备字典中剔除样本$X_{i}$
令$er=\phi{(x_{ij})}-B_{i}\hat{t}=\phi{(x_{ij})}-C\hat{h}$,即样本由鉴别信息权重重构后得到的残差，对其对样本中的其他类进行重构：
$$\begin{array}{ll}
        \min_{s} & \Vert{s}\Vert_{1}\\
        \\
        s.t.  & \Vert{er-C_{i}s}\Vert < \epsilon \\
        & Is = 0
    \end{array}$$
同理由于函数$\phi$不能直接求解，将${er-C_{i}s}$左乘$C^{T}$转换成${C_{i}^{T}(er-C_{i}s)}$，即$C^{T}\phi{(x_{ij})}-C^{T}C\hat{h}-C^{T}C_{i}s$

求证 \[section\]

对任意$\epsilon\ge 0$都存在$\delta\ge 0$，只要$C^{T}\phi{(x_{ij})}-C^{T}C\hat{h}-C^{T}C_{i}s<\delta$
就有$er-C_{i}s < \epsilon$.

$$C^{T}\phi{(x_{ij})}=
        \left[
            \begin{array}{cccc}
                \phi{(x_{11})} & \phi{(x_{12})} & \ldots & \phi{(x_{1k})}\\
                \phi{(x_{21})} & \phi{(x_{22})} & \ldots & \phi{(x_{2k})}\\
                \vdots & \vdots & \ddots & \vdots\\
                \phi{(x_{c1})} & \phi{(x_{c2})} & \ldots & \phi{(x_{ck})}
            \end{array}
        \right]\phi{(x_{ij})}
        =\left[
            \begin{array}{cccc}
              k(x_{11},x_{ij}) & k(x_{12},x_{ij}) & \ldots &\k(x_{1k},x_{ij}) \\
              k(x_{21},x_{ij}) & k(x_{22},x_{ij}) & \ldots &\k(x_{2k},x_{ij}) \\
              \vdots & \vdots & \ddots & \vdots \\
              k(x_{c1},x_{ij}) & k(x_{c2},x_{ij}) & \ldots &\k(x_{ck},x_{ij}) 
            \end{array}
         \right]$$

$$C^{T}C=
        \left[
            \begin{array}{cccc}
                \sum_{j=1}^{k}k(x_{1j},x_{1j}) & \sum_{j=1}^{k}k(x_{1j},x_{2j}) & \ldots & \sum_{j=1}^{k}k(x_{1j},x_{cj}) \\
                \sum_{j=1}^{k}k(x_{2j},x_{1j}) & \sum_{j=1}^{k}k(x_{2j},x_{2j}) & \ldots & \sum_{j=1}^{k}k(x_{2j},x_{cj}) \\
                \vdots & \vdots & \vdots & \ddots \\
                \sum_{j=1}^{k}k(x_{cj},x_{1j}) & \sum_{j=1}^{k}k(x_{cj},x_{2j}) & \ldots & \sum_{j=1}^{k}k(x_{cj},x_{cj}) 
            \end{array}
        \right]$$

$$C^{T}C_{i}=
        \left[
            \begin{array}{cccccc}
                \sum_{j=1}^{k}k(x_{1j},x_{1j}) & \ldots & \sum_{j=1}^{k}k(x_{1j},x_{i-1,j}) & \sum_{j=1}^{k}k(x_{1j},x_{i+1,j}) &\ldots & \sum_{j=1}^{k}k(x_{1j},x_{cj}) \\
                \sum_{j=1}^{k}k(x_{2j},x_{1j}) & \ldots & \sum_{j=1}^{k}k(x_{2j},x_{i-1,j}) & \sum_{j=1}^{k}k(x_{1j},x_{i+1,j}) &\ldots & \sum_{j=1}^{k}k(x_{2j},x_{cj}) \\
                \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
                \sum_{j=1}^{k}k(x_{cj},x_{1j}) &\ldots & \sum_{j=1}^{k}k(x_{cj},x_{i-1,j}) & \sum_{j=1}^{k}k(x_{cj},x_{i+1,j}) &\ldots & \sum_{j=1}^{k}k(x_{cj},x_{cj}) 
            \end{array}
        \right]$$

将$\hat{s}$按照不同类别分块，得$\hat{s}=[\hat{s}_{1},\ldots,\hat{s}_{i-1},\hat{s}_{i+1},\ldots,\hat{s}_{c}]$,将$\hat{s}$稀疏到完备字典上得$\hat{l}=[\hat{s}_{1},\ldots,\hat{s}_{i-1},\vec{0},\hat{s}_{i+1},\ldots,\hat{s}_{c}]$
结合得到的鉴别权重$\hat{t}$,将系数串联得到$\hat{d}=\hat{l}+\hat{h}$,这样我们就得到了核鉴别稀疏权重。由于$I\hat{l}=I\hat{s}=0$，$I\hat{t}=1$，故$I\hat{d}=1$，能够保持旋转，平移，尺度不变的特性，证明过程见[@马小虎2014基于鉴别稀疏保持嵌入的人脸识别算法]

KDSPE的目标函数
---------------

与DSPE同理，KDSPE的目标函数数学模型如下：
$$\min_{W}\quad\sum_{i=1}^{n}\Vert{W^{T}x_{i}-W^{T}X\hat{d}_{i}}\Vert^{2}$$
$$s.t.\quad W^{T}XX^{T}W = 1$$ 与DSPE的推导相似，可得到
$$XMX^{T}W = \lambda XX^{T}W$$
选取得到的最大的$d$个特征值所对应的特征向量$\mathbf{a}_{i}$构成的特征子空间，即可得到KDPE的线性降维映射$W_{DSPE}=[a_{1},a_{2},\ldots,a_{d}]$.

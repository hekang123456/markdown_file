---
title: 贝叶斯公式+最大似然估计+最大后验概率公式
categories: ML
tags: [贝叶斯公式， Bayes’ Theorem, 最大似然估计， 最大后验概率估计， MLE, MAP, maximum likelihood estimation, maximum a posterior probability estimation]
date: 2018-11-26
---
> 来源： https://blog.csdn.net/u011508640/article/details/7281598
# 贝叶斯公式+最大似然估计(MLE)+最大后验概率公式(MAP)
## 1.贝叶斯公式
$$ P(A|B) = \frac{ P(B|A) \times P(A) }{ P(B|A) \times P(A) + P(B|\sim A) \times P(\sim A) } $$
- 作用： 
  - 你有多大把握相信一件证据。 给定 $B$ 的时候，你有多大的可能性会去相信 $A$ 能够成立。
  - 在做判断的时候需要考虑所有的因素。
    - 一件很难发生的事情 $P(A)$ 即使出现某个证据 $B$ 和它强相关 $P(B|A)$ 也要谨慎，因为证据可能来自其他虽然不是强相关但发生概率较高的事情 因为 $P(B|\sim A) \times P(\sim A)$ 可能会比较大从而导致$P(B|A)$ 比较小。 
  - 根据已知的或者主观容易断定的条件概率事件，计算出未知的或者较难评估的条件概率事件
  
## 2. 似然函数
对于函数 $P(x| \theta)$:
- 当 $\theta$ 是已知的情况下， $x$ 是变量， 这个函数叫做概率函数（probability function）, 用来描述对于不同的样本点 $x$ , 其出现的概率是多少。
- 当 $x$ 是已知的情况下， $\theta$ 是变量， 这个函数叫做似然函数（likelihood function）, **用来描述对于不同的模型参数**， 这个样本点出现的概率是多少。 

## 3. 最大似然估计（maximum likelihood estimation : MLE）
- 构造一个关于参数 $\theta$ 的函数， 这个函数用来表示在已知的一组实验中产生了一组实验数据 $x$ 的可能性。
  - 在抛硬币实验中，每次抛硬币出现正反的概率满足二项分布。
  - 比如抛了10次，出现的一组实验数据 $x=[0111101110]$。 
  - 似然函数为： $f(\theta) = ((1−\theta) × \theta × \theta × \theta × \theta × (1 − \theta)× \theta × \theta × \theta ×(1−\theta))=\theta^7 \times (1 - \theta)^3$
- 计算使似然函数最大的参数值， 一般先取对数然后计算。
  - $\log f(\theta) = 7\log \theta + 3\log (1-\theta) $ 
  - 求导可以得到

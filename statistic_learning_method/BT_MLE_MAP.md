---
title: 贝叶斯公式+最大似然估计+最大后验概率公式
categories: ML
tags: [贝叶斯公式， Bayes’ Theorem, 最大似然估计， 最大后验概率估计， MLE, MAP, maximum likelihood estimation, maximum a posterior probability estimation]
date: 2018-11-26
---
> 来源： https://blog.csdn.net/u011508640/article/details/7281598
# 贝叶斯公式+最大似然估计(MLE)+最大后验概率公式(MAP)
## 1.贝叶斯公式
$$ P(A|B) = \frac{ P(B|A) \times P(A) }{ P(B|A) \times P(A) + P(B|\~A) \times P(\~A) } $$
- 作用： 
  - 你有多大把握相信一件证据。 给定 $B$ 的时候，你有多大的可能性会去相信 $A$ 能够成立。
  - 在做判断的时候需要考虑所有的因素。
    - 一件很难发生的事情 $P(A)$ 即使出现某个证据 $B$ 和它强相关 $P(B|A)$ 也要谨慎，因为证据可能来自其他虽然不是强相关但发生概率较高的事情 因为 $P(B|\~A) \times P(\~A)$ 可能会比较大从而导致$P(B|A)$ 比较小。 
    

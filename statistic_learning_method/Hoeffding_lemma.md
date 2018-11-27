---
title: Hoeffding's lemma (引理) 的证明
categories: 数学证明
tags: [Hoeffding's lemma, 霍夫丁引理]
---
source: https://en.wikipedia.org/wiki/Hoeffding%27s_lemma
# Statement of the Lemma
Let $X$ be any real-valued random variable with expected value $E(x)=0$, and such that $a \leq X \leq b$ almost truely,  Then, for all $\lambda \in R $,
$$ E[e^{\lambda X}] \leq \begin{cases}  \frac{\lambda^2 (b-a)^2}{8} \end{cases}$$
Note that because of the assumption that the random variable $X$ has zero expectation,  the $a$ and $b$ in the lemma must satisfy $a \leq 0 \leq b$.

# A Brief Proof of the Lemma 

$$ f(x)=\left\{
\begin{aligned}
x & = & \cos(t) \\
y & = & \sin(t) \\
z & = & \frac xy
\end{aligned}
\right.
$$


$$ F^{HLLC}=\left\{
\begin{array}{rcl}
F_L       &      & {0      <      S_L}\\
F^*_L     &      & {S_L \leq 0 < S_M}\\
F^*_R     &      & {S_M \leq 0 < S_R}\\
F_R       &      & {S_R \leq 0}
\end{array} \right. $$

$$f(x)=
\begin{cases}
0& \text{x=0}\\
1& \text{x!=0}
\end{cases}$$


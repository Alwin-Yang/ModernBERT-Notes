# RLCD

This chapter explains the public Laya interpretation of Reinforcement Learning
for Calibrated Decisions. It must not be treated as a disclosure of Jev's
private training algorithm.

There is also a 2023 method called Reinforcement Learning from Contrastive
Distillation with the same acronym. That alignment method is unrelated to the
decision-calibration recipe here.

## Calibration as a training target

Accuracy asks whether the top label is correct. Calibration asks whether
reported probabilities match long-run frequencies. Among predictions made at
80% confidence, roughly 80% should be correct.

A model can be calibrated but uninformative, such as always reporting 50/50 on
a balanced binary task. A model can also be accurate but dangerously
overconfident. Decision systems need both discrimination and calibration.

## Strictly proper scoring rules

A scoring rule compares a reported distribution $q$ with an outcome sampled
from the true distribution $p$. It is strictly proper when expected reward is
uniquely maximized by reporting $q=p$.

For the log score and observed class $y$:

$$
S_{log}(q,y)=\log q_y.
$$

Its expected value is:

$$
\mathbb{E}_{y\sim p}[S_{log}(q,y)]
=\sum_y p_y\log q_y
=-H(p)-D_{KL}(p\Vert q).
$$

$H(p)$ is fixed and KL divergence is zero only when $q=p$. Honest reporting is
therefore the unique optimum.

Laya combines log score with spherical score:

$$
S_{sph}(q,t)=\frac{t^\top q}{\lVert q\rVert_2},
$$

where $t$ may be a one-hot label or a soft teacher distribution.

## Ordinal reward with RPS

Score levels are ordered. Predicting level 1 when the answer is level 2 should
be less wrong than predicting level 0. Cross entropy alone does not know this
distance. Ranked Probability Score compares cumulative distributions:

$$
\operatorname{RPS}(q,t)=\frac{1}{K-1}
\sum_{k=0}^{K-1}
\left(\sum_{i=0}^{k}q_i-\sum_{i=0}^{k}t_i\right)^2.
$$

Laya's public reward is:

$$
R(q,t)=S_{log}+w_{sph}S_{sph}
-\mathbb{1}[\text{score}]w_{rps}\operatorname{RPS}.
$$

RPS belongs only on ordered score questions. Applying it to unordered choice
labels would invent a meaningless distance based on label order.

## Noisy-logit policy

Let the model logits $\mu_\theta(x)$ be the mean of a Gaussian policy. Sample
$G$ perturbed reports:

$$
\epsilon_g\sim\mathcal{N}(0,\sigma^2I), \qquad
z_g=\operatorname{stopgrad}(\mu_\theta)+\epsilon_g.
$$

The implementation subtracts each noise vector's option-wise mean. Softmax is
unchanged when the same constant is added to every logit, so exploring that
direction would be wasted variance.

Each action becomes a distribution $q_g=\operatorname{softmax}(z_g)$ and
receives reward $R_g$. A group-relative baseline gives:

$$
A_g=R_g-\frac{1}{G}\sum_{j=1}^{G}R_j.
$$

Subtracting a baseline does not change the expected policy gradient. It reduces
variance. Laya also normalizes advantages by their standard deviation.

Ignoring constants, the Gaussian policy log-probability is:

$$
\log\pi_\theta(z_g|x)
=-\frac{\lVert z_g-\mu_\theta(x)\rVert^2}{2\sigma^2}+C.
$$

The REINFORCE loss is:

$$
L_{RL}=-\frac{1}{G}\sum_g A_g\log\pi_\theta(z_g|x).
$$

The sampled logits are detached, but the policy mean inside $\log\pi$ is not.
The reward itself does not need a gradient.

## RL plus supervised guidance

The public Laya notebook does not use pure reinforcement learning. It adds soft
cross entropy:

$$
L=L_{RL}+\lambda L_{CE}, \qquad
L_{CE}=-\sum_i t_i\log\operatorname{softmax}(\mu)_i.
$$

The published notebook uses $\lambda=1$. Cross entropy supplies a direct,
low-variance signal. The policy term allows the same training interface to
compare reports using a richer or even non-differentiable reward.

When the reward is differentiable and complete targets are available, whether
RL improves on direct proper-loss training is an empirical question. A useful
experiment compares CE-only, proper-loss-only, RL-only, and RL+CE across several
random seeds.

## Tensor shapes for one update

For batch size $B=8$, maximum options $K=5$, and group size $G=4$:

```text
logits:           (B, K)       = (8, 5)
target:           (B, K)       = (8, 5)
option_mask:      (B, K)       = (8, 5)
noise:            (G, B, K)    = (4, 8, 5)
sampled_logits:   (G, B, K)
probabilities:    (G, B, K)
reward:           (G, B)
advantage:        (G, B)
gaussian_logp:    (G, B)
loss:             scalar
```

## Hands-on

Run:

```bash
python -m examples.train_rlcd_toy
python -m pytest tests/test_rlcd.py -q
```

The toy policy has only two trainable logits and a soft target of
`[0.25, 0.75]`. An honest solution approaches $P(true)=0.75$ instead of forcing
the probability to one.

---

# 中文版本

本章解释 Laya 公开实现中的 Reinforcement Learning for Calibrated Decisions，不能把
它当作 Jev 私有训练算法的披露。

2023 年还有一个同样缩写为 RLCD 的 Reinforcement Learning from Contrastive
Distillation。它是另一种 alignment method，与本章的 decision calibration recipe
无关。

## 把 calibration 当作训练目标

accuracy 问 top label 是否正确；calibration 问报告的概率是否与长期频率一致。模型在
一批样本上报告 80% confidence 时，理想情况下约 80% 应该正确。

模型可以校准良好但没有信息量，例如在平衡二分类上总是报告 50/50；也可以准确率较高却
危险地过度自信。decision system 同时需要 discrimination 与 calibration。

## Strictly proper scoring rule

scoring rule 比较报告分布 $q$ 与从真实分布 $p$ 采样的结果。若期望 reward 只在
$q=p$ 时唯一最大，它就是 strictly proper。

对 observed class $y$，log score 为：

$$
S_{log}(q,y)=\log q_y.
$$

期望值为：

$$
\mathbb{E}_{y\sim p}[S_{log}(q,y)]
=\sum_y p_y\log q_y
=-H(p)-D_{KL}(p\Vert q).
$$

$H(p)$ 固定，而 KL divergence 只在 $q=p$ 时为零，所以诚实报告是唯一最优解。

Laya 还组合 spherical score：

$$
S_{sph}(q,t)=\frac{t^\top q}{\lVert q\rVert_2},
$$

其中 $t$ 可以是 one-hot label，也可以是 soft teacher distribution。

## 用 RPS 表示 ordinal distance

score level 有顺序。真实答案为 level 2 时，预测 level 1 应比预测 level 0 错得更轻。
cross entropy 不知道这种距离，Ranked Probability Score 比较累积分布：

$$
\operatorname{RPS}(q,t)=\frac{1}{K-1}
\sum_{k=0}^{K-1}
\left(\sum_{i=0}^{k}q_i-\sum_{i=0}^{k}t_i\right)^2.
$$

Laya 的公开 reward 是：

$$
R(q,t)=S_{log}+w_{sph}S_{sph}
-\mathbb{1}[\text{score}]w_{rps}\operatorname{RPS}.
$$

RPS 只适合有序 score question。对无序 choice label 使用它，会因为排列顺序凭空制造
没有意义的距离。

## Noisy-logit policy

把模型 logits $\mu_\theta(x)$ 当作 Gaussian policy 的均值，采样 $G$ 组扰动：

$$
\epsilon_g\sim\mathcal{N}(0,\sigma^2I), \qquad
z_g=\operatorname{stopgrad}(\mu_\theta)+\epsilon_g.
$$

实现会减去每个 noise vector 在 option 轴上的均值。softmax 对所有 logit 同加一个常数
不敏感，因此探索这个方向只会增加 variance。

每个 action 变成 $q_g=\operatorname{softmax}(z_g)$ 并获得 reward $R_g$。group-relative
baseline 为：

$$
A_g=R_g-\frac{1}{G}\sum_{j=1}^{G}R_j.
$$

减去 baseline 不改变期望 policy gradient，只降低 variance。Laya 还用标准差归一化
advantage。

忽略常数项，Gaussian policy log-probability 为：

$$
\log\pi_\theta(z_g|x)
=-\frac{\lVert z_g-\mu_\theta(x)\rVert^2}{2\sigma^2}+C.
$$

REINFORCE loss 为：

$$
L_{RL}=-\frac{1}{G}\sum_g A_g\log\pi_\theta(z_g|x).
$$

sampled logits 要 detach，但 $\log\pi$ 内的 policy mean 不能 detach；reward 本身不需要
gradient。

## RL 加 supervised guidance

Laya 的公开 notebook 不是纯 RL，还加入 soft cross entropy：

$$
L=L_{RL}+\lambda L_{CE}, \qquad
L_{CE}=-\sum_i t_i\log\operatorname{softmax}(\mu)_i.
$$

公开 notebook 使用 $\lambda=1$。cross entropy 提供直接、低 variance 的监督信号；
policy term 则让同一训练接口可以使用更复杂甚至不可微的 reward。

当 reward 可微且有完整 target 时，RL 是否优于直接 proper-loss training，需要实验回答。
值得比较 CE-only、proper-loss-only、RL-only 与 RL+CE，并使用多个 random seed。

## 一次 update 的 tensor shape

完整 shape 见英文部分。关键是 group 维放在 batch 之前：probability、reward 与 advantage
分别为 `(G,B,K)`、`(G,B)`、`(G,B)`，最终 loss 为 scalar。

## 动手实现

运行：

```bash
python -m examples.train_rlcd_toy
python -m pytest tests/test_rlcd.py -q
```

toy policy 只有两个 trainable logits，soft target 是 `[0.25, 0.75]`。诚实解应接近
$P(true)=0.75$，而不是把概率强行推到 1。

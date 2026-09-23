# Decision Models Q&A

<div class="qa-list" markdown>

## Why is schema-valid output not the same as a correct decision?

A type constraint prevents malformed output, but the selected option and its
probability can still be wrong.

## Why use an option marker instead of only the CLS token?

Each marker produces a contextual representation tied to one candidate, so a
shared scorer can compare variable option sets directly.

## Why is a proper scoring rule important?

Its expected reward is maximized by reporting the true conditional
distribution, so systematic overconfidence or underconfidence cannot improve
expected score.

## Why center Gaussian exploration noise across options?

Softmax ignores a common shift of all logits, so centering removes exploration
in a direction that cannot change the reported distribution.

## Does the group-mean baseline bias REINFORCE?

No. A baseline independent of the sampled action's score function preserves the
expected gradient while reducing variance.

## Why is Laya's public RLCD recipe not pure reinforcement learning?

Its published loss adds full soft cross entropy to the policy-gradient term, so
supervised target guidance remains a central part of every update.

## Can temperature scaling improve accuracy?

Not by itself. A positive temperature preserves logit ordering; it can improve
probability quality without changing top-1 predictions.

## Does good calibration imply a useful classifier?

No. A balanced binary classifier that always reports 50/50 can be calibrated
while providing no discrimination.

</div>

---

# 中文版本

<div class="qa-list" markdown>

## 为什么 schema-valid output 不等于正确决策？

type constraint 可以阻止格式错误，但选中的 option 与报告的概率仍然可能错误。

## 为什么使用 option marker，而不只使用 CLS token？

每个 marker 都产生与一个 candidate 对应的 contextual representation，共享 scorer 因此
可以直接比较可变的 option set。

## proper scoring rule 为什么重要？

它的期望 reward 只在报告真实条件分布时最大，因此系统性的过度自信或不足自信无法提高
期望得分。

## 为什么 Gaussian exploration noise 要在 option 轴居中？

softmax 忽略所有 logit 的共同平移，居中可以移除无法改变报告分布的无效探索方向。

## group-mean baseline 会让 REINFORCE 产生 bias 吗？

不会。与 sampled action 的 score function 独立的 baseline 保持期望 gradient，同时
降低 variance。

## 为什么 Laya 的公开 RLCD recipe 不是纯 reinforcement learning？

公开 loss 在 policy-gradient term 之外加入完整 soft cross entropy，所以 supervised
target guidance 仍是每次 update 的核心部分。

## temperature scaling 能提高 accuracy 吗？

它本身不能。正 temperature 保留 logit 排序，可以改善 probability quality，但不会
改变 top-1 prediction。

## calibration 良好是否代表 classifier 有用？

不代表。平衡二分类器总是报告 50/50 可以 calibration 良好，却没有 discrimination。

</div>

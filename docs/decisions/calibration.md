# Calibration

Proper scoring rules encourage honest probabilities on the training
distribution. They do not guarantee calibration after domain shift, model
selection, class imbalance changes, or fine-tuning. Calibration must be
measured on held-out data from the intended deployment domain.

## Temperature scaling

Temperature scaling learns one positive scalar $T$ while keeping the model
weights fixed:

$$
p_T(y=i|x)=\operatorname{softmax}(z(x)/T)_i.
$$

- $T>1$ flattens the distribution and reduces confidence.
- $T<1$ sharpens the distribution and increases confidence.
- positive scaling preserves the logit ordering, so top-1 accuracy is unchanged.

Fit $T$ by minimizing negative log-likelihood on a calibration split that never
entered training. Fitting on the training set usually produces a misleadingly
sharp value because the model has already adapted to those examples.

## Expected Calibration Error

Expected Calibration Error groups predictions by confidence. For confidence
bins $B_b$:

$$
\operatorname{ECE}=\sum_b\frac{|B_b|}{N}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.
$$

ECE is intuitive but depends on bin boundaries and usually examines only the
top label. It should not be the only metric.

## A useful evaluation set

Report several complementary views:

- accuracy or macro-F1 for discrimination;
- negative log-likelihood and Brier score for probability quality;
- ECE plus a reliability diagram for calibration;
- coverage and error at each confidence threshold;
- results split by domain, question type, and option count.

For an act-or-escalate system, the most operational curve is selective risk.
As the confidence threshold rises, coverage falls. The accepted subset should
become more accurate. Choose a threshold from a held-out set based on the cost
of a wrong automatic action versus the cost of escalation.

## Common mistakes

1. **Calibrating on training data.** This measures memorization, not deployment
   reliability.
2. **Calling entropy confidence a probability of correctness.** Low entropy only
   says the distribution is concentrated.
3. **Reusing a threshold after changing candidates.** Shortlisting and option
   count change the conditional distribution.
4. **Reporting only aggregate ECE.** One option-count or language group can be
   badly miscalibrated while the average looks acceptable.
5. **Assuming calibration transfers.** A temperature fitted on support tickets
   need not work for security alerts.

## Hands-on

`fit_temperature` in
[`modernbert_notes/rlcd.py`](https://github.com/Alwin-Yang/ModernBERT-Notes/blob/main/modernbert_notes/rlcd.py)
optimizes held-out NLL. The test creates a classifier that reports 98% confidence
but is correct only 70% of the time, then verifies that $T>1$ lowers held-out
NLL.

---

# 中文版本

proper scoring rule 鼓励模型在训练分布上诚实报告概率，但不能保证 domain shift、model
selection、class imbalance 变化或 fine-tuning 之后仍然 calibrated。必须在目标部署
domain 的 held-out data 上测量 calibration。

## Temperature scaling

temperature scaling 固定模型权重，只学习一个正数 $T$：

$$
p_T(y=i|x)=\operatorname{softmax}(z(x)/T)_i.
$$

- $T>1$ 让分布变平，降低 confidence。
- $T<1$ 让分布变尖，提高 confidence。
- 正数缩放不改变 logit 排序，因此 top-1 accuracy 不变。

在从未参与训练的 calibration split 上最小化 negative log-likelihood 来拟合 $T$。
若在 training set 上拟合，模型已经适应这些样本，通常会得到误导性的过尖温度。

## Expected Calibration Error

Expected Calibration Error 按 confidence 分桶。对 confidence bin $B_b$：

$$
\operatorname{ECE}=\sum_b\frac{|B_b|}{N}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.
$$

ECE 直观，但依赖 bin boundary，并且通常只看 top label，不能作为唯一指标。

## 有用的 evaluation set

应该同时报告：

- accuracy 或 macro-F1，衡量 discrimination；
- negative log-likelihood 与 Brier score，衡量 probability quality；
- ECE 与 reliability diagram，衡量 calibration；
- 每个 confidence threshold 下的 coverage 与 error；
- 按 domain、question type 与 option count 分组的结果。

对 act-or-escalate system，最实用的是 selective risk curve。threshold 提高时 coverage
下降，被接受的子集应更加准确。阈值要根据错误自动执行与 escalation 的成本，在 held-out
set 上选择。

## 常见错误

1. **在 training data 上校准。** 这测到的是记忆，不是部署可靠性。
2. **把 entropy confidence 当作正确概率。** 低 entropy 只表示分布集中。
3. **候选改变后继续使用旧 threshold。** shortlist 与 option count 会改变条件分布。
4. **只报告 aggregate ECE。** 某种 option count 或语言可能很差，却被平均值掩盖。
5. **假设 calibration 会迁移。** support ticket 上拟合的 temperature 未必适合 security
   alert。

## 动手实现

[`modernbert_notes/rlcd.py`](https://github.com/Alwin-Yang/ModernBERT-Notes/blob/main/modernbert_notes/rlcd.py)
中的 `fit_temperature` 最小化 held-out NLL。测试构造一个报告 98% confidence、实际只
正确 70% 的 classifier，并验证 $T>1$ 可以降低 held-out NLL。

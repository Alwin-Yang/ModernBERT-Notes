# Practice

The repository includes a small PyTorch package, two runnable examples, and a
test suite. They require no model download.

## Quick start

```bash
python -m examples.inspect_modernbert
python -m examples.train_rlcd_toy
python -m pytest -q
```

The first example prints ModernBERT tensor shapes and verifies the alternating
attention pattern. The second trains two logits toward an honest 75% binary
probability using the Laya-style RLCD loss.

## Test map

`tests/test_model.py` checks:

- global layers occur at indices 0, 3, 6, ...;
- RoPE preserves vector norms;
- local attention cannot see outside its window;
- global attention can see distant real tokens;
- padded hidden states remain zero;
- tied MLM weights receive gradients;
- invalid option markers receive zero probability.

`tests/test_rlcd.py` checks:

- the honest probability report receives the highest proper reward;
- RPS prefers an ordinal error that is closer to the target;
- the policy-gradient loss backpropagates finite gradients;
- held-out temperature fitting lowers NLL for an overconfident classifier.

## Suggested experiments

1. Sweep `sigma` and `group_size`; record convergence variance across seeds.
2. Compare CE-only, RL-only, and RL+CE on the same synthetic task.
3. Create a five-level ordinal task and ablate RPS.
4. Introduce a train/calibration/test split and plot a reliability diagram.
5. Replace synthetic token IDs with a tokenizer only after the mechanics are
   understood.

---

# 中文版本

仓库包含一个小型 PyTorch package、两个可运行 example 和一套测试，不需要下载模型。

## 快速开始

```bash
python -m examples.inspect_modernbert
python -m examples.train_rlcd_toy
python -m pytest -q
```

第一个 example 打印 ModernBERT tensor shape，并验证交替 attention pattern；第二个用
Laya-style RLCD loss 训练两个 logits，使二元概率诚实地接近 75%。

## Test map

`tests/test_model.py` 检查：

- global layer 出现在 0、3、6 等位置；
- RoPE 保持向量 norm；
- local attention 看不到 window 外的位置；
- global attention 可以看到远处真实 token；
- padding hidden state 保持为零；
- 共享 MLM weight 能收到 gradient；
- 无效 option marker 的概率为零。

`tests/test_rlcd.py` 检查：

- 诚实 probability report 获得最高 proper reward；
- RPS 更偏好距离 target 较近的 ordinal error；
- policy-gradient loss 能反向传播有限 gradient；
- held-out temperature fitting 能降低过度自信 classifier 的 NLL。

## 建议实验

1. sweep `sigma` 与 `group_size`，记录多个 seed 的收敛 variance。
2. 在同一个 synthetic task 上比较 CE-only、RL-only 与 RL+CE。
3. 构造五级 ordinal task，并 ablate RPS。
4. 引入 train/calibration/test split，绘制 reliability diagram。
5. 只有在理解 mechanics 之后，再把 synthetic token ID 换成真实 tokenizer。

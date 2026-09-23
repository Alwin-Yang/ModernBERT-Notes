# ModernBERT from Scratch

The implementation in `modernbert_notes/model.py` is intentionally small. It
keeps model mathematics explicit and omits production-only kernels.

## Configuration invariants

`ModernBERTConfig` validates three constraints before constructing a model:

- `hidden_size` must be divisible by `num_attention_heads`;
- $d_k$ must be even because RoPE rotates feature pairs;
- the local window must be a positive even number.

It also defines `is_global_layer(layer_idx)`, which returns true for layers
0, 3, 6, and so on.

## Reading the implementation

Follow the code in this order:

1. `apply_rotary_embeddings` builds frequencies and rotates adjacent pairs.
2. `make_local_attention_mask` constructs bidirectional window visibility.
3. `MultiHeadSelfAttention` performs fused QKV projection and masking.
4. `GeGLU` applies the gated feed-forward transformation.
5. `ModernBERTLayer` adds the two Pre-LN residual branches.
6. `ModernBERT` adds embeddings, the layer stack, and final normalization.
7. `ModernBERTForMaskedLM` ties the output classifier to token embeddings.

## Masked language modeling

For hidden states $H_{B\times T\times d_{model}}$ and embedding matrix
$E_{V\times d_{model}}$, the tied decoder computes:

$$
Z_{B\times T\times V}=HE^\top+b.
$$

Only selected masked positions contribute to cross entropy. Other labels use
the ignore index `-100`. The model predicts specified hidden tokens in parallel;
it does not run an autoregressive generation loop.

## What the teaching code leaves out

The implementation does not claim checkpoint compatibility with Hugging Face.
It omits variable-length unpadding, FlashAttention, fused kernels, truncated
normal initialization, compile paths, gradient checkpointing, and sparse MLM
prediction. These are important for production efficiency but obscure the
first pass through the architecture.

## Verify the implementation

From the repository root:

```bash
python -m examples.inspect_modernbert
python -m pytest tests/test_model.py -q
```

The tests check RoPE norm preservation, local visibility, global visibility,
padding removal, tied MLM gradients, and option-marker masking.

## Exercises

1. Set `local_attention=4` and print layer 1's attention matrix.
2. Replace all layers with global attention and time lengths 128, 256, and 512.
3. Remove RoPE. Explain why shapes still pass while position understanding does
   not.
4. Replace GeGLU with a standard GELU MLP and compare parameter counts.
5. Train the tiny MLM on a repeated synthetic sequence and verify that loss
   decreases.

---

# 中文版本

`modernbert_notes/model.py` 的实现刻意保持小而清晰。它显式展示模型数学，并省略
只与生产 kernel 有关的复杂度。

## Config 不变量

`ModernBERTConfig` 在构造模型前检查三个约束：

- `hidden_size` 必须能被 `num_attention_heads` 整除；
- $d_k$ 必须为偶数，因为 RoPE 要旋转成对特征；
- local window 必须是正偶数。

它还提供 `is_global_layer(layer_idx)`，第 0、3、6 层返回 true。

## 阅读实现的顺序

建议按下面顺序阅读：

1. `apply_rotary_embeddings` 构造频率并旋转相邻特征对。
2. `make_local_attention_mask` 构造双向 window visibility。
3. `MultiHeadSelfAttention` 完成融合 QKV projection 与 mask。
4. `GeGLU` 完成 gated feed-forward transformation。
5. `ModernBERTLayer` 加上两条 Pre-LN residual branch。
6. `ModernBERT` 加上 embedding、layer stack 与 final normalization。
7. `ModernBERTForMaskedLM` 让输出 classifier 与 token embedding 共享权重。

## Masked language modeling

对 hidden states $H_{B\times T\times d_{model}}$ 和 embedding matrix
$E_{V\times d_{model}}$，共享权重的 decoder 计算：

$$
Z_{B\times T\times V}=HE^\top+b.
$$

只有选中的 masked position 参与 cross entropy，其余 label 使用 ignore index
`-100`。模型并行预测指定的隐藏 token，不运行 autoregressive generation loop。

## 教学代码省略了什么

该实现不追求与 Hugging Face checkpoint 兼容。它省略 variable-length unpadding、
FlashAttention、fused kernel、truncated normal initialization、compile path、
gradient checkpointing 和 sparse MLM prediction。这些对生产效率很重要，但会干扰第一遍
理解架构。

## 验证实现

在仓库根目录运行：

```bash
python -m examples.inspect_modernbert
python -m pytest tests/test_model.py -q
```

测试覆盖 RoPE 长度保持、local/global visibility、padding 移除、共享 MLM weight 的
gradient，以及 option-marker mask。

## 练习

1. 设置 `local_attention=4`，打印第 1 层 attention matrix。
2. 把所有层改为 global attention，测量长度 128、256、512 的耗时。
3. 删除 RoPE，解释为什么 shape 仍然正确，但模型失去位置理解。
4. 把 GeGLU 换成标准 GELU MLP，并比较参数量。
5. 用重复的 synthetic sequence 训练 tiny MLM，确认 loss 下降。

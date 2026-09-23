# ModernBERT Q&A

<div class="qa-list" markdown>

## Why is ModernBERT bidirectional rather than causal?

Its encoder is trained to build a representation from both left and right
context, which suits classification and retrieval but prevents next-token
generation without a separate decoder.

## Why does RoPE require an even head dimension?

RoPE treats every two features as a 2D plane and rotates that pair, so an odd
feature would have no rotation partner.

## What breaks if local attention is implemented only as a dense mask?

The outputs remain correct, but memory and compute stay quadratic because the
full score matrix was already created.

## Why is every third layer global?

Local layers make long sequences affordable, while periodic global layers let
information cross the whole document instead of remaining trapped in a window.

## Why does layer 0 skip its attention LayerNorm?

Embeddings are already normalized immediately before layer 0, so another
LayerNorm there would repeat the same operation.

## Does unpadding change model semantics?

No. It removes storage and computation for padding while preserving sequence
boundaries and the attention result for every real token.

</div>

---

# 中文版本

<div class="qa-list" markdown>

## 为什么 ModernBERT 使用双向 attention，而不是 causal attention？

encoder 要同时利用左右上下文构造表示，这适合分类与检索；没有单独 decoder 时，它不能
直接做 next-token generation。

## 为什么 RoPE 要求 head dimension 为偶数？

RoPE 把每两个特征看作一个二维平面并旋转这一对；奇数维会留下一个无法配对的特征。

## 如果 local attention 只通过 dense mask 实现，会损失什么？

输出仍然正确，但完整 score matrix 已经生成，所以显存与计算仍然是二次复杂度。

## 为什么每三层设置一个 global layer？

local layer 让长序列变得可负担，周期性的 global layer 则让信息可以跨越整个文档，而
不是被困在局部 window 中。

## 为什么第 0 层跳过 attention LayerNorm？

embedding 在进入第 0 层前已经完成归一化，再做一次 LayerNorm 会重复相同操作。

## unpadding 会改变模型语义吗？

不会。它只移除 padding 的存储和计算，同时保留序列边界及每个真实 token 的 attention
结果。

</div>

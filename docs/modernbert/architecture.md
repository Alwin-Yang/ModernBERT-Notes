# ModernBERT Architecture

This page derives the main changes that turn BERT into ModernBERT. Let batch
size be $B$, sequence length be $T$, hidden width be $d_{model}$, the number of
attention heads be $H$, and per-head width be $d_k=d_{model}/H$.

## Fused QKV projection

After token embedding and LayerNorm, the hidden states have shape
$X_{B \times T \times d_{model}}$. One bias-free linear layer produces all
queries, keys, and values:

$$
[Q,K,V] = XW_{QKV}, \qquad
W_{QKV\,d_{model}\times3d_{model}}.
$$

The implementation reshapes the result as follows:

```text
(B, T, d_model)
  -> Linear(d_model, 3 * d_model)
(B, T, 3 * d_model)
  -> view(B, T, 3, H, d_k)
  -> unbind and transpose
Q, K, V: (B, H, T, d_k)
```

Attention is still scaled dot-product attention:

$$
A = \operatorname{softmax}\left(
  \frac{QK^\top}{\sqrt{d_k}} + M
\right), \qquad O = AV.
$$

Unlike a decoder, the encoder does not use a causal mask. Every real token may
use context on both sides. The mask $M$ only removes padding and, in local
layers, positions outside the sliding window.

## Rotary positional embeddings

Original BERT adds a learned absolute position vector to each token embedding.
ModernBERT rotates pairs of query and key features instead. For position $m$:

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(m\omega_i)&-\sin(m\omega_i)\\
\sin(m\omega_i)& \cos(m\omega_i)
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix},
\qquad \omega_i=\theta^{-2i/d_k}.
$$

This rotation preserves the norm of each vector. The inner product between a
query at position $m$ and key at position $n$ depends naturally on their
relative displacement $m-n$.

ModernBERT uses two RoPE bases:

- global layers use $\theta=160{,}000$ for long-range positions;
- local layers use $\theta=10{,}000$ because they only see nearby tokens.

RoPE is applied to $Q$ and $K$, where position affects similarity. It is not
applied to $V$, which carries the content aggregated after similarity is known.

## Alternating attention

Full attention has $T^2$ query-key relationships. That becomes expensive at
the native context length of 8,192 tokens. ModernBERT uses a repeating pattern:

```text
layer:  0       1      2      3       4      5      6
type: global  local  local  global  local  local  global
```

A local layer allows a token at position $i$ to attend only when
$|i-j|\leq w/2$:

$$
M_{ij}=\begin{cases}
0,&|i-j|\leq w/2\\
-\infty,&\text{otherwise}.
\end{cases}
$$

The official `local_attention=128` corresponds to a half-window of 64 on each
side. With a true sliding-window kernel, local cost changes from $O(T^2)$ to
$O(Tw)$. Periodic global layers carry information across the full document.

!!! warning "Masking is not the optimization"

    Building a full $T\times T$ score matrix and then masking it gives the
    correct result but keeps the quadratic work. The speedup requires a kernel
    that never computes the masked entries.

## Pre-LayerNorm residual blocks

Post-LayerNorm, common in early BERT implementations, normalizes after adding
the residual. ModernBERT uses Pre-LayerNorm:

$$
x' = x + \operatorname{Attention}(\operatorname{LN}(x)),
$$

$$
x'' = x' + \operatorname{MLP}(\operatorname{LN}(x')).
$$

The residual path stays close to an identity map, which improves gradient flow
through deep networks. Embeddings receive a LayerNorm first, so layer 0 skips
the otherwise repeated attention normalization. A final LayerNorm closes the
stack.

## GeGLU feed-forward layer

A standard BERT feed-forward layer applies one GELU path. GeGLU creates a value
and a learned gate:

$$
[a,g]=xW_{in}, \qquad
\operatorname{GeGLU}(x)=W_{out}(\operatorname{GELU}(a)\odot g).
$$

The shapes are:

```text
x:                 (B, T, d_model)
W_in:              d_model -> 2 * d_ff
a, gate:           each (B, T, d_ff)
GELU(a) * gate:    (B, T, d_ff)
W_out:             d_ff -> d_model
output:            (B, T, d_model)
```

The gate can suppress or pass individual intermediate features for each token.
ModernBERT-base uses $d_{model}=768$ and $d_{ff}=1152$.

## Unpadding and FlashAttention

Padding makes all sequences in a batch the same length, but padded tokens have
no semantic value. ModernBERT's optimized path removes them before embedding,
packs real tokens into one buffer, and supplies cumulative sequence boundaries
to variable-length FlashAttention. Outputs are repadded only when needed.

Unpadding changes storage and execution, not the model function. The teaching
implementation keeps regular $(B,T,d_{model})$ tensors and zeros padded hidden
states after every layer. This keeps the mathematics visible.

## Hands-on

Run [`examples/inspect_modernbert.py`](https://github.com/Alwin-Yang/ModernBERT-Notes/blob/main/examples/inspect_modernbert.py).
It prints every central shape, the global/local layer pattern, padding behavior,
and one attention weight outside the local window.

---

# 中文版本

本页推导把 BERT 变成 ModernBERT 的主要改动。记 batch size 为 $B$，序列长度为
$T$，hidden width 为 $d_{model}$，attention head 数为 $H$，每个 head 的宽度为
$d_k=d_{model}/H$。

## 融合 QKV projection

token embedding 和 LayerNorm 之后，hidden states 的形状为
$X_{B \times T \times d_{model}}$。一个不带 bias 的线性层同时产生 Q、K、V：

$$
[Q,K,V] = XW_{QKV}, \qquad
W_{QKV\,d_{model}\times3d_{model}}.
$$

具体 shape 变化见英文部分的代码块。attention 仍然是 scaled dot-product
attention：

$$
A = \operatorname{softmax}\left(
  \frac{QK^\top}{\sqrt{d_k}} + M
\right), \qquad O = AV.
$$

encoder 不使用 causal mask。每个真实 token 都能读取左右两侧的上下文。mask $M$
只负责移除 padding；在 local layer 中，它还会移除 sliding window 以外的位置。

## Rotary positional embeddings

原始 BERT 把学习式绝对位置向量加到 token embedding 上。ModernBERT 改为旋转
query 与 key 的成对特征。位置 $m$ 的旋转为：

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(m\omega_i)&-\sin(m\omega_i)\\
\sin(m\omega_i)& \cos(m\omega_i)
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix},
\qquad \omega_i=\theta^{-2i/d_k}.
$$

旋转保持向量长度不变，位置 $m$ 的 query 与位置 $n$ 的 key 的内积会自然包含相对
位移 $m-n$。global layer 使用 $\theta=160{,}000$，local layer 使用
$\theta=10{,}000$。RoPE 作用于决定相似度的 Q、K，不作用于承载内容的 V。

## 交替 attention

full attention 包含 $T^2$ 个 query-key 关系。在 8,192 token 下，每层都这样计算
代价很高。ModernBERT 按 `global, local, local` 重复：第 0、3、6 层使用 global
attention，其余层使用 local attention。

local layer 的 mask 为：

$$
M_{ij}=\begin{cases}
0,&|i-j|\leq w/2\\
-\infty,&\text{otherwise}.
\end{cases}
$$

官方 `local_attention=128` 表示左右各 64 的 half-window。真正的 sliding-window
kernel 把局部层复杂度从 $O(T^2)$ 降到 $O(Tw)$，周期性的 global layer 则负责
长距离信息传播。

!!! warning "mask 正确不等于获得加速"

    如果先算完整 $T\times T$ score matrix 再 mask，结果是对的，但计算仍然是二次的。
    只有不计算被遮蔽位置的 kernel 才能获得实际加速。

## Pre-LayerNorm 残差结构

ModernBERT 使用 Pre-LayerNorm：

$$
x' = x + \operatorname{Attention}(\operatorname{LN}(x)),
$$

$$
x'' = x' + \operatorname{MLP}(\operatorname{LN}(x')).
$$

残差主路径接近 identity map，有利于深层网络的 gradient flow。embedding 已经先做
LayerNorm，所以第 0 层跳过重复的 attention norm；整个 stack 末尾还有 final
LayerNorm。

## GeGLU feed-forward layer

GeGLU 同时生成 value 与 gate：

$$
[a,g]=xW_{in}, \qquad
\operatorname{GeGLU}(x)=W_{out}(\operatorname{GELU}(a)\odot g).
$$

完整 shape 见英文部分。gate 可以针对每个 token、每个 intermediate feature 决定
信息通过或被抑制。ModernBERT-base 使用 $d_{model}=768$、$d_{ff}=1152$。

## Unpadding 与 FlashAttention

padding token 没有语义，却会消耗计算。ModernBERT 的优化路径在 embedding 前移除
padding，把真实 token 打包进连续 buffer，并把各序列边界传给 variable-length
FlashAttention。只有需要时才 repad 输出。

unpadding 改变的是存储和执行方式，不改变模型函数。本仓库为了让数学清晰，仍使用普通
$(B,T,d_{model})$ tensor，并在每层后把 padding hidden state 置零。

## 动手实现

运行 [`examples/inspect_modernbert.py`](https://github.com/Alwin-Yang/ModernBERT-Notes/blob/main/examples/inspect_modernbert.py)。
它会打印关键 shape、global/local layer pattern、padding 行为，以及一个 window 外的
attention weight。

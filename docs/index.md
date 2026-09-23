# ModernBERT Notes

Learn how a modern bidirectional encoder becomes a fast typed decision model.
The path starts with BERT and ModernBERT, follows the option-marker head used by Laya,
and ends with the public Laya interpretation of Reinforcement Learning for
Calibrated Decisions (RLCD).

These notes have two halves. The site explains the mathematics and design. The
Python package in this repository supplies small CPU-runnable implementations
and tests for every central idea.

## Learning path

1. **BERT**: the encoder stack, bidirectional attention, and masked language
   modeling.
2. **ModernBERT internals**: RoPE, alternating local/global attention, GeGLU,
   Pre-LayerNorm, and unpadding.
3. **Typed decisions**: turn an encoder into a model that scores bounded options
   instead of generating text.
4. **RLCD**: train probability reports with proper scoring rules, policy
   gradients, and held-out calibration.
5. **Practice**: inspect tensor shapes, run the toy policy, and use tests to
   check each invariant.

!!! important "What this site does not claim"

    TypeSafe has not published Jev's internal architecture or complete RLCD
    recipe. The decision-model chapters study Laya's public, independent
    implementation. They do not claim to reproduce Jev.

---

# 中文版本

学习一个现代双向 encoder 如何变成快速的 typed decision model。路线从 BERT 和
ModernBERT 开始，接着分析 Laya 使用的 option-marker head，最后理解 Laya 公开版本的
Reinforcement Learning for Calibrated Decisions（RLCD）。

笔记分成两半：网页负责公式、原理和架构；仓库中的 Python package 提供可在 CPU
运行的小型实现和测试，用来验证每个核心概念。

## 学习路线

1. **BERT**：encoder stack、双向 attention 与 masked language modeling。
2. **ModernBERT 内部结构**：RoPE、交替 local/global attention、GeGLU、
   Pre-LayerNorm 与 unpadding。
3. **Typed decisions**：把 encoder 变成给有限候选项打分，而不是生成文字的模型。
4. **RLCD**：用 proper scoring rule、policy gradient 和独立校准集训练概率输出。
5. **动手实践**：查看 tensor 形状、运行 toy policy，并用测试检查关键不变量。

!!! important "本站不作出的声明"

    TypeSafe 没有公开 Jev 的内部架构或完整 RLCD recipe。decision-model 章节研究的是
    Laya 的公开独立实现，并不声称复现 Jev。

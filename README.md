# ModernBERT Notes

这是一个与 [Alwin-Yang/DL-Notes](https://github.com/Alwin-Yang/DL-Notes)
结构一致的双语 MkDocs 学习网站：网页负责数学推导和知识路线，Python package、examples
与 tests 负责验证实现。每个网页先写英文，再提供对应的中文版本。

目标不是复刻一个生产模型，而是回答三件事：

1. ModernBERT 相比 BERT 到底改了什么？
2. Laya 怎样把 ModernBERT 变成不生成文本的 typed decision model？
3. Laya 公开实现里的 RLCD 怎样用 proper scoring rule 训练概率输出？

> [!IMPORTANT]
> TypeSafe 在 2026-09-15 公布了 Jev 和“Reinforcement Learning for Calibrated
> Decisions”这个名字，但没有公开 Jev 的模型结构、权重或完整训练算法。本仓库不声称
> 复现 Jev。这里的 RLCD 代码学习的是 **Laya 的公开实现**；它是一个 Jev-compatible
> 的独立开源实现。

## 推荐学习顺序

1. 阅读 [BERT](docs/modernbert/bert.md)、[ModernBERT Overview](docs/modernbert/index.md)
   与 [Architecture](docs/modernbert/architecture.md)
2. 运行最小 ModernBERT：`python -m examples.inspect_modernbert`
3. 阅读 [Laya Decision Head](docs/decisions/laya-head.md) 与
   [RLCD](docs/decisions/rlcd.md)
4. 运行 RLCD 小实验：`python -m examples.train_rlcd_toy`
5. 跑测试：`python -m pytest -q`

## 仓库结构

```text
ModernBERT-Notes/
├── docs/                  # 双语网页内容
├── mkdocs.yml             # 网站导航与主题
├── modernbert_notes/
│   ├── model.py       # RoPE、local/global attention、GeGLU、Pre-LN
│   ├── decision.py    # Laya 风格 option-marker decision head
│   └── rlcd.py        # proper reward、REINFORCE loss、temperature scaling
├── examples/
│   ├── inspect_modernbert.py
│   └── train_rlcd_toy.py
├── tests/
└── .github/workflows/     # GitHub Pages 自动发布
```

## 快速开始

预览和严格构建网页：

```bash
make docs-only
make serve       # http://127.0.0.1:3408/
make build
```

本机已有 PyTorch 时，无需下载任何模型权重即可运行学习代码：

```bash
python -m examples.inspect_modernbert
python -m examples.train_rlcd_toy
python -m pytest -q
```

也可以安装为 editable package：

```bash
python -m pip install -e '.[dev]'
```

所有默认配置都很小，只用于理解与单元测试。官方 ModernBERT-base 是 22 层、隐藏维度
768、12 个头、约 149M 参数；不要把这里的随机 tiny model 当作可用语言模型。

## 事实边界

- **ModernBERT**：公开论文、代码和权重，架构细节可验证。
- **Laya**：Apache-2.0 开源实现；英文 checkpoint 以 ModernBERT-large 为 backbone，
  公开了输入格式、decision head、训练 notebook 和校准代码。
- **Jev**：公开的是产品行为——输入 state 与 typed questions，输出 choice / score /
  noul 的概率决策；内部架构与完整 RLCD recipe 未公开。

主要资料：

- [ModernBERT paper](https://arxiv.org/abs/2412.13663)
- [ModernBERT official repository](https://github.com/AnswerDotAI/ModernBERT)
- [Hugging Face ModernBERT docs](https://huggingface.co/docs/transformers/model_doc/modernbert)
- [Laya source](https://github.com/NandhaKishorM/laya)
- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

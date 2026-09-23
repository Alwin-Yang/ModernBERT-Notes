# AGENTS.md

Guidance for contributors and coding agents working in this repository.

## Repository purpose

This is a public bilingual MkDocs site for learning ModernBERT, typed decision
models, Laya, and RLCD. It also contains a small runnable PyTorch implementation.
The site builds understanding; the code verifies it.

## Layout

```text
docs/                 published notes site
modernbert_notes/     CPU-runnable teaching implementation
examples/             guided command-line experiments
tests/                mathematical and implementation invariants
mkdocs.yml            navigation and site configuration
```

Every new site page must appear in `nav` in `mkdocs.yml`. `make build` must pass
with `--strict` before a change is complete.

## Language and writing

Every page is bilingual. English comes first and is the source of truth. After
the English body, add `---` and `# 中文版本`, then mirror the same sections in
the same order. Chinese text keeps technical terms such as LayerNorm, logits,
and policy gradient in English.

- Explain why each step exists, not only what it computes.
- Define symbols before using them and include tensor dimensions.
- Use Mermaid for data flow and plain code blocks for shape walkthroughs.
- Code examples must run. Do not present pseudocode as verified code.
- Separate public facts from inference. Never describe Laya as Jev or claim
  that Laya's public RLCD recipe is Jev's undisclosed training algorithm.
- Q&A pages should ask why and what breaks, not test vocabulary recall.

## Validation

```bash
make build
make test
make examples
```

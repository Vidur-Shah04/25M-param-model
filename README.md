# LMA Individual Project — Gujarati & Assamese

## Building and Analyzing Monolingual Transformer LMs

This repository contains two **independent monolingual decoder-only Transformer language models**:

- **Model H:** Gujarati — higher-resource language
- **Model L:** Assamese — lower-resource language

Each language has its own corpus, preprocessing pipeline, tokenizer, vocabulary, model, training pipeline, and finetuning pipeline. No shared tokenizer, vocabulary, corpus, or weights are used.

---

# Repository Structure

```text
.
├── Gujarati/
│   ├── data/
│   ├── tokenizer/
│   ├── tokenizer_clean/
│   ├── model/
│   ├── generator/
│   ├── sft/
│   ├── evaluation/
│   └── checkpoints/          # large checkpoints kept outside Git
│
├── Assamese/
│   ├── data/
│   ├── tokenizer/
│   ├── tokenizer_clean/
│   ├── model/
│   ├── generator/
│   ├── sft/
│   ├── evaluation/
│   └── checkpoints/          # large checkpoints kept outside Git
│
├── phase2_figures/
├── report/
├── requirements.txt
└── README.md
```

---

# Phase 1 — Data Collection and Tokenizer Construction

Both languages use separate monolingual datasets and independently trained SentencePiece tokenizers.

## Dataset Links

### Gujarati

Google Drive:

https://drive.google.com/drive/folders/129j6mhhpm2RhNcGUcCj3BajeBfRYW93p?usp=sharing

Place the extracted `splits_v2` at:

```text
Gujarati/data/splits_v2/
```

### Assamese

Google Drive:

https://drive.google.com/file/d/16zlWmnr8lRZn80Mc2s__1UXltB18AfsB/view?usp=sharing

Place the extracted `splits_v2` at:

```text
Assamese/data/splits_v2/
```

## Tokenizers

The final project models use **separate 10,000-token SentencePiece BPE vocabularies**, one tokenizer per language.

Tokenizer-related code and files are stored in the corresponding language directories.

The two models never share tokenizer vocabulary or tokenizer state.

---

# Phase 2 — Transformer Implementation and Pretraining

## Architecture

Both language models use the same architectural dimensions while remaining completely independent in their data, tokenizer, and weights.

| Parameter | Value |
|---|---:|
| Vocabulary size | 10,000 |
| Model dimension (`d_model`) | 512 |
| Transformer layers | 7 |
| Attention heads | 8 |
| Head dimension | 64 |
| FFN dimension (`d_ff`) | 1,792 |
| Context length | 512 |
| Dropout | 0.1 |
| Parameters | 25,613,056 |

The Transformer is implemented from scratch in PyTorch using primitive components. No Hugging Face Transformer classes, `nn.Transformer*` modules, or pretrained language models/tokenizers are used.

## Forward Pass

```text
Token IDs
   ↓
Token embedding + learned positional embedding
   ↓
Embedding dropout
   ↓
7 × Transformer block
      ├── Pre-LayerNorm
      ├── Multi-head causal self-attention
      │      ├── Q, K, V projections
      │      ├── 8 attention heads
      │      ├── scaled dot-product attention
      │      ├── causal upper-triangular mask
      │      └── output projection
      └── Pre-LayerNorm + position-wise FFN
   ↓
Final LayerNorm
   ↓
Vocabulary projection
   ↓
Next-token logits
   ↓
Causal language-model cross-entropy
```

## Pretraining Configuration

| Setting | Value |
|---|---:|
| Batch size | 1 |
| Gradient accumulation | 32 |
| Tokens / optimizer step | 16,384 |
| Target pretraining tokens | ~500M |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Minimum learning rate | 3e-5 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| LR schedule | 2,000-step warmup + cosine decay |
| CUDA precision | float16 AMP |
| Context length | 512 |
| Final training step | 30,518 |

## Pretrained Checkpoints

Large model checkpoints are not committed to Git because of their size.

Kaggle checkpoint/evaluation datasets:

- **Gujarati:** https://www.kaggle.com/datasets/vidurshah04/guj-ckcpts-and-json
- **Assamese:** https://www.kaggle.com/datasets/vidurshah04/assam-ckcpts-and-json

Expected local locations for Phase 2 checkpoints:

```text
Gujarati/checkpoints/checkpoint_030518_gujarati.pt
Assamese/checkpoints/checkpoint_030518_assamese.pt
```

## Phase 2 Evaluation

Available evaluation code includes:

```text
Gujarati/evaluation/
Assamese/evaluation/
```

Main evaluation components include checkpoint validation, full validation LM evaluation, BPB, generation metrics, attention analysis, aggregate attention statistics, and causal-mask tests.

A final Phase 2 audit was performed and produced **56/56 PASS**.

## Phase 2 Validation Results

| Metric | Gujarati | Assamese |
|---|---:|---:|
| Validation prediction tokens | 199,677,338 | 51,967,678 |
| Cross-entropy | 5.685608 | 5.375766 |
| Perplexity | 294.5969 | 216.1053 |
| BPB | 0.816889 | 0.761198 |
| Parameters | 25,613,056 | 25,613,056 |
| Final checkpoint step | 30,518 | 30,518 |

Generation evaluation and attention analysis outputs are stored in the corresponding `evaluation/` directories.

---

# Phase 3 — Reasoning Finetuning and Attention Analysis

Each pretrained model was finetuned **independently on reasoning data in its own language**. No finetuning data was mixed across languages.

## Synthetic Reasoning Dataset

Reasoning examples are generated programmatically using controlled templates with known ground-truth answers.

Tasks include:

```text
direct_comparison
greater_smaller
equal
two_hop
three_hop
ordering
reversed_wording
distractor
```

Each language has:

```text
data_sft/
    train.jsonl
    val.jsonl
    test.jsonl

generator/
    generate_<language>.py
    templates_<language>.py
```

Dataset sizes for each language:

| Split | Examples |
|---|---:|
| Train | 6,400 |
| Validation | 800 |
| Test | 800 |

The reasoning dataset is monolingual and uses the target-language script and phrasing.

## Gujarati SFT

Notebook:

```text
Gujarati/sft/gujarati-train-phase3 98.ipynb
```

Configuration:

| Setting | Value |
|---|---:|
| Epochs | 5 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Betas | (0.9, 0.95) |
| Gradient clipping | 1.0 |
| Vocabulary | 10,000 |
| EOS ID | 2 |

Final test result:

**796 / 800 = 99.50%**

Metrics:

```text
gujarati_phase3_metrics.json
Gujarati/sft/gujarati_phase3_metrics 98.json
```

Predictions:

```text
gujarati_phase3_test_predictions.jsonl
```

## Assamese SFT

Notebook:

```text
Assamese/sft/assamese-train-phase-3.ipynb
```

Configuration:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Betas | (0.9, 0.95) |
| Gradient clipping | 1.0 |
| Vocabulary | 10,000 |
| EOS ID | 2 |

Final test result:

**787 / 800 = 98.375%**

Metrics:

```text
assamese_phase3_metrics.json
Assamese/sft/assamese_phase3_metrics.json
```

Predictions:

```text
assamese_phase3_test_predictions.jsonl
```

---

# Phase 3 Checkpoints

Large SFT checkpoints are kept outside Git.

### Gujarati

https://www.kaggle.com/datasets/vidurshah04/guj-ckcpts-and-json

### Assamese

https://www.kaggle.com/datasets/vidurshah04/assam-ckcpts-and-json

Each dataset contains:

```text
phase3_<language>_sft_best.pt
phase3_<language>_sft_final.pt
<language>_phase3_metrics.json
<language>_phase3_test_predictions.jsonl
```

Resume-capable checkpoint variants are also maintained locally in the corresponding `checkpoints/` directories.

---

# Pretrained vs Finetuned Reasoning Evaluation

| Language | Pretrained | Finetuned |
|---|---:|---:|
| Gujarati | 0 / 800 = 0.00% | 796 / 800 = 99.50% |
| Assamese | 0 / 800 = 0.00% | 787 / 800 = 98.375% |

The finetuned models show a large improvement on the synthetic reasoning test suite relative to the corresponding pretrained checkpoints.

---

# Phase 3 Attention Analysis

The Phase 2 attention toolkit was reused to compare **pretrained vs finetuned** models on reasoning prompts.

For each language, comparisons include:

- Early layer: **Layer 1**
- Late layer: **Layer 7**
- Multiple heads: **Heads 0 and 4**
- Pretrained and SFT heatmaps
- Attention entropy
- Mean attention distance
- Aggregate layer/head statistics

Artifacts are stored under:

```text
Gujarati/evaluation/figures/attention_reasoning/
Assamese/evaluation/figures/attention_reasoning/

Gujarati/evaluation/results/attention_sft/
Assamese/evaluation/results/attention_sft/
```

Attention analysis is treated as evidence of attention patterns rather than proof of causal semantic roles for individual heads.

---

# Error Analysis

Final prediction JSONL files are preserved for reproducible error analysis.

Observed failures:

- **Gujarati:** 4 errors out of 800; ordering accounts for 2 of the 4 errors.
- **Assamese:** 13 errors out of 800; 12 of the 13 errors are ordering failures, with the remaining error on reversed wording.

Full predictions:

```text
gujarati_phase3_test_predictions.jsonl
assamese_phase3_test_predictions.jsonl
```

---

# Reproduction

## Install dependencies

```bash
pip install -r requirements.txt
```

## Obtain data

Download the Phase 1 `splits_v2` datasets from the links above and place them in the language-specific locations described in Phase 1.

## Obtain checkpoints

Download the pretrained checkpoints from the linked checkpoint datasets and place them at:

```text
Gujarati/checkpoints/checkpoint_030518_gujarati.pt
Assamese/checkpoints/checkpoint_030518_assamese.pt
```

For Phase 3, the SFT checkpoint artifacts are available through the Gujarati and Assamese Kaggle datasets linked above.

## Generate reasoning data

Use:

```text
Gujarati/generator/generate_gujarati.py
Assamese/generator/generate_assamese.py
```

The generated splits are stored in the corresponding `data_sft/` directories.

## Run evaluation

Use the evaluation scripts under:

```text
Gujarati/evaluation/
Assamese/evaluation/
```

For Phase 3 attention comparison:

```text
Gujarati/evaluation/attention_reasoning_compare.py
Assamese/evaluation/attention_reasoning_compare.py
```

---

# Report

The `report/` directory contains project reports and supporting figures/tables.

The final Phase 3 report consolidates the Phase 1, Phase 2 and Phase 3 results.

---

# Requirements

Python 3.12 was used.

```bash
pip install -r requirements.txt
```

---

# Constraints Followed

- Two independent monolingual language models.
- Separate datasets and tokenizers.
- No pretrained language models.
- No pretrained tokenizers.
- No Hugging Face Transformer model classes.
- Decoder-only Transformer implemented from scratch in PyTorch.
- Causal self-attention and masking implemented explicitly.
- Independent pretraining and finetuning pipelines.
- Checkpoint-based training recovery supported.

# LMA Individual Project — Gujarati & Assamese

This repository contains two independent monolingual decoder-only language-model pipelines:

- **Model H:** Gujarati — higher-resource language
- **Model L:** Assamese — lower-resource language

Each language has its own corpus, preprocessing pipeline, tokenizer, vocabulary and model weights. Phase 1 established the final cleaned corpora and SentencePiece tokenizers; Phase 2 implements, pretrains and evaluates the two Transformer language models independently.

## Phase 1: Data Collection and Tokenizer Construction

The Phase 1 pipeline covers corpus collection, cleaning, exact and near-deduplication, document-level train/validation/test splitting, SentencePiece tokenizer training and tokenizer evaluation. The final cleaned datasets are supplied separately through Google Drive so that reproduction does not require repeating the original collection step.

### Phase 1 dataset links

**Gujarati**  
https://drive.google.com/drive/folders/129j6mhhpm2RhNcGUcCj3BajeBfRYW93p?usp=sharing

Place the extracted `splits_v2` at:

```text
Gujarati/data/splits_v2/
```

**Assamese**  
https://drive.google.com/file/d/16zlWmnr8lRZn80Mc2s__1UXltB18AfsB/view?usp=sharing

Place the extracted `splits_v2` at:

```text
Assamese/data/splits_v2/
```

The repository retains the Phase 1 collection and preprocessing scripts for provenance.

### Phase 1 final tokenizer decision

SentencePiece BPE vocabularies of 16K, 24K and 32K were evaluated independently for both languages. All six tokenizers had 0% UNK on held-out text. Fertility decreased as vocabulary size increased, but larger vocabularies consume more of the approximately 25M model budget. The working 24K choice was therefore retained when the complete Transformer architecture was fixed.

Relevant Phase 1 values:

| Language | 16K fertility | 24K fertility | 32K fertility | UNK |
|---|---:|---:|---:|---:|
| Gujarati | 1.6603 | 1.5701 | 1.5166 | 0% |
| Assamese | 1.8791 | 1.7368 | 1.6633 | 0% |

## Phase 2: Model Implementation, Pretraining and Evaluation

Phase 2 implements two **fully independent** GPT-style decoder-only Transformers from scratch in PyTorch. No Hugging Face model classes, `nn.Transformer*` modules, or pre-built attention blocks are used. Only primitive components such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and `nn.Dropout` are used.

### Final architecture

Both models use the same architecture, but their tokenizers, vocabulary instances, data and weights remain independent.

| Parameter | Value |
|---|---:|
| Vocabulary | 24,000 |
| Model dimension (`d_model`) | 512 |
| Transformer layers | 5 |
| Attention heads | 8 |
| Head dimension | 64 |
| FFN dimension (`d_ff`) | 1,536 |
| Context length | 512 |
| Dropout | 0.1 |
| Parameters | 25,689,088 |

The 24K vocabulary was kept after the full architecture was fixed because tied input/output embeddings cost 24,000 × 512 = 12,288,000 parameters, while a 32K vocabulary would push the same architecture beyond the target model budget. The final configuration therefore keeps substantial capacity in the five Transformer blocks while remaining close to the approximately 25M target.

### Transformer implementation

Each input sequence follows this path:

```text
Token IDs
   ↓
Token embedding + learned positional embedding
   ↓
Embedding dropout
   ↓
5 × Transformer block
      ├─ Pre-LayerNorm
      ├─ Multi-head causal self-attention
      │    ├─ Q, K, V projections
      │    ├─ reshape to 8 heads × 64 dimensions
      │    ├─ scaled dot-product attention
      │    ├─ causal upper-triangular mask
      │    └─ output projection
      └─ Pre-LayerNorm + position-wise FFN
   ↓
Final LayerNorm
   ↓
Tied vocabulary projection
   ↓
Logits over 24,000 tokens
   ↓
Causal next-token cross-entropy
```

### Training configuration

The final training setup used:

| Setting | Value |
|---|---:|
| Batch size | 1 |
| Gradient accumulation | 32 |
| Tokens per optimizer step | 16,384 |
| Target pretraining tokens | approximately 500M |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Minimum LR | 3e-5 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| LR schedule | 2,000-step linear warmup + cosine decay |
| Precision on CUDA | float16 AMP |
| Validation/checkpoint cadence | 1,500 optimizer steps |

With 1 × 32 × 512 = 16,384 source tokens per optimizer step, the approximate 500M-token target gives `ceil(500,000,000 / 16,384) = 30,518` optimizer steps. The final checkpoints are therefore `checkpoint_030518_gujarati.pt` and `checkpoint_030518_assamese.pt`.

### Checkpoints and large files

Final checkpoints are intentionally **not committed to Git**. They are protected by `.gitignore` and should be stored/downloaded separately from Google Drive or the designated project storage. Add the final Drive links here before submission if they are not already present in the repository:

- Gujarati checkpoint: <https://kaggle.com/datasets/7675e723bf3ee6067b0bdb2863eda06180da5f9a6ebdefb6b078522d70cb3707>
- Assamese checkpoint: <https://kaggle.com/datasets/a66bb0be411e6283524dbc18977ff9cd952a4741cd82195f68509a1f2d417788>

Expected local locations:

```text
Gujarati/checkpoints/checkpoint_030518_gujarati.pt
Assamese/checkpoints/checkpoint_030518_assamese.pt
```

### Phase 2 reproduction / evaluation

After cloning the repository, installing dependencies and placing the final `splits_v2` data and checkpoints in the expected locations, the following commands reproduce the main Phase 2 checks.

#### 1. Verify the checkpoints

```bash
python Gujarati/evaluation/evaluate_checkpoint.py
python Assamese/evaluation/evaluate_checkpoint.py
```

These verify the checkpoint state, configuration, parameter count and a finite forward pass.

#### 2. Run full validation LM evaluation

```bash
python Gujarati/evaluation/evaluate_lm.py
python Assamese/evaluation/evaluate_lm.py
```

This streams `val_*.txt`, tokenizes with the 24K SentencePiece model, evaluates next-token cross-entropy over the full validation stream and reports perplexity.

#### 3. Compute BPB

```bash
python Gujarati/evaluation/bpb.py
python Assamese/evaluation/bpb.py
```

BPB converts the token-level cross-entropy into bits per UTF-8 byte, providing a metric less tied to the tokenization boundary than perplexity.

#### 4. Generate validation continuations

```bash
python Gujarati/evaluation/generate.py
python Assamese/evaluation/generate.py
```

The generation script uses held-out validation prompts and temperatures 0.0, 0.5, 1.0 and 1.5, with 100 generated tokens per sample.

#### 5. Compute automatic generation metrics

```bash
python Gujarati/evaluation/generation_metrics_standard.py
python Assamese/evaluation/generation_metrics_standard.py
```

The final reported generation table uses 50 held-out examples, a 128-token prompt, a 100-token reference continuation and a 100-token generated continuation for each temperature.

#### 6. Inspect attention

```bash
python Gujarati/evaluation/attention_analysis.py
python Assamese/evaluation/attention_analysis.py
python Gujarati/evaluation/attention_analysis_aggregate.py
python Assamese/evaluation/attention_analysis_aggregate.py
```

The first analysis records all 5 × 8 heads for one 128-token validation sequence and produces selected head heatmaps. The aggregate analysis repeats the statistics over 10 validation sequences and saves layer/head CSV summaries plus entropy and distance heatmaps.

#### 7. Verify causal masking

```bash
python Gujarati/evaluation/causal_mask_test.py
python Assamese/evaluation/causal_mask_test.py
```

These tests change a future input token and verify that logits at earlier positions remain unchanged.

#### 8. Run the final combined audit

```bash
python final_phase2_audit.py
```

The final audit checks files, checkpoint state, configuration, tokenizer size, parameter count, forward shapes, attention extraction and causal masking for both models. The final run produced **56/56 PASS**.

### Final Phase 2 validation results

| Metric | Gujarati | Assamese |
|---|---:|---:|
| Validation prediction tokens | 199,677,338 | 51,967,678 |
| Cross-Entropy | 5.685608 | 5.375766 |
| Perplexity | 294.5969 | 216.1053 |
| BPB | 0.816889 | 0.761198 |
| Model parameters | 25,689,088 | 25,689,088 |
| Final checkpoint step | 30,518 | 30,518 |

Cross-language perplexity/BPB values should be interpreted with care because the validation corpora and linguistic/tokenization characteristics differ. The metrics are most directly used to evaluate each model on its own held-out distribution.

### Generation evaluation

The final generation evaluation tracks reference overlap (BLEU-4, chrF, ROUGE-L) and diversity/repetition (repetition rate, Distinct-1 and Distinct-2). The results show a strong temperature trade-off: 0.0 is highly repetitive, 0.5 improves diversity but still repeats, 1.0 provides the strongest balance for these experiments, and 1.5 maximizes diversity at the cost of reference overlap/coherence.

### Attention analysis

The final attention analysis reports mean entropy and mean attention distance for every layer/head over 10 validation sequences. Low entropy indicates concentrated attention; higher entropy indicates attention spread over more positions. Mean attention distance is the expected absolute positional gap under the attention distribution.

The analysis identifies candidate concentrated/local heads and candidate long-range heads, but the statistics alone do **not** prove a semantic role for any head.

### Phase 2 artifact layout

```text
Gujarati/
  model/
    config.py
    dataset.py
    train.py
    transformer.py
  evaluation/
    evaluate_checkpoint.py
    evaluate_lm.py
    generate.py
    generation_metrics_standard.py
    bpb.py
    attention_analysis.py
    attention_analysis_aggregate.py
    causal_mask_test.py
    figures/attention/
    results/

Assamese/
  model/
    config.py
    dataset.py
    train.py
    transformer.py
  evaluation/
    evaluate_checkpoint.py
    evaluate_lm.py
    generate.py
    generation_metrics_standard.py
    bpb.py
    attention_analysis.py
    attention_analysis_aggregate.py
    causal_mask_test.py
    figures/attention/
    results/

final_phase2_audit.py
report/
```

## Report

- Phase 1 report: `report/`
- Phase 2 report: `report/Phase_2_Model_Implementation_Pretraining_and_Evaluation.docx`

## Requirements

Python 3.12 was used. Install the repository dependencies with:

```bash
pip install -r requirements.txt
```

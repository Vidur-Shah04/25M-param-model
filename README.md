# LMA Individual Project --- Gujarati & Assamese

## Phase 1: Data Collection and Tokenizer Construction

This repository contains two independent monolingual pipelines:

-   **Model H:** Gujarati --- higher-resource language
-   **Model L:** Assamese --- lower-resource language

Each language has its own corpus, preprocessing pipeline, tokenizer and
vocabulary. The Phase 1 target is approximately **500M pretraining
tokens per language**.

## 1. Reproduction

### Clone

``` bash
git clone https://github.com/CL3-410/individual-project-Vidur-Shah04.git
cd individual-project-Vidur-Shah04
```

### Install dependencies

Python 3.12 was used.

``` bash
pip install -r requirements.txt
```

### Obtain final datasets

The final cleaned datasets are supplied through Google Drive so
reproduction does not require repeating the original data collection.

**Gujarati**

https://drive.google.com/drive/folders/129j6mhhpm2RhNcGUcCj3BajeBfRYW93p?usp=sharing

Place the extracted `splits_v2` at:

``` text
Gujarati/data/splits_v2/
```

**Assamese**

https://drive.google.com/file/d/16zlWmnr8lRZn80Mc2s\_\_1UXltB18AfsB/view?usp=sharing

Place the extracted `splits_v2` at:

``` text
Assamese/data/splits_v2/
```

## 2. Original Data Sources

### Gujarati

Manual: - Archive.org - eBooks - Vishwakosh - Wiki dumps - Wikisource

Premade: - IITB-IndicMonoDoc

### Assamese

Manual: - Archive.org

Premade: - IITB-IndicMonoDoc - Sangraha - HPLT

The collection scripts are retained for provenance. They do not need to
be rerun when using the supplied final datasets.

## 3. Cleaning and Deduplication

The preprocessing pipeline:

1.  Converts source data into a common document representation.
2.  Normalizes and filters text for the target script.
3.  Removes invalid or very short documents.
4.  Removes exact duplicates.
5.  Removes near duplicates using MinHash.
6.  Creates document-level train/validation/test splits.

Parameters:

  Parameter                    Value
  ---------------------------- -----------------------------------------
  Minimum document size        50 words
  Script filtering threshold   30% target-script alphabetic characters
  Exact duplicate detection    Hash matching
  Near-duplicate method        MinHash
  MinHash threshold            0.85
  MinHash permutations         64
  Shingle size                 5 words
  Split                        Approximately 80/10/10

To reproduce preprocessing from raw data:

``` bash
python clean_and_dedupe.py
```

This can be skipped when using the supplied `splits_v2` data.

## 4. Final Dataset Statistics

### Gujarati

  Split                  Words
  ------------ ---------------
  Train          1,014,306,216
  Validation       133,399,162
  Test             130,941,945

Final source accounting:

  Source              Type            Documents             Words
  ------------------- --------- --------------- -----------------
  Archive.org         Manual              1,934        65,304,956
  eBooks              Manual                 14           702,628
  Vishwakosh          Manual             25,143        15,377,594
  Wiki dumps          Manual             18,587         6,028,936
  Wikisource          Manual                109         3,857,632
  IITB-IndicMonoDoc   Premade         1,110,905       506,990,013
  **Total**                       **1,156,692**   **598,261,759**

### Assamese

  Split                Words
  ------------ -------------
  Train          280,833,391
  Validation      32,466,286
  Test            33,631,378

Final source accounting:

  Source              Type            Documents             Words
  ------------------- --------- --------------- -----------------
  Archive.org         Manual              4,108       152,394,605
  IITB-IndicMonoDoc   Premade           138,471        52,504,259
  Sangraha            Premade           839,987        88,303,035
  HPLT                Premade           145,556        53,729,156
  **Total**                       **1,128,122**   **346,931,055**

## 5. Manual Data Requirement

Manual-token estimates use the mean fertility of the 16K, 24K and 32K
tokenizers.

  ---------------------------------------------------------------------------
  Language   Manual words         Mean    Estimated  500M target     Manual /
                             fertility       manual                    target
                                             tokens              
  ---------- ------------ ------------ ------------ ------------ ------------
  Gujarati         91.27M       1.5823      144.42M         500M   **28.88%**

  Assamese        152.39M       1.7597      268.17M         500M   **53.63%**
  ---------------------------------------------------------------------------

Both satisfy the required 20% manual-data contribution relative to the
nominal 500M-token training target.

## 6. Train Tokenizers

Separate SentencePiece BPE tokenizers were trained from scratch.

Vocabulary sizes evaluated:

-   16K
-   24K
-   32K

Run:

``` bash
python train_tokenizer.py
```

Run independently for Gujarati and Assamese using their respective
training shards.

## 7. Evaluate Tokenizers

Run:

``` bash
python vocab_size_metrics.py
```

The evaluation reports fertility, token count, UNK statistics,
characters/token, tokens/character and processing time.

Measured fertility:

  Language        16K      24K          32K
  ---------- -------- -------- ------------
  Gujarati     1.6603   1.5701   **1.5166**
  Assamese     1.8791   1.7368   **1.6633**

All six evaluated tokenizers produced **0 unknown tokens** on the
held-out test sets.

## 8. Vocabulary Selection

Increasing vocabulary size reduces fertility, but larger vocabularies
consume more of the approximately 25M-parameter model budget.

For illustration, assuming `d_model = 512` and tied input/output
embeddings:

  -----------------------------------------------------------------------
  Vocabulary              Embedding       Share of 25M   Remaining budget
                         parameters             budget 
  -------------- ------------------ ------------------ ------------------
  16K                         8.19M              32.8%             16.81M

  24K                        12.29M              49.2%             12.71M

  32K                        16.38M              65.5%              8.62M
  -----------------------------------------------------------------------

Thus 16K is cheapest, 32K has the best fertility, and 24K provides an
intermediate trade-off. **24K is the working candidate for Phase 1; the
final choice will be revisited in Phase 2 after fixing the complete
Transformer architecture and parameter count.**

## 9. Phase 1 Artifacts

The repository contains:

1.  Dataset collection scripts.
2.  Cleaning and deduplication pipeline.
3.  Per-language dataset statistics.
4.  Train/validation/test splits.
5.  SentencePiece tokenizer training code.
6.  Vocabulary files.
7.  Tokenizer model files.
8.  Tokenizer evaluation results.
9.  Phase 1 report and figures.

## 10. Recommended Reproduction Order

``` text
Clone repository
      ↓
Install requirements
      ↓
Download Gujarati splits_v2
Download Assamese splits_v2
      ↓
Verify directory structure
      ↓
Train Gujarati tokenizer
Train Assamese tokenizer
      ↓
Evaluate 16K / 24K / 32K
      ↓
Generate tokenizer statistics
      ↓
Review report/
```

The raw collection scripts are included for provenance and do not need
to be rerun when reproducing the final Phase 1 tokenizer results from
the supplied cleaned datasets.

## 11. Report

The Phase 1 report is stored under:

``` text
report/
```

It contains dataset methodology, source accounting,
cleaning/deduplication parameters, split statistics, manual-data
calculations, tokenizer comparison, vocabulary-size analysis and
figures.

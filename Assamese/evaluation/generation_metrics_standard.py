from pathlib import Path
import sys
import math
import csv
from collections import Counter

import torch
import sentencepiece as spm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.transformer import DecoderOnlyTransformer


# ============================================================
# CONFIGURATION
# ============================================================

LANGUAGE = "assamese"

DATA_DIR = PROJECT_DIR / "data" / "splits_v2"

TOKENIZER_PATH = (
    PROJECT_DIR
    / "tokenizer_clean"
    / "assamese_10k.model"
)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "checkpoint_030518_assamese.pt"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "results"
)

NUM_SAMPLES = 50
PROMPT_TOKENS = 128
REFERENCE_TOKENS = 100
GENERATE_TOKENS = 100

TEMPERATURES = [0.0, 0.5, 1.0, 1.5]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print(f"{LANGUAGE.upper()} STANDARD GENERATION EVALUATION")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(str(TOKENIZER_PATH)):
    raise RuntimeError(
        f"Could not load tokenizer: {TOKENIZER_PATH}"
    )

print(
    f"Vocabulary size: {sp.get_piece_size()}"
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

config = ModelConfig(
    **checkpoint["config"]
)

model = DecoderOnlyTransformer(
    config
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Checkpoint step: {checkpoint['step']}"
)

print(
    f"Parameters: {model.count_parameters():,}"
)


# ============================================================
# COLLECT VALIDATION SAMPLES
# ============================================================

validation_files = sorted(
    DATA_DIR.glob("val_*.txt")
)

if not validation_files:
    raise FileNotFoundError(
        f"No validation files found in {DATA_DIR}"
    )


samples = []

required_tokens = (
    PROMPT_TOKENS + REFERENCE_TOKENS
)

for file_path in validation_files:

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            ids = sp.encode(
                line,
                out_type=int,
            )

            if len(ids) >= required_tokens:

                prompt = ids[
                    :PROMPT_TOKENS
                ]

                reference = ids[
                    PROMPT_TOKENS:
                    PROMPT_TOKENS
                    + REFERENCE_TOKENS
                ]

                samples.append(
                    (prompt, reference)
                )

                if len(samples) >= NUM_SAMPLES:
                    break

    if len(samples) >= NUM_SAMPLES:
        break


if len(samples) < NUM_SAMPLES:
    raise RuntimeError(
        f"Only found {len(samples)} usable "
        f"validation samples."
    )


print(
    f"Validation samples: {len(samples)}"
)


# ============================================================
# GENERATION
# ============================================================

def generate(
    prompt_ids,
    max_new_tokens,
    temperature,
):

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):

        context = generated[
            :, -config.context_length:
        ]

        with torch.no_grad():

            logits, _ = model(context)

        next_logits = logits[:, -1, :]

        if temperature == 0.0:

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )

        else:

            scaled_logits = (
                next_logits / temperature
            )

            probabilities = torch.softmax(
                scaled_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=1,
        )

    return generated[
        0,
        len(prompt_ids):
    ].tolist()


# ============================================================
# TOKENIZATION FOR METRICS
# ============================================================

def word_tokens(text):

    return text.split()


# ============================================================
# BLEU-4
# ============================================================

def ngram_counts(tokens, n):

    return Counter(
        tuple(tokens[i:i+n])
        for i in range(
            len(tokens) - n + 1
        )
    )


def corpus_bleu(references, hypotheses):

    clipped_totals = [0, 0, 0, 0]
    total_hypothesis_ngrams = [0, 0, 0, 0]

    reference_length = 0
    hypothesis_length = 0

    for reference, hypothesis in zip(
        references,
        hypotheses,
    ):

        reference_length += len(reference)
        hypothesis_length += len(hypothesis)

        for n in range(1, 5):

            ref_counts = ngram_counts(
                reference,
                n,
            )

            hyp_counts = ngram_counts(
                hypothesis,
                n,
            )

            clipped = sum(
                min(
                    count,
                    ref_counts[ngram],
                )
                for ngram, count
                in hyp_counts.items()
            )

            clipped_totals[n - 1] += clipped

            total_hypothesis_ngrams[n - 1] += max(
                len(hypothesis) - n + 1,
                0,
            )

    precisions = []

    for clipped, total in zip(
        clipped_totals,
        total_hypothesis_ngrams,
    ):

        if total == 0 or clipped == 0:
            return 0.0

        precisions.append(
            clipped / total
        )

    geometric_mean = math.exp(
        sum(
            math.log(p)
            for p in precisions
        ) / 4
    )

    if hypothesis_length == 0:
        return 0.0

    if hypothesis_length > reference_length:
        brevity_penalty = 1.0

    else:
        brevity_penalty = math.exp(
            1.0
            - reference_length
            / hypothesis_length
        )

    return (
        brevity_penalty
        * geometric_mean
    )


# ============================================================
# chrF
# ============================================================

def character_ngrams(text, n):

    return Counter(
        text[i:i+n]
        for i in range(
            len(text) - n + 1
        )
    )


def chrf_score(reference, hypothesis):

    reference = reference.replace(
        " ", ""
    )

    hypothesis = hypothesis.replace(
        " ", ""
    )

    scores = []

    for n in range(1, 7):

        ref_counts = character_ngrams(
            reference,
            n,
        )

        hyp_counts = character_ngrams(
            hypothesis,
            n,
        )

        overlap = sum(
            min(
                count,
                ref_counts[ngram],
            )
            for ngram, count
            in hyp_counts.items()
        )

        hyp_total = sum(
            hyp_counts.values()
        )

        ref_total = sum(
            ref_counts.values()
        )

        if hyp_total == 0 or ref_total == 0:
            continue

        precision = (
            overlap / hyp_total
        )

        recall = (
            overlap / ref_total
        )

        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        scores.append(f_score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ============================================================
# ROUGE-L
# ============================================================

def lcs_length(a, b):

    previous = [0] * (len(b) + 1)

    for token_a in a:

        current = [0]

        for j, token_b in enumerate(b, 1):

            if token_a == token_b:

                current.append(
                    previous[j - 1] + 1
                )

            else:

                current.append(
                    max(
                        previous[j],
                        current[-1],
                    )
                )

        previous = current

    return previous[-1]


def rouge_l(reference, hypothesis):

    if not reference or not hypothesis:
        return 0.0

    lcs = lcs_length(
        reference,
        hypothesis,
    )

    precision = lcs / len(hypothesis)
    recall = lcs / len(reference)

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


# ============================================================
# REPETITION RATE
# ============================================================

def repetition_rate(tokens):

    if len(tokens) < 2:
        return 0.0

    bigrams = [
        tuple(tokens[i:i+2])
        for i in range(
            len(tokens) - 1
        )
    ]

    unique = len(set(bigrams))

    return (
        1.0
        - unique / len(bigrams)
    )


# ============================================================
# DISTINCT-N
# ============================================================

def distinct_n(tokens, n):

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(
            len(tokens) - n + 1
        )
    ]

    return (
        len(set(ngrams))
        / len(ngrams)
    )


# ============================================================
# RUN EACH TEMPERATURE
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_rows = []

for temperature in TEMPERATURES:

    print()
    print(
        "=" * 70
    )

    print(
        f"Temperature: {temperature}"
    )

    print(
        "=" * 70
    )

    references_word = []
    hypotheses_word = []

    references_text = []
    hypotheses_text = []

    repetition_scores = []
    distinct1_scores = []
    distinct2_scores = []

    for sample_index, (
        prompt,
        reference_ids,
    ) in enumerate(samples):

        generated_ids = generate(
            prompt,
            GENERATE_TOKENS,
            temperature,
        )

        reference_text = sp.decode(
            reference_ids
        )

        generated_text = sp.decode(
            generated_ids
        )

        reference_tokens = word_tokens(
            reference_text
        )

        generated_tokens = word_tokens(
            generated_text
        )

        references_word.append(
            reference_tokens
        )

        hypotheses_word.append(
            generated_tokens
        )

        references_text.append(
            reference_text
        )

        hypotheses_text.append(
            generated_text
        )

        repetition_scores.append(
            repetition_rate(
                generated_tokens
            )
        )

        distinct1_scores.append(
            distinct_n(
                generated_tokens,
                1,
            )
        )

        distinct2_scores.append(
            distinct_n(
                generated_tokens,
                2,
            )
        )

        print(
            f"Sample "
            f"{sample_index + 1}/"
            f"{len(samples)} complete"
        )


    # --------------------------------------------------------
    # BLEU
    # --------------------------------------------------------

    bleu = corpus_bleu(
        references_word,
        hypotheses_word,
    )


    # --------------------------------------------------------
    # chrF
    # --------------------------------------------------------

    chrf_scores = []

    for reference, hypothesis in zip(
        references_text,
        hypotheses_text,
    ):

        chrf_scores.append(
            chrf_score(
                reference,
                hypothesis,
            )
        )

    chrf = (
        sum(chrf_scores)
        / len(chrf_scores)
    )


    # --------------------------------------------------------
    # ROUGE-L
    # --------------------------------------------------------

    rouge_scores = []

    for reference, hypothesis in zip(
        references_word,
        hypotheses_word,
    ):

        rouge_scores.append(
            rouge_l(
                reference,
                hypothesis,
            )
        )

    rouge = (
        sum(rouge_scores)
        / len(rouge_scores)
    )


    # --------------------------------------------------------
    # DIVERSITY
    # --------------------------------------------------------

    repetition = (
        sum(repetition_scores)
        / len(repetition_scores)
    )

    distinct1 = (
        sum(distinct1_scores)
        / len(distinct1_scores)
    )

    distinct2 = (
        sum(distinct2_scores)
        / len(distinct2_scores)
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"BLEU-4:       {bleu:.6f}"
    )

    print(
        f"chrF:         {chrf:.6f}"
    )

    print(
        f"ROUGE-L:      {rouge:.6f}"
    )

    print(
        f"Repetition:   {repetition:.6f}"
    )

    print(
        f"Distinct-1:   {distinct1:.6f}"
    )

    print(
        f"Distinct-2:   {distinct2:.6f}"
    )


    # --------------------------------------------------------
    # SAVE GENERATED TEXT
    # --------------------------------------------------------

    temp_label = str(
        temperature
    ).replace(
        ".",
        "_",
    )

    generation_file = (
        RESULTS_DIR
        / f"assamese_standard_generation_"
          f"temp_{temp_label}.txt"
    )

    with open(
        generation_file,
        "w",
        encoding="utf-8",
    ) as f:

        for i in range(
            len(samples)
        ):

            f.write(
                f"===== SAMPLE {i + 1} =====\n"
            )

            f.write(
                "REFERENCE:\n"
            )

            f.write(
                references_text[i]
                + "\n\n"
            )

            f.write(
                "GENERATED:\n"
            )

            f.write(
                hypotheses_text[i]
                + "\n\n"
            )


    summary_rows.append([
        temperature,
        bleu,
        chrf,
        rouge,
        repetition,
        distinct1,
        distinct2,
    ])


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_path = (
    RESULTS_DIR
    / "assamese_standard_generation_metrics.csv"
)

with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "temperature",
        "BLEU-4",
        "chrF",
        "ROUGE-L",
        "repetition_rate",
        "Distinct-1",
        "Distinct-2",
    ])

    writer.writerows(
        summary_rows
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print(
    "assamese STANDARD GENERATION "
    "EVALUATION COMPLETE"
)
print("=" * 70)

print(
    f"Summary: {summary_path}"
)
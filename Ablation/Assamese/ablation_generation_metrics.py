from pathlib import Path
import os
import sys
import json
import math
import random
import re

import torch
import sentencepiece as spm


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = Path(
    os.environ.get(
        "ABLATION_CHECKPOINT",
        ROOT / "checkpoint_030518.pt"
    )
)

TOKENIZER_PATH = Path(
    os.environ.get(
        "ABLATION_TOKENIZER",
        ROOT / "assamese_10k.model"
    )
)

DATA_DIR = Path(
    os.environ.get(
        "ABLATION_DATA_DIR",
        ROOT.parent.parent / "Assamese" / "data" / "splits_v2"
    )
)

EVAL_DIR = Path(
    os.environ.get(
        "ABLATION_EVAL_DIR",
        ROOT / "evaluation_results"
    )
)

RESULTS_DIR = EVAL_DIR / "generation_metrics"
GENERATIONS_DIR = EVAL_DIR / "generation_metrics_data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.insert(0, str(ROOT))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

NUM_SAMPLES = 50

PROMPT_TOKENS = 128
REFERENCE_TOKENS = 100
MAX_NEW_TOKENS = 100

TEMPERATURES = [
    0.0,
    0.5,
    1.0,
    1.5
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(
    str(TOKENIZER_PATH)
):
    raise RuntimeError(
        f"Could not load tokenizer:\n{TOKENIZER_PATH}"
    )


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
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


print("=" * 80)
print("ASSAMESE GENERATION METRICS")
print("=" * 80)

print("Device:", device)
print("Checkpoint:", CHECKPOINT_PATH)
print("Checkpoint step:", checkpoint["step"])
print("Vocabulary:", sp.get_piece_size())
print("Samples:", NUM_SAMPLES)
print()


# ============================================================
# FIND VALIDATION FILES
# ============================================================

def get_text_files(directory):

    if directory.is_file():
        return [directory]

    if not directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n{directory}"
        )

    files = sorted(
        directory.rglob("*.txt")
    )

    if not files:
        raise FileNotFoundError(
            f"No validation .txt files found in:\n{directory}"
        )

    return files


# ============================================================
# LOAD VALIDATION EXAMPLES
# ============================================================

candidate_examples = []

validation_files = get_text_files(DATA_DIR)

for file_path in validation_files:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            ids = sp.encode(
                line,
                out_type=int
            )

            required = (
                PROMPT_TOKENS +
                REFERENCE_TOKENS
            )

            if len(ids) >= required:

                candidate_examples.append(
                    ids
                )

            if len(candidate_examples) >= NUM_SAMPLES * 5:
                break

    if len(candidate_examples) >= NUM_SAMPLES * 5:
        break


if len(candidate_examples) < NUM_SAMPLES:

    raise RuntimeError(
        f"Only found {len(candidate_examples)} "
        "suitable validation examples."
    )


random.shuffle(candidate_examples)

examples = candidate_examples[:NUM_SAMPLES]


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_continuation(
    prompt_ids,
    temperature
):

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(MAX_NEW_TOKENS):

        context = (
            generated[
                :,
                -config.context_length:
            ]
        )

        logits, _ = model(context)

        next_logits = logits[:, -1, :]

        if temperature == 0.0:

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True
            )

        else:

            scaled_logits = (
                next_logits / temperature
            )

            probabilities = torch.softmax(
                scaled_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )

    return generated[
        0,
        -MAX_NEW_TOKENS:
    ].tolist()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def word_tokens(text):

    return normalize_text(text).split()


def character_tokens(text):

    return list(
        normalize_text(text)
    )


# ============================================================
# N-GRAMS
# ============================================================

def ngrams(tokens, n):

    return [
        tuple(tokens[i:i+n])
        for i in range(
            len(tokens) - n + 1
        )
    ]


# ============================================================
# BLEU-4
# ============================================================

def modified_precision(
    reference,
    candidate,
    n
):

    ref_counts = {}

    for gram in ngrams(
        reference,
        n
    ):

        ref_counts[gram] = (
            ref_counts.get(gram, 0) + 1
        )

    cand_counts = {}

    for gram in ngrams(
        candidate,
        n
    ):

        cand_counts[gram] = (
            cand_counts.get(gram, 0) + 1
        )

    clipped = 0

    for gram, count in cand_counts.items():

        clipped += min(
            count,
            ref_counts.get(gram, 0)
        )

    total = max(
        len(candidate) - n + 1,
        0
    )

    return clipped, total


def bleu4(
    reference,
    candidate
):

    if not candidate:
        return 0.0

    precisions = []

    for n in range(1, 5):

        clipped, total = modified_precision(
            reference,
            candidate,
            n
        )

        precision = (
            clipped + 1
        ) / (
            total + 1
        )

        precisions.append(
            precision
        )

    log_precision = sum(
        math.log(p)
        for p in precisions
    ) / 4.0

    geometric_mean = math.exp(
        log_precision
    )

    brevity_penalty = 1.0

    if len(candidate) < len(reference):

        brevity_penalty = math.exp(
            1.0 -
            len(reference) /
            max(len(candidate), 1)
        )

    return (
        brevity_penalty *
        geometric_mean
    )


# ============================================================
# chrF
# ============================================================

def chrf(
    reference,
    candidate,
    max_order=6,
    beta=2
):

    scores = []

    for n in range(
        1,
        max_order + 1
    ):

        ref_ngrams = ngrams(
            character_tokens(reference),
            n
        )

        cand_ngrams = ngrams(
            character_tokens(candidate),
            n
        )

        ref_counts = {}
        cand_counts = {}

        for gram in ref_ngrams:
            ref_counts[gram] = (
                ref_counts.get(gram, 0) + 1
            )

        for gram in cand_ngrams:
            cand_counts[gram] = (
                cand_counts.get(gram, 0) + 1
            )

        overlap = sum(
            min(
                count,
                ref_counts.get(gram, 0)
            )
            for gram, count in cand_counts.items()
        )

        precision = (
            overlap /
            max(len(cand_ngrams), 1)
        )

        recall = (
            overlap /
            max(len(ref_ngrams), 1)
        )

        if precision + recall == 0:

            f_score = 0.0

        else:

            f_score = (
                (1 + beta ** 2)
                * precision
                * recall
                /
                (
                    beta ** 2 * precision
                    + recall
                )
            )

        scores.append(f_score)

    return sum(scores) / len(scores)


# ============================================================
# ROUGE-L
# ============================================================

def lcs_length(a, b):

    previous = [
        0
    ] * (
        len(b) + 1
    )

    for token_a in a:

        current = [0]

        for j, token_b in enumerate(
            b,
            start=1
        ):

            if token_a == token_b:

                current.append(
                    previous[j - 1] + 1
                )

            else:

                current.append(
                    max(
                        previous[j],
                        current[-1]
                    )
                )

        previous = current

    return previous[-1]


def rouge_l(
    reference,
    candidate
):

    ref = word_tokens(reference)
    cand = word_tokens(candidate)

    if not ref or not cand:
        return 0.0

    lcs = lcs_length(
        ref,
        cand
    )

    precision = (
        lcs / len(cand)
    )

    recall = (
        lcs / len(ref)
    )

    if precision + recall == 0:
        return 0.0

    beta = 1.2

    return (
        (1 + beta ** 2)
        * precision
        * recall
        /
        (
            recall
            + beta ** 2 * precision
        )
    )


# ============================================================
# DISTINCT-N
# ============================================================

def distinct_n(
    text,
    n
):

    tokens = word_tokens(text)

    if len(tokens) < n:
        return 0.0

    grams = ngrams(
        tokens,
        n
    )

    return (
        len(set(grams))
        /
        len(grams)
    )


# ============================================================
# REPETITION RATE
# ============================================================

def repetition_rate(text):

    tokens = word_tokens(text)

    if len(tokens) < 2:
        return 0.0

    grams = ngrams(
        tokens,
        2
    )

    seen = set()
    repeated = 0

    for gram in grams:

        if gram in seen:
            repeated += 1

        seen.add(gram)

    return (
        repeated /
        len(grams)
    )


# ============================================================
# RUN
# ============================================================

all_results = []


for temperature in TEMPERATURES:

    print(
        f"Generating/evaluating T={temperature}..."
    )

    temperature_results = []

    for index, ids in enumerate(examples):

        prompt_ids = ids[
            :PROMPT_TOKENS
        ]

        reference_ids = ids[
            PROMPT_TOKENS:
            PROMPT_TOKENS + REFERENCE_TOKENS
        ]

        generated_ids = generate_continuation(
            prompt_ids,
            temperature
        )

        reference_text = normalize_text(
            sp.decode(
                reference_ids
            )
        )

        generated_text = normalize_text(
            sp.decode(
                generated_ids
            )
        )

        result = {

            "sample": index,

            "temperature": temperature,

            "reference": reference_text,

            "generated": generated_text,

            "bleu4": bleu4(
                word_tokens(reference_text),
                word_tokens(generated_text)
            ),

            "chrf": chrf(
                reference_text,
                generated_text
            ),

            "rouge_l": rouge_l(
                reference_text,
                generated_text
            ),

            "repetition_rate": repetition_rate(
                generated_text
            ),

            "distinct_1": distinct_n(
                generated_text,
                1
            ),

            "distinct_2": distinct_n(
                generated_text,
                2
            )
        }

        temperature_results.append(
            result
        )

        all_results.append(
            result
        )

    output_file = (
        GENERATIONS_DIR /
        f"assamese_metrics_temp_{temperature}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            temperature_results,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# AGGREGATE
# ============================================================

summary = []


for temperature in TEMPERATURES:

    rows = [
        r for r in all_results
        if r["temperature"] == temperature
    ]

    summary.append({

        "temperature": temperature,

        "samples": len(rows),

        "BLEU-4": sum(
            r["bleu4"]
            for r in rows
        ) / len(rows),

        "chrF": sum(
            r["chrf"]
            for r in rows
        ) / len(rows),

        "ROUGE-L": sum(
            r["rouge_l"]
            for r in rows
        ) / len(rows),

        "Repetition Rate": sum(
            r["repetition_rate"]
            for r in rows
        ) / len(rows),

        "Distinct-1": sum(
            r["distinct_1"]
            for r in rows
        ) / len(rows),

        "Distinct-2": sum(
            r["distinct_2"]
            for r in rows
        ) / len(rows)
    })


# ============================================================
# SAVE JSON
# ============================================================

summary_file = (
    RESULTS_DIR /
    "assamese_generation_metrics.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    RESULTS_DIR /
    "assamese_generation_metrics.csv"
)

with open(
    csv_file,
    "w",
    encoding="utf-8"
) as f:

    headers = [
        "temperature",
        "samples",
        "BLEU-4",
        "chrF",
        "ROUGE-L",
        "Repetition Rate",
        "Distinct-1",
        "Distinct-2"
    ]

    f.write(
        ",".join(headers) + "\n"
    )

    for row in summary:

        f.write(
            f"{row['temperature']},"
            f"{row['samples']},"
            f"{row['BLEU-4']:.6f},"
            f"{row['chrF']:.6f},"
            f"{row['ROUGE-L']:.6f},"
            f"{row['Repetition Rate']:.6f},"
            f"{row['Distinct-1']:.6f},"
            f"{row['Distinct-2']:.6f}\n"
        )


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("ASSAMESE GENERATION METRICS")
print("=" * 80)

for row in summary:

    print(
        f"\nTemperature: {row['temperature']}"
    )

    print(
        f"  BLEU-4          : {row['BLEU-4']:.6f}"
    )

    print(
        f"  chrF            : {row['chrF']:.6f}"
    )

    print(
        f"  ROUGE-L         : {row['ROUGE-L']:.6f}"
    )

    print(
        f"  Repetition Rate : {row['Repetition Rate']:.6f}"
    )

    print(
        f"  Distinct-1      : {row['Distinct-1']:.6f}"
    )

    print(
        f"  Distinct-2      : {row['Distinct-2']:.6f}"
    )


print()
print("Saved CSV :", csv_file)
print("Saved JSON:", summary_file)
print("=" * 80) 
from pathlib import Path
import sys
import json
import re
from collections import defaultdict

import torch
import sentencepiece as spm


# ============================================================
# MAKE assamese/model IMPORTABLE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.transformer import DecoderOnlyTransformer


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_PATH = (
    PROJECT_DIR
    / "tokenizer_clean"
    / "assamese_10k.model"
)

# IMPORTANT:
# This is the ORIGINAL PRETRAINED checkpoint.
CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "checkpoint_030518_assamese.pt"
)

TEST_FILE = (
    PROJECT_DIR
    / "data_sft"
    / "test.jsonl"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "assamese_pretrained_phase3_predictions.jsonl"
)

MAX_NEW_TOKENS = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("assamese PHASE-3 PRETRAINED EVALUATION")
print("=" * 70)

print(f"Device: {device}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Test set: {TEST_FILE}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(str(TOKENIZER_PATH)):
    raise RuntimeError(
        f"Could not load tokenizer: {TOKENIZER_PATH}"
    )

print(f"Vocabulary size: {sp.get_piece_size()}")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

# Pretrained checkpoint contains its original config.
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
    f"Checkpoint step: "
    f"{checkpoint['step']}"
)

print(
    f"Parameters: "
    f"{model.count_parameters():,}"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}"
    )

test_rows = []

with open(
    TEST_FILE,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        test_rows.append(
            json.loads(line)
        )


print(
    f"Test examples: "
    f"{len(test_rows)}"
)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_answer(
    input_text,
    max_new_tokens=32,
):
    """
    Generate an answer from the pretrained assamese model
    using greedy decoding.
    """

    input_ids = sp.encode(
        input_text,
        out_type=int,
    )

    generated = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):

        # Respect the model context length.
        context = generated[
            :, -config.context_length:
        ]

        outputs = model(context)

        # DecoderOnlyTransformer returns:
        # logits, loss
        logits = outputs[0]

        next_logits = logits[:, -1, :]

        # Greedy decoding.
        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=1,
        )

        # Stop at EOS.
        eos_id = sp.eos_id()

        if (
            eos_id >= 0
            and next_token.item() == eos_id
        ):
            break

    # Only decode newly generated tokens.
    generated_tokens = generated[
        0,
        len(input_ids):,
    ].tolist()

    return sp.decode(
        generated_tokens
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Basic assamese text normalization.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.strip()

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_prediction(
    text,
    task=None,
):
    """
    Normalize model predictions and gold answers
    before exact-match comparison.

    Ordering answers are preserved as ordered lists.
    Other reasoning tasks use the normalized answer text.
    """

    text = normalize_text(text)

    # Remove common sentence-ending punctuation.
    text = text.strip(
        " \t\n\r.,!?;:।"
    )

    # --------------------------------------------------------
    # ORDERING
    # --------------------------------------------------------
    #
    # Do NOT reduce an ordering answer to its first item.
    # The complete ordered list must be preserved.
    #
    if task == "ordering":

        # Normalize spaces around commas.
        text = re.sub(
            r"\s*,\s*",
            ",",
            text,
        )

        return text

    # --------------------------------------------------------
    # OTHER TASKS
    # --------------------------------------------------------

    return text


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test_set(rows):

    predictions = []

    correct = 0

    for i, row in enumerate(rows):

        pred_raw = generate_answer(
            row["input"],
            max_new_tokens=MAX_NEW_TOKENS,
        )

        task = row.get(
            "task"
        )

        pred = normalize_prediction(
            pred_raw,
            task,
        )

        gold = normalize_prediction(
            row["answer"],
            task,
        )

        is_correct = (
            pred == gold
        )

        if is_correct:
            correct += 1

        predictions.append(
            {
                "id": row.get("id"),
                "input": row["input"],
                "gold": gold,
                "prediction_raw": pred_raw,
                "prediction": pred,
                "correct": is_correct,
                "task": task,
                "difficulty": row.get(
                    "difficulty"
                ),
                "domain": row.get(
                    "domain"
                ),
            }
        )

        if (i + 1) % 100 == 0:

            print(
                f"{i + 1}/{len(rows)}"
            )

    accuracy = (
        correct / len(rows)
    )

    return (
        predictions,
        correct,
        accuracy,
    )


# ============================================================
# RUN TEST
# ============================================================

predictions, correct, accuracy = (
    evaluate_test_set(
        test_rows
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

with open(
    PREDICTIONS_FILE,
    "w",
    encoding="utf-8",
) as f:

    for row in predictions:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# OVERALL ACCURACY
# ============================================================

print()
print("=" * 70)
print("assamese PHASE-3 PRETRAINED TEST")
print("=" * 70)

print(
    f"Correct : {correct}"
)

print(
    f"Total   : {len(test_rows)}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# GROUPED ACCURACY
# ============================================================

def grouped_accuracy(
    predictions,
    key,
):

    groups = defaultdict(
        lambda: [0, 0]
    )

    for row in predictions:

        value = row[key]

        groups[value][1] += 1

        if row["correct"]:

            groups[value][0] += 1

    return groups


# ============================================================
# TASK ACCURACY
# ============================================================

task_groups = grouped_accuracy(
    predictions,
    "task",
)

print()
print("TASK")
print("-" * 50)

for task, (c, n) in sorted(
    task_groups.items()
):

    print(
        f"{task:22s} "
        f"{c:3d}/{n:3d} "
        f"{100 * c / n:6.2f}%"
    )


# ============================================================
# DIFFICULTY ACCURACY
# ============================================================

difficulty_groups = (
    grouped_accuracy(
        predictions,
        "difficulty",
    )
)

print()
print("DIFFICULTY")
print("-" * 50)

for difficulty, (c, n) in sorted(
    difficulty_groups.items(),
    key=lambda x: str(x[0]),
):

    print(
        f"{str(difficulty):22s} "
        f"{c:3d}/{n:3d} "
        f"{100 * c / n:6.2f}%"
    )


# ============================================================
# DOMAIN ACCURACY
# ============================================================

domain_groups = grouped_accuracy(
    predictions,
    "domain",
)

print()
print("DOMAIN")
print("-" * 50)

for domain, (c, n) in sorted(
    domain_groups.items()
):

    print(
        f"{domain:22s} "
        f"{c:3d}/{n:3d} "
        f"{100 * c / n:6.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PRETRAINED EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Predictions: "
    f"{PREDICTIONS_FILE}"
)

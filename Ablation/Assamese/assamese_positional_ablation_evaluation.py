from pathlib import Path
import os
import sys
import json
import math
import time
import subprocess


import torch
import sentencepiece as spm


# ============================================================
# ACTUAL PROJECT STRUCTURE
# ============================================================

ROOT = Path(__file__).resolve().parent

PROJECT_ROOT = ROOT.parent.parent

DATA_DIR = Path(
    os.environ.get(
        "ABLATION_DATA_DIR",
        ROOT.parent.parent / "Assamese" / "data" / "splits_v2"
    )
)

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

EVAL_DIR = Path(
    os.environ.get(
        "ABLATION_EVAL_DIR",
        ROOT / "evaluation_results"
    )
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR = (
    EVAL_DIR / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.insert(
    0,
    str(ROOT)
)

from config import ModelConfig
from transformer import DecoderOnlyTransformer


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 80)
print("ASSAMESE POSITIONAL-EMBEDDING ABLATION")
print("PHASE 2 EVALUATION")
print("=" * 80)

print()
print("Root       :", ROOT)
print("Checkpoint :", CHECKPOINT_PATH)
print("Tokenizer  :", TOKENIZER_PATH)
print("Data       :", DATA_DIR)
print("Output     :", EVAL_DIR)
print()


if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


if not TOKENIZER_PATH.exists():

    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER_PATH}"
    )


if not DATA_DIR.exists():

    raise FileNotFoundError(
        f"Validation data not found:\n{DATA_DIR}"
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# LOAD CHECKPOINT
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


parameter_count = (
    model.count_parameters()
)


print(
    "Checkpoint step:",
    checkpoint["step"]
)

print(
    "Parameters:",
    parameter_count
)

print(
    "Vocabulary:",
    config.vocab_size
)

print()


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
# ARCHITECTURE CHECKS
# ============================================================

print("=" * 80)
print("ARCHITECTURE VERIFICATION")
print("=" * 80)


if config.vocab_size != 10_000:

    raise AssertionError(
        f"Expected vocab size 10000, "
        f"got {config.vocab_size}"
    )


if hasattr(
    model,
    "position_embedding"
):

    raise AssertionError(
        "Ablation model still contains "
        "'position_embedding'."
    )


expected_params = 25_350_912

if parameter_count != expected_params:

    raise AssertionError(
        f"Unexpected parameter count: "
        f"{parameter_count}. "
        f"Expected {expected_params}."
    )


if sp.get_piece_size() != config.vocab_size:

    raise AssertionError(
        "Tokenizer vocabulary does not "
        "match model vocabulary."
    )


print(
    "Vocabulary = 10,000: PASS"
)

print(
    "No positional embedding: PASS"
)

print(
    f"Parameter count = {expected_params:,}: PASS"
)

print(
    "Tokenizer/model vocabulary match: PASS"
)

print()


# ============================================================
# 1. VALIDATION CE + PPL
# ============================================================

print("=" * 80)
print("1. VALIDATION CROSS-ENTROPY + PERPLEXITY")
print("=" * 80)


SEQ_LEN = 512

MAX_VALIDATION_TOKENS = 100_000_000


def get_validation_files():

    if DATA_DIR.is_file():

        return [
            DATA_DIR
        ]

    files = sorted(
        DATA_DIR.rglob("*.txt")
    )

    if not files:

        raise FileNotFoundError(
            f"No .txt validation files found in:\n{DATA_DIR}"
        )

    return files


validation_files = (
    get_validation_files()
)



total_loss = 0.0
total_tokens = 0
num_sequences = 0

token_buffer = []

started = time.time()

stop = False


with torch.no_grad():

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

                token_buffer.extend(
                    sp.encode(
                        line,
                        out_type=int
                    )
                )

                while len(token_buffer) >= SEQ_LEN:

                    if (
                        total_tokens
                        >= MAX_VALIDATION_TOKENS
                    ):

                        stop = True
                        break

                    chunk = token_buffer[
                        :SEQ_LEN
                    ]

                    del token_buffer[
                        :SEQ_LEN
                    ]

                    x = torch.tensor(
                        [chunk[:-1]],
                        dtype=torch.long,
                        device=device
                    )

                    y = torch.tensor(
                        [chunk[1:]],
                        dtype=torch.long,
                        device=device
                    )

                    remaining = (
                        MAX_VALIDATION_TOKENS
                        - total_tokens
                    )

                    if y.numel() > remaining:

                        x = x[:, :remaining]

                        y = y[:, :remaining]

                    _, loss = model(
                        x,
                        y
                    )

                    n = y.numel()

                    total_loss += (
                        float(loss.item())
                        * n
                    )

                    total_tokens += n

                    num_sequences += 1

                    if (
                        total_tokens
                        >= MAX_VALIDATION_TOKENS
                    ):

                        stop = True
                        break

                if stop:
                    break

        if stop:
            break


if total_tokens == 0:

    raise RuntimeError(
        "No validation tokens were evaluated."
    )


validation_ce = (
    total_loss /
    total_tokens
)

validation_ppl = (
    math.exp(validation_ce)
)


validation_metrics = {

    "checkpoint":
        CHECKPOINT_PATH.name,

    "checkpoint_step":
        int(checkpoint["step"]),

    "validation_sequences":
        int(num_sequences),

    "validation_tokens":
        int(total_tokens),

    "cross_entropy":
        float(validation_ce),

    "perplexity":
        float(validation_ppl),

    "protocol_max_tokens":
        MAX_VALIDATION_TOKENS,

    "elapsed_minutes":
        (
            time.time() - started
        ) / 60.0
}


with open(
    EVAL_DIR /
    "validation_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        validation_metrics,
        f,
        indent=2
    )


print(
    "Validation tokens:",
    f"{total_tokens:,}"
)

print(
    "Cross-Entropy:",
    f"{validation_ce:.6f}"
)

print(
    "Perplexity:",
    f"{validation_ppl:.6f}"
)

print()


# ============================================================
# 2. BPB
# ============================================================

print("=" * 80)
print("2. BITS PER BYTE (BPB)")
print("=" * 80)


total_words = 0
total_sp_tokens = 0
total_bytes = 0


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

            total_words += (
                len(line.split())
            )

            total_sp_tokens += (
                len(
                    sp.encode(
                        line,
                        out_type=int
                    )
                )
            )

            total_bytes += (
                len(
                    line.encode("utf-8")
                )
            )


if (
    total_words == 0
    or total_bytes == 0
):

    raise RuntimeError(
        "Validation text is empty."
    )


tokens_per_word = (
    total_sp_tokens /
    total_words
)


bpb = (
    validation_ce *
    total_sp_tokens /
    (
        total_bytes *
        math.log(2)
    )
)


bpb_metrics = {

    "checkpoint":
        CHECKPOINT_PATH.name,

    "validation_words":
        int(total_words),

    "sentencepiece_tokens":
        int(total_sp_tokens),

    "utf8_bytes":
        int(total_bytes),

    "tokens_per_word":
        float(tokens_per_word),

    "cross_entropy_used":
        float(validation_ce),

    "perplexity":
        float(validation_ppl),

    "bits_per_byte":
        float(bpb)
}


with open(
    EVAL_DIR /
    "bpb_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        bpb_metrics,
        f,
        indent=2
    )


print(
    "Validation words     :",
    f"{total_words:,}"
)

print(
    "SentencePiece tokens :",
    f"{total_sp_tokens:,}"
)

print(
    "UTF-8 bytes          :",
    f"{total_bytes:,}"
)

print(
    "Tokens / word        :",
    f"{tokens_per_word:.6f}"
)

print(
    "BPB                  :",
    f"{bpb:.6f}"
)

print()


# ============================================================
# RUN SUB-SCRIPTS
# ============================================================

def run_script(
    script_name,
    log_name
):

    script_path = (
        ROOT / script_name
    )

    log_path = (
        LOG_DIR / log_name
    )

    if not script_path.exists():

        raise FileNotFoundError(
            f"Evaluation script not found:\n"
            f"{script_path}"
        )


    env = os.environ.copy()

    env[
        "ABLATION_CHECKPOINT"
    ] = str(
        CHECKPOINT_PATH
    )

    env[
        "ABLATION_TOKENIZER"
    ] = str(
        TOKENIZER_PATH
    )

    env[
        "ABLATION_DATA_DIR"
    ] = str(
        DATA_DIR
    )

    env[
        "ABLATION_EVAL_DIR"
    ] = str(
        EVAL_DIR
    )


    print("=" * 80)
    print(
        f"RUNNING {script_name}"
    )
    print("=" * 80)


    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as log:

        result = subprocess.run(

            [
                sys.executable,
                "-u",
                str(script_path)
            ],

            env=env,

            stdout=log,

            stderr=subprocess.STDOUT,

            text=True
        )


    if result.returncode != 0:

        print()

        print(
            log_path.read_text(
                encoding="utf-8"
            )[-10000:]
        )

        raise RuntimeError(
            f"{script_name} failed "
            f"with exit code "
            f"{result.returncode}"
        )


    print(
        f"Completed: {script_name}"
    )

    print(
        f"Log: {log_path}"
    )

    print()


# ============================================================
# 3. STANDARD GENERATION
# ============================================================

run_script(
    "ablation_standard_generation.py",
    "standard_generation.log"
)


# ============================================================
# 4. GENERATION METRICS
# ============================================================

run_script(
    "ablation_generation_metrics.py",
    "generation_metrics.log"
)


# ============================================================
# 5. ATTENTION ANALYSIS
# ============================================================

run_script(
    "ablation_attention_aggregate.py",
    "attention.log"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

generation_summary = (
    EVAL_DIR /
    "generation_metrics" /
    "assamese_generation_metrics.json"
)


summary = {

    "experiment":
        "Assamese positional-embedding ablation",

    "checkpoint":
        CHECKPOINT_PATH.name,

    "checkpoint_step":
        int(checkpoint["step"]),

    "parameters":
        int(parameter_count),

    "validation":
        validation_metrics,

    "bpb":
        bpb_metrics,

    "generation_metrics_file":
        str(generation_summary),

    "standard_generation_dir":
        str(
            EVAL_DIR /
            "generations"
        ),

    "attention_csv":
        str(
            EVAL_DIR /
            "attention" /
            "assamese_attention_aggregate.csv"
        ),

    "attention_figures_dir":
        str(
            EVAL_DIR /
            "attention_figures"
        ),

    "completed_at_epoch_seconds":
        time.time()
}


with open(
    EVAL_DIR /
    "summary.json",
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
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("PHASE 2 ASSAMESE ABLATION EVALUATION COMPLETE")
print("=" * 80)

print(
    "Validation CE :",
    f"{validation_ce:.6f}"
)

print(
    "Validation PPL:",
    f"{validation_ppl:.6f}"
)

print(
    "BPB           :",
    f"{bpb:.6f}"
)

print(
    "Evaluation dir:",
    EVAL_DIR
)

print()
print("All evaluation stages completed successfully.")
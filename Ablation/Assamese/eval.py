from pathlib import Path
import os
import sys
import subprocess
import json
import shutil


# ============================================================
# LOCAL ASSAMESE POSITIONAL ABLATION EVALUATION
# ============================================================
#
# Runs ONLY the remaining Phase-2 ablation evaluations:
#
#   1. Standard Generation
#   2. Generation Metrics
#   3. Attention Aggregate
#
# Validation + BPB were already completed on Kaggle and are
# intentionally skipped.
#
# ============================================================


# ------------------------------------------------------------
# ROOT DIRECTORIES
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

EVAL_DIR = ROOT / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS_DIR = EVAL_DIR / "scripts"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = EVAL_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GEN_DIR = EVAL_DIR / "generations"
GEN_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# IMPORTANT:
# Your Kaggle checkpoint was downloaded/extracted as:
#
#     .checkpoint_030518/
#
# We DO NOT attempt to reconstruct it.
#
# The original evaluation script expects:
#
#     checkpoints/checkpoint_030518.pt
#
# However, the extracted checkpoint is a PyTorch serialization
# archive represented as a directory by Kaggle.
#
# We therefore first look for an actual .pt checkpoint.
# ------------------------------------------------------------

def find_checkpoint():

    candidates = [

        ROOT / "checkpoint_030518.pt",

        ROOT / "checkpoints" / "checkpoint_030518.pt",

        ROOT / ".checkpoint_030518.pt",

        ROOT / ".checkpoint_030518" / "checkpoint_030518.pt",

        ROOT / "checkpoint_030518" / "checkpoint_030518.pt",
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # Recursive search for any actual checkpoint
    found = list(ROOT.rglob("checkpoint_*.pt"))

    if found:
        found.sort(
            key=lambda p: (
                int(p.stem.split("_")[-1])
                if p.stem.split("_")[-1].isdigit()
                else -1
            )
        )

        return found[-1]

    return None


CHECKPOINT_PATH = find_checkpoint()


# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------

def find_tokenizer():

    candidates = [

        ROOT / "assamese_10k.model",

        ROOT / "tokenizer.model",

        ROOT / "tokenizer" / "tokenizer.model",

        ROOT / "../../Assamese/tokenizer_clean/tokenizer.model",

        ROOT / "../../Assamese/tokenizer/tokenizer.model",

        ROOT / "../../Assamese" / "assamese_10k.model",
    ]

    for p in candidates:
        if p.exists():
            return p

    # Search repo
    for pattern in [
        "*.model",
        "*.model.model",
    ]:

        found = list(ROOT.parent.parent.rglob(pattern))

        # Prefer files containing Assamese
        assamese = [
            p for p in found
            if "assam" in str(p).lower()
        ]

        if assamese:
            return assamese[0]

        if found:
            return found[0]

    return None


TOKENIZER_PATH = find_tokenizer()


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def find_data():

    candidates = [

        ROOT / "data",

        ROOT / "data_ablation",

        ROOT / "../../Assamese/data",

        ROOT / "../../Assamese/data_sft",

        ROOT / "../../Assamese/data/validation",

    ]

    for p in candidates:

        if p.exists() and p.is_dir():
            return p

    return None


DATA_DIR = find_data()


# ------------------------------------------------------------
# MODEL FILES
# ------------------------------------------------------------

def find_model_files():

    candidates = [

        ROOT,
        ROOT.parent,
        ROOT.parent.parent,
        ROOT.parent.parent / "Assamese",
        ROOT.parent.parent / "Gujarati",
    ]

    config_file = None
    transformer_file = None

    for base in candidates:

        c = base / "config.py"
        t = base / "transformer.py"

        if c.exists() and config_file is None:
            config_file = c

        if t.exists() and transformer_file is None:
            transformer_file = t

    return config_file, transformer_file


CONFIG_FILE, TRANSFORMER_FILE = find_model_files()


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

print("=" * 80)
print("ASSAMESE POSITIONAL-EMBEDDING ABLATION")
print("REMAINING PHASE-2 EVALUATION")
print("=" * 80)

print()
print("ROOT       :", ROOT)
print("Evaluation :", EVAL_DIR)
print("Checkpoint :", CHECKPOINT_PATH)
print("Tokenizer  :", TOKENIZER_PATH)
print("Data       :", DATA_DIR)
print("config.py  :", CONFIG_FILE)
print("transformer:", TRANSFORMER_FILE)
print()


if CHECKPOINT_PATH is None:

    print("ERROR: No actual .pt checkpoint found.")
    print()
    print("The downloaded .checkpoint_030518 directory cannot")
    print("simply be renamed to .pt.")
    print()
    print("Please make sure the actual checkpoint file is present.")
    print()
    sys.exit(1)


if TOKENIZER_PATH is None:

    print("ERROR: SentencePiece tokenizer not found.")
    print()
    print("Expected an Assamese *.model file.")
    sys.exit(1)


if DATA_DIR is None:

    print("ERROR: Assamese evaluation data directory not found.")
    print()
    sys.exit(1)


if CONFIG_FILE is None:

    print("ERROR: config.py not found.")
    sys.exit(1)


if TRANSFORMER_FILE is None:

    print("ERROR: transformer.py not found.")
    sys.exit(1)


# ============================================================
# COPY MODEL FILES INTO A TEMP LOCAL IMPORT LOCATION
# ============================================================

# The original Kaggle evaluator does:
#
#     sys.path.insert(0, "/kaggle/working")
#     from config import ModelConfig
#     from transformer import DecoderOnlyTransformer
#
# We reproduce that locally.

LOCAL_MODEL_DIR = ROOT / "_local_model"

LOCAL_MODEL_DIR.mkdir(exist_ok=True)

local_config = LOCAL_MODEL_DIR / "config.py"
local_transformer = LOCAL_MODEL_DIR / "transformer.py"

if CONFIG_FILE.resolve() != local_config.resolve():

    shutil.copy2(
        CONFIG_FILE,
        local_config
    )

if TRANSFORMER_FILE.resolve() != local_transformer.resolve():

    shutil.copy2(
        TRANSFORMER_FILE,
        local_transformer
    )


# ============================================================
# VALIDATE CHECKPOINT BEFORE RUNNING EXPENSIVE EVALUATIONS
# ============================================================

print("=" * 80)
print("CHECKPOINT TEST")
print("=" * 80)

test_code = r'''
import torch
import sys

checkpoint_path = sys.argv[1]

print("Loading:", checkpoint_path)

x = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint type:", type(x))

if isinstance(x, dict):

    print("Checkpoint keys:")
    for k in x.keys():
        print("  ", k)

    if "config" in x:
        print("Config found: PASS")

    if "model_state_dict" in x:
        print("Model state dict found: PASS")

    if "step" in x:
        print("Training step:", x["step"])

else:

    print("WARNING: checkpoint is not a dictionary")
'''

result = subprocess.run(
    [
        sys.executable,
        "-c",
        test_code,
        str(CHECKPOINT_PATH)
    ],
    text=True
)

if result.returncode != 0:

    print()
    print("=" * 80)
    print("CHECKPOINT TEST FAILED")
    print("=" * 80)
    print()
    print("The file found is not a directly loadable PyTorch checkpoint.")
    print()
    print("Do NOT continue with evaluation.")
    sys.exit(1)


# ============================================================
# PREPARE ENVIRONMENT
# ============================================================

env = os.environ.copy()

env["ABLATION_CHECKPOINT"] = str(
    CHECKPOINT_PATH.resolve()
)

env["ABLATION_EVAL_DIR"] = str(
    EVAL_DIR.resolve()
)

env["ABLATION_DATA_DIR"] = str(
    DATA_DIR.resolve()
)

env["ABLATION_TOKENIZER_PATH"] = str(
    TOKENIZER_PATH.resolve()
)

# Some generated scripts may use these names.
env["CHECKPOINT_PATH"] = str(
    CHECKPOINT_PATH.resolve()
)

env["TOKENIZER_PATH"] = str(
    TOKENIZER_PATH.resolve()
)

env["DATA_DIR"] = str(
    DATA_DIR.resolve()
)

env["EVAL_DIR"] = str(
    EVAL_DIR.resolve()
)


# ============================================================
# FIND EVALUATION SCRIPTS
# ============================================================

def locate_script(filename):

    candidates = [

        SCRIPTS_DIR / filename,

        ROOT / filename,

        ROOT / "evaluation" / filename,

        ROOT / "evaluation" / "scripts" / filename,
    ]

    for p in candidates:

        if p.exists():
            return p

    found = list(
        ROOT.rglob(filename)
    )

    if found:
        return found[0]

    return None


STANDARD_SCRIPT = locate_script(
    "ablation_standard_generation.py"
)

GEN_METRICS_SCRIPT = locate_script(
    "ablation_generation_metrics.py"
)

ATTENTION_SCRIPT = locate_script(
    "ablation_attention_aggregate.py"
)


print()
print("=" * 80)
print("EVALUATION SCRIPTS")
print("=" * 80)

print("Standard generation :", STANDARD_SCRIPT)
print("Generation metrics  :", GEN_METRICS_SCRIPT)
print("Attention aggregate :", ATTENTION_SCRIPT)
print()


if STANDARD_SCRIPT is None:
    print("ERROR: ablation_standard_generation.py not found.")
    sys.exit(1)

if GEN_METRICS_SCRIPT is None:
    print("ERROR: ablation_generation_metrics.py not found.")
    sys.exit(1)

if ATTENTION_SCRIPT is None:
    print("ERROR: ablation_attention_aggregate.py not found.")
    sys.exit(1)


# ============================================================
# SCRIPT RUNNER
# ============================================================

def run_script(
    script_path,
    log_name,
    title
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Script:", script_path)
    print()

    log_path = LOG_DIR / log_name

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as log:

        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(script_path)
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:

            print(line, end="")

            log.write(line)
            log.flush()

        return_code = process.wait()

    print()
    print("Return code:", return_code)
    print("Log:", log_path)

    if return_code != 0:

        print()
        print("=" * 80)
        print(title, "FAILED")
        print("=" * 80)
        print()
        print(
            log_path.read_text(
                encoding="utf-8"
            )[-10000:]
        )

        raise RuntimeError(
            f"{script_path.name} failed."
        )

    print()
    print(title, "COMPLETED")


# ============================================================
# 1. STANDARD GENERATION
# ============================================================

run_script(
    STANDARD_SCRIPT,
    "standard_generation.log",
    "1. STANDARD GENERATION"
)


# ============================================================
# 2. GENERATION METRICS
# ============================================================

run_script(
    GEN_METRICS_SCRIPT,
    "generation_metrics.log",
    "2. GENERATION METRICS"
)


# ============================================================
# 3. ATTENTION
# ============================================================

run_script(
    ATTENTION_SCRIPT,
    "attention.log",
    "3. ATTENTION AGGREGATE"
)


# ============================================================
# COLLECT RESULTS
# ============================================================

print()
print("=" * 80)
print("COLLECTING RESULTS")
print("=" * 80)


result_files = []

for p in EVAL_DIR.rglob("*"):

    if p.is_file():

        result_files.append(
            str(
                p.relative_to(ROOT)
            )
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "experiment":
        "Assamese positional-embedding ablation",

    "checkpoint":
        str(CHECKPOINT_PATH),

    "tokenizer":
        str(TOKENIZER_PATH),

    "data":
        str(DATA_DIR),

    "validation":
        "already completed on Kaggle",

    "bpb":
        "already completed on Kaggle",

    "remaining_evaluations": [

        "standard_generation",
        "generation_metrics",
        "attention_aggregate"
    ],

    "result_files":
        result_files
}


summary_path = (
    EVAL_DIR /
    "remaining_evaluation_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 80)
print("ALL REMAINING ABLATION EVALUATIONS COMPLETED")
print("=" * 80)

print()
print("Results directory:")
print(EVAL_DIR)

print()
print("Summary:")
print(summary_path)

print()
print("Validation/BPB were intentionally skipped because they")
print("were already completed in the Kaggle run.")

print()
print("=" * 80)
from pathlib import Path
import os
import sys
import json
import math
import time
import subprocess

import torch
import sentencepiece as spm

BASE = Path("/kaggle/working/assamese_10k")
CHECKPOINT_DIR = BASE / "checkpoints"
TOKENIZER_PATH = Path("/kaggle/working/assamese_10k.model")
DATA_DIR = Path(os.environ.get("ABLATION_DATA_DIR", "/kaggle/input/datasets/vidurshah04/assam-data"))
EVAL_DIR = BASE / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = EVAL_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = EVAL_DIR / "scripts"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Locate latest checkpoint numerically, not lexicographically.
# ------------------------------------------------------------
checkpoints = sorted(
    CHECKPOINT_DIR.glob("checkpoint_*.pt"),
    key=lambda p: int(p.stem.split("_")[1]),
)
if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")

CHECKPOINT_PATH = checkpoints[-1]

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(TOKENIZER_PATH)
if not DATA_DIR.exists():
    raise FileNotFoundError(DATA_DIR)

print("=" * 80)
print("ASSAMESE POSITIONAL-EMBEDDING ABLATION — PHASE 2 EVALUATION")
print("=" * 80)
print("Checkpoint :", CHECKPOINT_PATH)
print("Tokenizer  :", TOKENIZER_PATH)
print("Data       :", DATA_DIR)
print()

# ------------------------------------------------------------
# Load model once for checkpoint/validation/BPB verification.
# ------------------------------------------------------------
sys.path.insert(0, "/kaggle/working")
from config import ModelConfig
from transformer import DecoderOnlyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
config = ModelConfig(**checkpoint["config"])
model = DecoderOnlyTransformer(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

parameter_count = model.count_parameters()
print("Checkpoint step:", checkpoint["step"])
print("Parameters    :", parameter_count)
print("Vocabulary    :", config.vocab_size)

# Strict architecture checks for the ablation.
assert config.vocab_size == 10_000
assert not hasattr(model, "position_embedding"), "Positional embedding attribute exists."
expected_params = 25_350_912
assert parameter_count == expected_params, (
    f"Unexpected ablation parameter count: {parameter_count}; "
    f"expected {expected_params} for this architecture."
)

sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
assert sp.get_piece_size() == config.vocab_size
print("Tokenizer/model vocabulary match: PASS")
print("No positional embedding attribute: PASS")
print("Ablation parameter count: PASS")
print()

# ------------------------------------------------------------
# 1. Validation CE + PPL
# Matches the supplied Phase-2 evaluator protocol:
# 512-token chunks, at most 100M target tokens.
# ------------------------------------------------------------
print("=" * 80)
print("1. VALIDATION CROSS-ENTROPY + PERPLEXITY")
print("=" * 80)

SEQ_LEN = 512
MAX_VALIDATION_TOKENS = 100_000_000

total_loss = 0.0
total_tokens = 0
num_sequences = 0
token_buffer = []
started = time.time()

stop = False
with torch.no_grad():
    for file_path in sorted(DATA_DIR.glob("val_*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_buffer.extend(sp.encode(line, out_type=int))

                while len(token_buffer) >= SEQ_LEN:
                    if total_tokens >= MAX_VALIDATION_TOKENS:
                        stop = True
                        break

                    chunk = token_buffer[:SEQ_LEN]
                    del token_buffer[:SEQ_LEN]

                    x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
                    y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

                    remaining = MAX_VALIDATION_TOKENS - total_tokens
                    if y.numel() > remaining:
                        x = x[:, :remaining]
                        y = y[:, :remaining]

                    _, loss = model(x, y)
                    n = y.numel()
                    total_loss += float(loss.item()) * n
                    total_tokens += n
                    num_sequences += 1

                    if total_tokens >= MAX_VALIDATION_TOKENS:
                        stop = True
                        break

                if stop:
                    break
        if stop:
            break

if total_tokens == 0:
    raise RuntimeError("No validation tokens were evaluated.")

validation_ce = total_loss / total_tokens
validation_ppl = math.exp(validation_ce)

validation_metrics = {
    "checkpoint": CHECKPOINT_PATH.name,
    "checkpoint_step": int(checkpoint["step"]),
    "validation_sequences": int(num_sequences),
    "validation_tokens": int(total_tokens),
    "cross_entropy": float(validation_ce),
    "perplexity": float(validation_ppl),
    "protocol_max_tokens": MAX_VALIDATION_TOKENS,
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(EVAL_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(validation_metrics, f, indent=2)

print("Validation tokens:", f"{total_tokens:,}")
print("Cross-Entropy    :", f"{validation_ce:.6f}")
print("Perplexity       :", f"{validation_ppl:.6f}")
print()

# ------------------------------------------------------------
# 2. BPB
# Matches the supplied BPB formula and uses the measured
# validation CE above.
# ------------------------------------------------------------
print("=" * 80)
print("2. BITS PER BYTE (BPB)")
print("=" * 80)

total_words = 0
total_sp_tokens = 0
total_bytes = 0

for file_path in sorted(DATA_DIR.glob("val_*.txt")):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_words += len(line.split())
            total_sp_tokens += len(sp.encode(line, out_type=int))
            total_bytes += len(line.encode("utf-8"))

if total_words == 0 or total_bytes == 0:
    raise RuntimeError("Validation text is empty.")

tokens_per_word = total_sp_tokens / total_words
bpb = validation_ce * total_sp_tokens / (total_bytes * math.log(2))

bpb_metrics = {
    "checkpoint": CHECKPOINT_PATH.name,
    "validation_words": int(total_words),
    "sentencepiece_tokens": int(total_sp_tokens),
    "utf8_bytes": int(total_bytes),
    "tokens_per_word": float(tokens_per_word),
    "cross_entropy_used": float(validation_ce),
    "perplexity": float(validation_ppl),
    "bits_per_byte": float(bpb),
}
with open(EVAL_DIR / "bpb_metrics.json", "w", encoding="utf-8") as f:
    json.dump(bpb_metrics, f, indent=2)

print("Validation words     :", f"{total_words:,}")
print("SentencePiece tokens :", f"{total_sp_tokens:,}")
print("UTF-8 bytes          :", f"{total_bytes:,}")
print("Tokens / word        :", f"{tokens_per_word:.6f}")
print("BPB                  :", f"{bpb:.6f}")
print()

# Release the validation model before subprocess-based generation/attention.
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# Helper to run a generated evaluation script and save stdout.
# ------------------------------------------------------------
def run_script(script_name, log_name):
    script_path = SCRIPTS_DIR / script_name
    log_path = LOG_DIR / log_name

    env = os.environ.copy()
    env["ABLATION_CHECKPOINT"] = str(CHECKPOINT_PATH)
    env["ABLATION_EVAL_DIR"] = str(EVAL_DIR)
    env["ABLATION_DATA_DIR"] = str(DATA_DIR)

    print("=" * 80)
    print(f"RUNNING {script_name}")
    print("=" * 80)

    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        print(log_path.read_text(encoding="utf-8")[-10000:])
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    print(f"Completed: {script_name}")
    print(f"Log: {log_path}")

# ------------------------------------------------------------
# 3A. Standard qualitative generation
# ------------------------------------------------------------
run_script(
    "ablation_standard_generation.py",
    "standard_generation.log",
)

# ------------------------------------------------------------
# 3B. Quantitative generation metrics
# BLEU-4, chrF, ROUGE-L, repetition rate,
# Distinct-1, Distinct-2 across T=0,0.5,1,1.5.
# ------------------------------------------------------------
run_script(
    "ablation_generation_metrics.py",
    "generation_metrics.log",
)

# ------------------------------------------------------------
# 4. Aggregated attention
# 10 validation sequences, all layers x heads, entropy and
# mean attention distance with CSV + heatmaps.
# ------------------------------------------------------------
run_script(
    "ablation_attention_aggregate.py",
    "attention.log",
)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
generation_summary = EVAL_DIR / "generation_metrics" / "assamese_generation_metrics.json"

summary = {
    "experiment": "Assamese positional-embedding ablation",
    "checkpoint": CHECKPOINT_PATH.name,
    "checkpoint_step": int(checkpoint["step"]),
    "parameters": int(parameter_count),
    "validation": validation_metrics,
    "bpb": bpb_metrics,
    "generation_metrics_file": str(generation_summary),
    "standard_generation_dir": str(EVAL_DIR / "generations"),
    "attention_csv": str(EVAL_DIR / "attention" / "assamese_attention_aggregate.csv"),
    "attention_figures_dir": str(EVAL_DIR / "attention_figures"),
    "completed_at_epoch_seconds": time.time(),
}
with open(EVAL_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 80)
print("PHASE 2 ABLATION EVALUATION COMPLETE")
print("=" * 80)
print("Validation CE :", f"{validation_ce:.6f}")
print("Validation PPL:", f"{validation_ppl:.6f}")
print("BPB           :", f"{bpb:.6f}")
print("Evaluation dir:", EVAL_DIR)

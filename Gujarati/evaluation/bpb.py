from pathlib import Path
import math
import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "splits_v2"
TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "tokenizer_clean"
    / "gujarati_24k.model"
)

SPLIT = "val"

# Final validation CE from evaluate_lm.py
VALIDATION_CE = 5.685608


# ============================================================
# LOAD TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(str(TOKENIZER_PATH)):
    raise RuntimeError(f"Could not load tokenizer: {TOKENIZER_PATH}")

print("=" * 70)
print("GUJARATI BPB EVALUATION")
print("=" * 70)

print(f"Tokenizer: {TOKENIZER_PATH}")
print(f"Vocabulary size: {sp.get_piece_size()}")
print(f"Data directory: {DATA_DIR}")


# ============================================================
# PROCESS VALIDATION FILES
# ============================================================

files = sorted(DATA_DIR.glob(f"{SPLIT}_*.txt"))

if not files:
    raise FileNotFoundError(
        f"No validation files found in {DATA_DIR}"
    )

total_words = 0
total_tokens = 0
total_bytes = 0

for file_path in files:

    print(f"Processing: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            # Word count
            words = line.split()
            total_words += len(words)

            # UTF-8 byte count
            total_bytes += len(line.encode("utf-8"))

            # SentencePiece token count
            ids = sp.encode(line, out_type=int)
            total_tokens += len(ids)


# ============================================================
# CALCULATE METRICS
# ============================================================

tokens_per_word = total_tokens / total_words

# Cross entropy is measured in natural logarithms (nats/token).
# Convert total negative log-likelihood to bits and normalize
# by UTF-8 bytes.
bpb = (
    VALIDATION_CE * total_tokens
    / (total_bytes * math.log(2))
)

ppl = math.exp(VALIDATION_CE)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("GUJARATI BPB RESULTS")
print("=" * 70)

print(f"Validation words       : {total_words:,}")
print(f"SentencePiece tokens   : {total_tokens:,}")
print(f"UTF-8 bytes            : {total_bytes:,}")
print(f"Tokens / word          : {tokens_per_word:.6f}")
print(f"Cross-Entropy          : {VALIDATION_CE:.6f}")
print(f"Perplexity             : {ppl:.4f}")
print(f"Bits Per Byte (BPB)    : {bpb:.6f}")

print("=" * 70)
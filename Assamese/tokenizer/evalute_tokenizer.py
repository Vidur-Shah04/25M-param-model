import glob
import sentencepiece as spm
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

LANG = "gujarati"

TEST_FILES = glob.glob(
    r"Gujarati\data\splits_v2\test_*.txt"
)

TOKENIZER = Path(
    r"Gujarati\tokenizer\gujarati.model"
)


# ============================================================
# CALCULATE TOKENS / WORD
# ============================================================

def evaluate():

    print("=" * 60)
    print(f"{LANG.upper()} TOKENIZER TEST")
    print("=" * 60)

    print(f"Test files : {len(TEST_FILES)}")
    print(f"Tokenizer  : {TOKENIZER.resolve()}")

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER)
    )

    words = 0
    tokens = 0

    for i, path in enumerate(TEST_FILES, 1):

        print(f"Reading {i}/{len(TEST_FILES)}: {Path(path).name}")

        with open(path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                words += len(line.split())

                tokens += len(
                    tokenizer.encode(
                        line,
                        out_type=int
                    )
                )

    ratio = tokens / words

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Words  : {words:,}")
    print(f"Tokens : {tokens:,}")
    print(f"Tokens/word : {ratio:.4f}")

    print()
    print(
        f"Estimated tokens for 500M words: "
        f"{500_000_000 * ratio:,.0f}"
    )


if __name__ == "__main__":
    evaluate()
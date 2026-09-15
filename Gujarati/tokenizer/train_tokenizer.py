import sentencepiece as spm
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

LANG = "Gujarati"

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    ROOT
    / "data"
    / "splits_v2"
)

OUTPUT_DIR = (
    ROOT
    / "tokenizer"
)

MODEL_PREFIX = (
    OUTPUT_DIR
    / LANG.lower()
)

VOCAB_SIZE = 10000
MODEL_TYPE = "bpe"
CHAR_COVERAGE = 1.0


# ============================================================
# TRAIN
# ============================================================

def train_tokenizer():

    print("=" * 60)
    print(f"Training {LANG} tokenizer")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_files = sorted(
        TRAIN_DIR.glob(
            "train_*.txt"
        )
    )

    if not train_files:

        raise FileNotFoundError(
            f"No training files found in:\n"
            f"{TRAIN_DIR}"
        )

    print(
        f"Training files: "
        f"{len(train_files):,}"
    )

    print(
        f"Vocabulary size: "
        f"{VOCAB_SIZE:,}"
    )

    print(
        f"Model type: "
        f"{MODEL_TYPE}"
    )

    print(
        f"Output: "
        f"{MODEL_PREFIX}"
    )

    # SentencePiece accepts multiple input files
    # as a comma-separated list.
    input_files = ",".join(
        str(path)
        for path in train_files
    )

    # spm.SentencePieceTrainer.train(

    #     input=input_files,

    #     model_prefix=str(
    #         MODEL_PREFIX
    #     ),

    #     vocab_size=VOCAB_SIZE,

    #     model_type=MODEL_TYPE,

    #     character_coverage=CHAR_COVERAGE,

    #     unk_id=0,
    #     bos_id=1,
    #     eos_id=2,
    #     pad_id=3,

    #     user_defined_symbols=[],

    #     # Useful for a large corpus
    #     shuffle_input_sentence=True,

    #     # Reproducibility
    #     # random_seed=42,
    # )

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHAR_COVERAGE,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=[],
        shuffle_input_sentence=True,
    )
    print(
        "\nTokenizer training complete."
    )

    print(
        f"Model: "
        f"{MODEL_PREFIX}.model"
    )

    print(
        f"Vocabulary: "
        f"{MODEL_PREFIX}.vocab"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_tokenizer()
# from pathlib import Path
# import sentencepiece as spm

# ROOT = Path(__file__).resolve().parent

# CONFIG = {
#     "Gujarati": {
#         "train": ROOT / "Gujarati/data/splits_v2/train_0000.txt" ,
#         "test": ROOT / "Gujarati/data/splits_v2",
#         "out": ROOT / "Gujarati/tokenizer"
#     },
#     "Assamese": {
#         "train": ROOT / "Assamese/data/splits_v2/train_0000.txt",
#         "test": ROOT / "Assamese/data/splits_v2",
#         "out": ROOT / "Assamese/tokenizer"
#     }
# }

# VOCAB_SIZES = [16000, 24000, 32000]


# def train_tokenizer(language, cfg, vocab_size):

#     model_prefix = cfg["out"] / f"{language.lower()}_{vocab_size // 1000}k"

#     print(f"\nTraining {language} {vocab_size:,}...")

#     spm.SentencePieceTrainer.train(
#         input=str(cfg["train"]),
#         model_prefix=str(model_prefix),
#         vocab_size=vocab_size,
#         model_type="bpe",
#         character_coverage=1.0,
#         unk_id=0,
#         bos_id=1,
#         eos_id=2,
#         pad_id=3,
#     )

#     return model_prefix.with_suffix(".model")


# def evaluate(model, test_dir):

#     tokenizer = spm.SentencePieceProcessor(
#         model_file=str(model)
#     )

#     words = 0
#     tokens = 0

#     files = sorted(test_dir.glob("test_*.txt"))

#     for file in files:

#         print(f"  Testing {file.name}")

#         with open(file, encoding="utf-8") as f:
#             for line in f:

#                 line = line.strip()

#                 if not line:
#                     continue

#                 words += len(line.split())
#                 tokens += len(
#                     tokenizer.encode(
#                         line,
#                         out_type=str
#                     )
#                 )

#     return tokens / words


# def main():

#     for language, cfg in CONFIG.items():

#         print("\n" + "=" * 60)
#         print(language.upper())
#         print("=" * 60)

#         cfg["out"].mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         for vocab_size in VOCAB_SIZES:

#             model = train_tokenizer(
#                 language,
#                 cfg,
#                 vocab_size
#             )

#             fertility = evaluate(
#                 model,
#                 cfg["test"]
#             )

#             print(
#                 f"{vocab_size // 1000}K → "
#                 f"fertility = {fertility:.4f}"
#             )


# if __name__ == "__main__":
#     main()
from pathlib import Path
import re
import sentencepiece as spm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "Gujarati": {
        "train_dir": ROOT / "Gujarati/data/splits_v2",
        "out": ROOT / "Gujarati/tokenizer_clean",
        "unicode_range": r"\u0A80-\u0AFF",
    },

    "Assamese": {
        "train_dir": ROOT / "Assamese/data/splits_v2",
        "out": ROOT / "Assamese/tokenizer_clean",
        "unicode_range": r"\u0980-\u09FF",
    },
}

VOCAB_SIZES = [10000]


# ============================================================
# UNICODE FILTER
# ============================================================

def clean_line(line, unicode_range):
    """
    Keep:
      - Target-language Unicode characters
      - whitespace
      - ASCII digits
      - common punctuation

    Remove:
      - Chinese/Japanese/Korean characters
      - emojis
      - Arabic/Latin/etc. letters
      - other unwanted Unicode scripts
    """

    pattern = re.compile(
        rf"[^{unicode_range}\s0-9"
        r"\.,!?;:'\""
        r"\(\)\[\]\{\}"
        r"\-_/\\"
        r"%&+=*#@"
        r"।॥"
        r"]"
    )

    line = pattern.sub("", line)

    # Collapse repeated whitespace
    line = re.sub(r"\s+", " ", line)

    return line.strip()


# ============================================================
# CREATE CLEAN TRAINING CORPUS
# ============================================================

def prepare_training_corpus(language, cfg):

    train_files = sorted(
        cfg["train_dir"].glob("train_*.txt")
    )

    if not train_files:
        raise FileNotFoundError(
            f"No train_*.txt files found in {cfg['train_dir']}"
        )

    print("\n" + "=" * 70)
    print(f"PREPARING {language.upper()} TRAINING CORPUS")
    print("=" * 70)

    print(f"Training files found: {len(train_files)}")

    clean_dir = cfg["out"] / "training_data"
    clean_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    clean_file = (
        clean_dir
        / f"{language.lower()}_train_clean.txt"
    )

    total_lines = 0
    kept_lines = 0

    with open(
        clean_file,
        "w",
        encoding="utf-8"
    ) as out:

        for i, file in enumerate(train_files, 1):

            print(
                f"  [{i}/{len(train_files)}] "
                f"{file.name}"
            )

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    total_lines += 1

                    cleaned = clean_line(
                        line,
                        cfg["unicode_range"]
                    )

                    if not cleaned:
                        continue

                    # Require at least one target-script character
                    if not re.search(
                        rf"[{cfg['unicode_range']}]",
                        cleaned
                    ):
                        continue

                    out.write(cleaned + "\n")
                    kept_lines += 1

    print()
    print(f"Original lines : {total_lines:,}")
    print(f"Kept lines     : {kept_lines:,}")
    print(f"Clean corpus   : {clean_file}")

    return clean_file


# ============================================================
# TRAIN TOKENIZER
# ============================================================

def train_tokenizer(
    language,
    cfg,
    clean_training_file,
    vocab_size
):

    model_prefix = (
        cfg["out"]
        / f"{language.lower()}_{vocab_size // 1000}k"
    )

    print()
    print(
        f"Training {language} "
        f"{vocab_size:,} vocabulary..."
    )

    spm.SentencePieceTrainer.train(

        input=str(clean_training_file),

        model_prefix=str(model_prefix),

        vocab_size=vocab_size,

        model_type="bpe",

        character_coverage=1.0,

        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,

        # Prevent SentencePiece from adding arbitrary
        # extra characters outside the training corpus.
        hard_vocab_limit=False,

        # Preserve whitespace information.
        remove_extra_whitespaces=True,

        # Use the normal SentencePiece whitespace marker.
        add_dummy_prefix=True,

        # No user-defined characters.
        user_defined_symbols=[],

        # No control symbols.
        control_symbols=[],
    )

    model = model_prefix.with_suffix(".model")

    print(
        f"Saved: {model}"
    )

    return model


# ============================================================
# EVALUATE FERTILITY
# ============================================================

def evaluate(model, test_dir):

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model)
    )

    words = 0
    tokens = 0

    test_files = sorted(
        test_dir.glob("test_*.txt")
    )

    for file in test_files:

        print(
            f"  Testing {file.name}"
        )

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                words += len(
                    line.split()
                )

                tokens += len(
                    tokenizer.encode(
                        line,
                        out_type=str
                    )
                )

    if words == 0:
        return 0

    return tokens / words


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN UNICODE TOKENIZER TRAINING")
    print("Gujarati + Assamese")
    print("=" * 70)

    for language, cfg in CONFIG.items():

        # ----------------------------------------------------
        # 1. Prepare ALL train_*.txt files
        # ----------------------------------------------------

        clean_training_file = (
            prepare_training_corpus(
                language,
                cfg
            )
        )

        # ----------------------------------------------------
        # 2. Train 16K / 24K / 32K
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(f"TRAINING {language.upper()} TOKENIZERS")
        print("=" * 70)

        for vocab_size in VOCAB_SIZES:

            model = train_tokenizer(
                language,
                cfg,
                clean_training_file,
                vocab_size
            )

            # ------------------------------------------------
            # 3. Evaluate on ORIGINAL test data
            # ------------------------------------------------

            fertility = evaluate(
                model,
                cfg["train_dir"]
            )

            print(
                f"{vocab_size // 1000}K "
                f"→ fertility = {fertility:.4f}"
            )

        print()
        print(
            f"{language}: DONE"
        )


if __name__ == "__main__":
    main()
from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "Gujarati": {
        "train": ROOT / "Gujarati/data/splits_v2/train_0000.txt" ,
        "test": ROOT / "Gujarati/data/splits_v2",
        "out": ROOT / "Gujarati/tokenizer"
    },
    "Assamese": {
        "train": ROOT / "Assamese/data/splits_v2/train_0000.txt",
        "test": ROOT / "Assamese/data/splits_v2",
        "out": ROOT / "Assamese/tokenizer"
    }
}

VOCAB_SIZES = [16000, 24000, 32000]


def train_tokenizer(language, cfg, vocab_size):

    model_prefix = cfg["out"] / f"{language.lower()}_{vocab_size // 1000}k"

    print(f"\nTraining {language} {vocab_size:,}...")

    spm.SentencePieceTrainer.train(
        input=str(cfg["train"]),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    return model_prefix.with_suffix(".model")


def evaluate(model, test_dir):

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model)
    )

    words = 0
    tokens = 0

    files = sorted(test_dir.glob("test_*.txt"))

    for file in files:

        print(f"  Testing {file.name}")

        with open(file, encoding="utf-8") as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                words += len(line.split())
                tokens += len(
                    tokenizer.encode(
                        line,
                        out_type=str
                    )
                )

    return tokens / words


def main():

    for language, cfg in CONFIG.items():

        print("\n" + "=" * 60)
        print(language.upper())
        print("=" * 60)

        cfg["out"].mkdir(
            parents=True,
            exist_ok=True
        )

        for vocab_size in VOCAB_SIZES:

            model = train_tokenizer(
                language,
                cfg,
                vocab_size
            )

            fertility = evaluate(
                model,
                cfg["test"]
            )

            print(
                f"{vocab_size // 1000}K → "
                f"fertility = {fertility:.4f}"
            )


if __name__ == "__main__":
    main()
from datasets import load_dataset
from pathlib import Path
import re
import unicodedata


# ============================================================
# SETTINGS
# ============================================================

OUT_DIR = Path(
    "Assamese/data/raw/downloaded/SangrahaVerified"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MIN_WORDS = 20
MIN_ASSAMESE_RATIO = 0.70


# ============================================================
# ASSAMESE SCRIPT
# ============================================================

def is_assamese(ch):
    return "\u0980" <= ch <= "\u09FF"


def assamese_ratio(text):
    """
    Percentage of alphabetic characters written
    in Assamese/Bengali Unicode block.

    Assamese uses the Bengali Unicode block, so this
    is a script-level filter rather than a perfect
    Assamese-vs-Bengali language classifier.
    """

    letters = [
        ch for ch in text
        if ch.isalpha()
    ]

    if not letters:
        return 0.0

    assamese = sum(
        is_assamese(ch)
        for ch in letters
    )

    return assamese / len(letters)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Remove null characters
    text = text.replace(
        "\x00",
        ""
    )

    kept = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        words = line.split()

        if len(words) < MIN_WORDS:
            continue

        # Skip Roman/English-heavy lines
        if assamese_ratio(line) < MIN_ASSAMESE_RATIO:
            continue

        kept.append(line)

    return "\n".join(kept)


# ============================================================
# DOWNLOAD
# ============================================================

def main():

    print("=" * 70)
    print("SANGRAHA VERIFIED — ASSAMESE")
    print("=" * 70)

    print("Loading verified/asm...")
    print("This may take some time.")

    dataset = load_dataset(
        "ai4bharat/sangraha",
        data_dir="verified/asm"
    )

    print(dataset)

    total = 0
    saved = 0
    skipped = 0

    total_words = 0
    total_chars = 0

    for split_name in dataset:

        split = dataset[split_name]

        print(
            f"\nProcessing split: "
            f"{split_name}"
        )

        output_file = (
            OUT_DIR /
            f"sangraha_{split_name}.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as out:

            for row in split:

                total += 1

                text = row["text"]

                if not text:
                    skipped += 1
                    continue

                cleaned = clean_text(
                    text
                )

                if not cleaned:
                    skipped += 1
                    continue

                out.write(
                    cleaned
                    + "\n"
                )

                words = len(
                    cleaned.split()
                )

                total_words += words
                total_chars += len(cleaned)

                saved += 1

                if saved % 10000 == 0:

                    print(
                        f"Saved: {saved:,} | "
                        f"Words: {total_words:,}"
                    )

        print(
            f"Written: {output_file}"
        )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Rows processed : {total:,}"
    )

    print(
        f"Rows saved     : {saved:,}"
    )

    print(
        f"Rows skipped   : {skipped:,}"
    )

    print(
        f"Words saved    : {total_words:,}"
    )

    print(
        f"Characters     : {total_chars:,}"
    )

    print(
        f"\nOutput:"
    )

    print(
        OUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()
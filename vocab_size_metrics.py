from pathlib import Path
import sentencepiece as spm
import json
import csv
import time


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

CONFIG = {

    "Gujarati": {

        # NEW CLEANED TOKENIZERS
        "tokenizers": {
            "16K": ROOT / "Gujarati/tokenizer_clean/gujarati_16k.model",
            "24K": ROOT / "Gujarati/tokenizer_clean/gujarati_24k.model",
            "32K": ROOT / "Gujarati/tokenizer_clean/gujarati_32k.model",
        },

        "test": ROOT / "Gujarati/data/splits_v2",
    },


    "Assamese": {

        # NEW CLEANED TOKENIZERS
        "tokenizers": {
            "16K": ROOT / "Assamese/tokenizer_clean/assamese_16k.model",
            "24K": ROOT / "Assamese/tokenizer_clean/assamese_24k.model",
            "32K": ROOT / "Assamese/tokenizer_clean/assamese_32k.model",
        },

        "test": ROOT / "Assamese/data/splits_v2",
    },
}


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model_path, test_dir):

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    test_files = sorted(
        test_dir.glob("test_*.txt")
    )

    if not test_files:
        return None

    total_words = 0
    total_tokens = 0
    total_unk = 0

    total_lines = 0
    total_chars = 0

    start = time.time()

    print(
        f"    Test files: {len(test_files)}"
    )

    for file in test_files:

        print(
            f"      {file.name}"
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

                total_lines += 1
                total_chars += len(line)

                # --------------------------------------------
                # WORD COUNT
                # --------------------------------------------

                words = line.split()

                total_words += len(words)

                # --------------------------------------------
                # TOKENIZATION
                # --------------------------------------------

                pieces = tokenizer.encode(
                    line,
                    out_type=str
                )

                total_tokens += len(pieces)

                # --------------------------------------------
                # UNKNOWN TOKENS
                # --------------------------------------------

                total_unk += sum(
                    1
                    for piece in pieces
                    if piece == "<unk>"
                )

    elapsed = time.time() - start

    if total_words == 0:
        return None

    # --------------------------------------------
    # METRICS
    # --------------------------------------------

    fertility = (
        total_tokens / total_words
    )

    unk_percentage = (
        total_unk / total_tokens * 100
        if total_tokens
        else 0
    )

    chars_per_token = (
        total_chars / total_tokens
        if total_tokens
        else 0
    )

    tokens_per_character = (
        total_tokens / total_chars
        if total_chars
        else 0
    )

    return {

        "test_files": len(test_files),

        "lines": total_lines,

        "words": total_words,

        "tokens": total_tokens,

        "fertility": fertility,

        "unk_tokens": total_unk,

        "unk_percentage": unk_percentage,

        "characters": total_chars,

        "characters_per_token": chars_per_token,

        "tokens_per_character": tokens_per_character,

        "time_seconds": elapsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TOKENIZER COMPARISON — CLEAN GUJARATI + ASSAMESE")
    print("=" * 75)

    all_results = []

    # ========================================================
    # EVALUATE ALL MODELS
    # ========================================================

    for language, cfg in CONFIG.items():

        print("\n")
        print("=" * 75)
        print(language.upper())
        print("=" * 75)

        for vocab, model_path in cfg["tokenizers"].items():

            print(
                f"\n  {vocab}"
            )

            # --------------------------------------------
            # CHECK MODEL
            # --------------------------------------------

            if not model_path.exists():

                print(
                    "    MODEL NOT FOUND:"
                )

                print(
                    f"    {model_path}"
                )

                continue

            # --------------------------------------------
            # EVALUATE
            # --------------------------------------------

            result = evaluate(
                model_path,
                cfg["test"]
            )

            if result is None:

                print(
                    "    No test files found."
                )

                continue

            # --------------------------------------------
            # ADD METADATA
            # --------------------------------------------

            result["language"] = language

            result["vocab"] = vocab

            result["model"] = str(
                model_path
            )

            all_results.append(
                result
            )

            # --------------------------------------------
            # PRINT RESULTS
            # --------------------------------------------

            print(
                f"    Words       : "
                f"{result['words']:,}"
            )

            print(
                f"    Tokens      : "
                f"{result['tokens']:,}"
            )

            print(
                f"    Fertility   : "
                f"{result['fertility']:.4f}"
            )

            print(
                f"    UNK tokens  : "
                f"{result['unk_tokens']:,}"
            )

            print(
                f"    UNK %       : "
                f"{result['unk_percentage']:.6f}%"
            )

            print(
                f"    Characters  : "
                f"{result['characters']:,}"
            )

            print(
                f"    Chars/token : "
                f"{result['characters_per_token']:.4f}"
            )

            print(
                f"    Tok/char    : "
                f"{result['tokens_per_character']:.4f}"
            )

            print(
                f"    Time        : "
                f"{result['time_seconds']:.2f}s"
            )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FERTILITY COMPARISON")
    print("=" * 100)

    print(
        f"{'Language':<12}"
        f"{'Vocab':<10}"
        f"{'Words':>15}"
        f"{'Tokens':>15}"
        f"{'Fertility':>12}"
        f"{'UNK %':>12}"
    )

    print("-" * 100)

    for r in all_results:

        print(
            f"{r['language']:<12}"
            f"{r['vocab']:<10}"
            f"{r['words']:>15,}"
            f"{r['tokens']:>15,}"
            f"{r['fertility']:>12.4f}"
            f"{r['unk_percentage']:>12.6f}"
        )


    # ========================================================
    # BEST FERTILITY
    # ========================================================

    print("\n")
    print("=" * 75)
    print("BEST FERTILITY")
    print("=" * 75)

    for language in CONFIG:

        results = [
            r
            for r in all_results
            if r["language"] == language
        ]

        if not results:
            continue

        best = min(
            results,
            key=lambda x: x["fertility"]
        )

        print(
            f"{language}: "
            f"{best['vocab']} "
            f"→ fertility "
            f"{best['fertility']:.4f}"
        )


    # ========================================================
    # SAVE JSON
    # ========================================================

    output_json = (
        ROOT / "tokenizer_comparison.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # SAVE CSV
    # ========================================================

    output_csv = (
        ROOT / "tokenizer_comparison.csv"
    )

    if all_results:

        fields = [
            "test_files",
            "lines",
            "words",
            "tokens",
            "fertility",
            "unk_tokens",
            "unk_percentage",
            "characters",
            "characters_per_token",
            "tokens_per_character",
            "time_seconds",
            "language",
            "vocab",
            "model",
        ]

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(
                all_results
            )


    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        f"JSON: {output_json}"
    )

    print(
        f"CSV : {output_csv}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

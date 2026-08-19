from pathlib import Path
from collections import Counter
import sentencepiece as spm
import pandas as pd

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "Gujarati": {
        "models": {
            "16K": ROOT / "Gujarati/tokenizer/gujarati_16k.model",
            "24K": ROOT / "Gujarati/tokenizer/gujarati_24k.model",
            "32K": ROOT / "Gujarati/tokenizer/gujarati_32k.model",
        },
        "test": ROOT / "Gujarati/data/splits_v2",
    },

    "Assamese": {
        "models": {
            "16K": ROOT / "Assamese/tokenizer/assamese_16k.model",
            "24K": ROOT / "Assamese/tokenizer/assamese_24k.model",
            "32K": ROOT / "Assamese/tokenizer/assamese_32k.model",
        },
        "test": ROOT / "Assamese/data/splits_v2",
    },
}


results = []


for language, cfg in CONFIG.items():

    test_files = sorted(
        cfg["test"].glob("test_*.txt")
    )

    for vocab, model_path in cfg["models"].items():

        print(f"{language} {vocab}")

        tokenizer = spm.SentencePieceProcessor(
            model_file=str(model_path)
        )

        counter = Counter()
        total_tokens = 0

        for file in test_files:

            with open(file, encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    tokens = tokenizer.encode(
                        line,
                        out_type=str
                    )

                    counter.update(tokens)
                    total_tokens += len(tokens)

        unique_tokens = len(counter)

        most_frequent_token, frequency = (
            counter.most_common(1)[0]
        )

        share = (
            frequency / total_tokens * 100
        )

        results.append({
            "Language": language,
            "Vocabulary": vocab,
            "Unique tokens observed": unique_tokens,
            "Most frequent token": most_frequent_token,
            "Frequency": frequency,
            "Share (%)": share,
        })


df = pd.DataFrame(results)

print("\n" + "=" * 80)
print(df.to_string(index=False))
print("=" * 80)

df.to_csv(
    "token_frequency_statistics.csv",
    index=False
)
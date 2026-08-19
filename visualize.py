# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("tokenizer_comparison.csv")

# plt.figure(figsize=(8, 5))

# for language in df["language"].unique():

#     data = df[df["language"] == language]

#     # Convert "16K" → 16
#     vocab = data["vocab"].str.replace("K", "").astype(int)

#     plt.plot(
#         vocab,
#         data["fertility"],
#         marker="o",
#         label=language
#     )

# plt.xlabel("Vocabulary Size (K)")
# plt.ylabel("Fertility (tokens/word)")
# plt.title("Vocabulary Size vs Fertility")

# plt.xticks([16, 24, 32])
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.tight_layout()
# plt.savefig("vocabulary_vs_fertility.png", dpi=300)
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv("tokenizer_comparison.csv")

df["vocab_k"] = (
    df["vocab"]
    .str.replace("K", "", regex=False)
    .astype(int)
)

# Sort properly
df = df.sort_values(["vocab_k", "language"])


# ============================================================
# FUNCTION FOR GROUPED BAR CHART
# ============================================================

def grouped_bar(
    df,
    value_column,
    ylabel,
    title,
    filename
):

    pivot = df.pivot(
        index="vocab_k",
        columns="language",
        values=value_column
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5)
    )

    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticklabels(
        [f"{x}K" for x in pivot.index],
        rotation=0
    )

    ax.legend(title="Language")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 1. FERTILITY
# ============================================================

grouped_bar(
    df,
    "fertility",
    "Fertility (tokens / word)",
    "Tokenizer Fertility Comparison",
    "fertility_comparison.png"
)


# ============================================================
# 2. CHARACTERS PER TOKEN
# ============================================================

grouped_bar(
    df,
    "characters_per_token",
    "Average characters per token",
    "Average Characters per Token",
    "characters_per_token.png"
)


# ============================================================
# 3. UNKNOWN TOKEN %
# ============================================================

grouped_bar(
    df,
    "unk_percentage",
    "Unknown tokens (%)",
    "Unknown Token Percentage",
    "unk_percentage.png"
)


# ============================================================
# 4. TOTAL TOKENS
# ============================================================

grouped_bar(
    df,
    "tokens",
    "Number of tokens",
    "Total Tokens Produced on Test Set",
    "total_tokens.png"
)
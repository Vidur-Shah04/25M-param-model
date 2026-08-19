# # from pathlib import Path
# # import re

# # DATA = Path(r"Gujarati\data\splits_v2")

# # DELETE = {
# #     "train": 200_000_000,
# #     "val": 25_000_000,
# #     "test": 25_000_000
# # }

# # for split, target in DELETE.items():

# #     files = sorted(DATA.glob(f"{split}_*.txt"))

# #     # Work backwards so earlier data is preserved
# #     for file in reversed(files):

# #         text = file.read_text(
# #             encoding="utf-8"
# #         )

# #         docs = re.split(
# #             r"\n\s*\n+",
# #             text
# #         )

# #         removed = 0
# #         kept = []

# #         # Remove documents from the end
# #         for doc in reversed(docs):

# #             if not doc.strip():
# #                 continue
# #             words = len(doc.split())
# #             if removed + words <= target:
# #                 removed += words
# #             else:
# #                 kept.append(doc)

# #         # Put remaining documents back in original order
# #         kept.reverse()

# #         output = file.with_name(
# #             file.stem + "_trimmed.txt"
# #         )

# #         output.write_text(
# #             "\n\n".join(kept),
# #             encoding="utf-8"
# #         )

# #         target -= removed

# #         print(
# #             f"{split}: removed {removed:,} words "
# #             f"from {file.name}"
# #         )

# #         if target <= 0:
# #             break

# #     print(
# #         f"{split}: deletion complete\n"
# #     )
# from pathlib import Path
# import re

# DATA = Path(r"Gujarati\data\splits_v2")

# REMOVE = 50_000_000


# def documents(path):
#     """Yield one document at a time."""
#     doc = []

#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 doc.append(line)
#             elif doc:
#                 yield "".join(doc)
#                 doc = []

#         if doc:
#             yield "".join(doc)


# def count_words(path):
#     total = 0

#     for doc in documents(path):
#         total += len(doc.split())

#     return total


# # Only ORIGINAL train files.
# # files = sorted(
# #     DATA.glob("train_*.txt"),
# #     key=lambda p: int(
# #         re.search(r"train_(\d+)\.txt$", p.name).group(1)
# #     )
# # )

# files = sorted(
#     DATA.glob("train_????.txt")
# )
# remaining = REMOVE

# for file in reversed(files):

#     if remaining <= 0:
#         break

#     # Ignore already-created trimmed files
#     if "_trimmed" in file.stem:
#         continue

#     total = count_words(file)

#     if total <= remaining:
#         print(
#             f"{file.name}: remove entire file "
#             f"({total:,} words)"
#         )
#         remaining -= total
#         continue

#     # We only need to trim part of this file.
#     target_keep = total - remaining

#     print(
#         f"{file.name}: "
#         f"{total:,} words -> "
#         f"keep ~{target_keep:,}"
#     )

#     output = file.with_name(
#         file.stem + "_final.txt"
#     )

#     kept_words = 0

#     with open(
#         output,
#         "w",
#         encoding="utf-8"
#     ) as out:

#         for doc in documents(file):

#             words = len(doc.split())

#             if kept_words + words > target_keep:
#                 break

#             out.write(doc)
#             out.write("\n")

#             kept_words += words

#     removed = total - kept_words

#     print(
#         f"  kept   : {kept_words:,}"
#     )

#     print(
#         f"  removed: {removed:,}"
#     )

#     remaining = 0


# print("\n" + "=" * 50)

# print(
#     f"Requested removal : {REMOVE:,}"
# )

# print(
#     f"Still remaining   : {max(remaining, 0):,}"
# )

# print("=" * 50)
from pathlib import Path

DATA = Path(r"Gujarati\data\raw\downloaded\IITB-IndicMonoDoc\gu")
OUT = DATA.parent / "gu_pruned"

REMOVE = 250_000_000


def documents(path):
    doc = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "<DOC_START>":
                continue

            if line == "<DOC_END>":
                if doc:
                    yield "\n".join(doc)
                    doc = []
            else:
                doc.append(line)

        if doc:
            yield "\n".join(doc)


OUT.mkdir(exist_ok=True)

remaining = REMOVE

files = sorted(DATA.glob("*.txt"))

for file in files:

    if remaining <= 0:
        break

    output = OUT / file.name

    removed = 0
    kept = 0

    with open(output, "w", encoding="utf-8") as out:

        for doc in documents(file):

            words = len(doc.split())

            if removed + words <= remaining:
                removed += words
                continue

            out.write(doc)
            out.write("\n\n")
            kept += words

    remaining -= removed

    print(
        f"{file.name}: "
        f"removed {removed:,} words | "
        f"kept {kept:,}"
    )

print("\n" + "=" * 60)
print(f"Requested removal : {REMOVE:,}")
print(f"Still remaining   : {max(remaining, 0):,}")
print("=" * 60)
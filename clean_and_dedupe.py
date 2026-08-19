import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

MIN_DOC_WORDS = 50
MIN_PARA_WORDS = 20

SHINGLE_SIZE = 5
NUM_PERM = 64
MINHASH_THRESHOLD = 0.85

TRAIN_PERCENT = 80
VAL_PERCENT = 10
TEST_PERCENT = 10

DOCS_PER_SHARD = 100_000

HPLT_REPO = "HPLT/HPLT2.0_cleaned"


# ============================================================
# LANGUAGE CONFIG
# ============================================================

CONFIG = {

    # ========================================================
    # GUJARATI
    # ========================================================

    "Gujarati": {

        "script": (0x0A80, 0x0AFF),

        # Measured on the held-out test set using the 32K tokenizer
        "tokens_per_word": 1.5166424632,

        # ========================================================
        # MANUAL SOURCES
        # ========================================================

        "manual": [

            (
                "archive_org",
                ROOT / "Gujarati/data/raw/manual/archive_org",
                "file"
            ),

            (
                "ebooks",
                ROOT / "Gujarati/data/raw/manual/ebooks",
                "file"
            ),

            (
                "vishwakosh",
                ROOT / "Gujarati/data/raw/manual/vishwakosh",
                "file"
            ),

            (
                "wiki_dumps",
                ROOT / "Gujarati/data/raw/manual/wiki dumps",
                "file"
            ),

            (
                "wikisource",
                ROOT / "Gujarati/data/raw/manual/wikisource",
                "file"
            ),
        ],

        # ========================================================
        # PREMADE SOURCES
        # ========================================================

        "premade": [

            (
                "iitb",
                ROOT / "Gujarati/data/raw/downloaded/IITB-IndicMonoDoc/gu",
                "markers"
            ),
        ],
    },
        # ========================================================
        # ASSAMESE
        # ========================================================


    "Assamese": {

        "script": (0x0980, 0x09FF),

        # Measured using the 32K tokenizer
        "tokens_per_word": 1.6633,

        # ----------------------------------------------------
        # MANUAL SOURCES
        # ----------------------------------------------------

        "manual": [

            (
                "archive_org",
                ROOT / "Assamese/data/raw/manual/archive_org",
                "file"
            ),
        ],

        # ----------------------------------------------------
        # PREMADE SOURCES
        # ----------------------------------------------------

        "premade": [

            (
                "iitb",
                ROOT / "Assamese/data/raw/downloaded/IITB-IndicMonoDoc/as",
                "markers"
            ),

            (
                "indiccorpv2",
                ROOT / "Assamese/data/raw/downloaded/IndicCorpV2/data/as.txt",
                "lines"
            ),

            (
                "sangraha",
                ROOT / "Assamese/data/raw/downloaded/SangrahaVerified/sangraha_train.txt",
                "lines"
            ),
        ],
    },
}
# ============================================================
# CLEANING
# ============================================================

URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\S+@\S+"
)

SPACE_RE = re.compile(
    r"[ \t]+"
)

PARAGRAPH_RE = re.compile(
    r"\n\s*\n+"
)


def clean_text(text, cfg):

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.replace(
        "\ufeff",
        ""
    )

    text = URL_RE.sub(
        " ",
        text
    )

    start, end = cfg["script"]

    cleaned = []

    for line in text.splitlines():

        line = SPACE_RE.sub(
            " ",
            line
        ).strip()

        if not line:
            continue

        letters = [
            c for c in line
            if c.isalpha()
        ]

        if not letters:
            continue

        script_count = sum(
            start <= ord(c) <= end
            for c in letters
        )

        # At least 30% of alphabetic characters
        # must belong to the target script.
        if (
            script_count / len(letters)
            < 0.30
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================================
# PARAGRAPHS
# ============================================================

def get_paragraphs(text):

    return [
        p.strip()
        for p in PARAGRAPH_RE.split(text)
        if p.strip()
    ]


def normalized(text):

    return " ".join(
        text.split()
    )


def sha256_text(text):

    return hashlib.sha256(
        normalized(text).encode(
            "utf-8"
        )
    ).digest()


# ============================================================
# MINHASH
# ============================================================

def make_minhash(text):

    words = text.split()

    m = MinHash(
        num_perm=NUM_PERM
    )

    for i in range(
        len(words) - SHINGLE_SIZE + 1
    ):

        shingle = " ".join(
            words[
                i:i + SHINGLE_SIZE
            ]
        )

        m.update(
            shingle.encode(
                "utf-8"
            )
        )

    return m


# ============================================================
# DOCUMENT SPLIT
# ============================================================

def choose_split(doc_hash):

    value = int.from_bytes(
        doc_hash[:4],
        "big"
    ) % 100

    if value < TEST_PERCENT:
        return "test"

    if value < TEST_PERCENT + VAL_PERCENT:
        return "val"

    return "train"

# ============================================================
# DOCUMENT READER
# ============================================================

def read_documents(path, mode):

    # --------------------------------------------------------
    # NORMAL FILE
    # --------------------------------------------------------

    if mode == "file":

        yield (
            path.stem,
            path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        )

        return


    # --------------------------------------------------------
    # ONE DOCUMENT PER LINE
    # --------------------------------------------------------

    if mode == "lines":

        with open(
            path,
            encoding="utf-8",
            errors="replace"
        ) as f:

            for number, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if line:

                    yield (
                        f"{path.stem}:{number}",
                        line
                    )

        return


    # --------------------------------------------------------
    # IITB <DOC_START> / <DOC_END>
    # --------------------------------------------------------

    if mode == "markers":

        document = []
        inside = False
        number = 0

        with open(
            path,
            encoding="utf-8",
            errors="replace"
        ) as f:

            for line in f:

                stripped = line.strip()

                if stripped == "<DOC_START>":

                    document = []
                    inside = True

                elif stripped == "<DOC_END>":

                    if document:

                        number += 1

                        yield (
                            f"{path.stem}:{number}",
                            "\n".join(
                                document
                            )
                        )

                    document = []
                    inside = False

                elif inside:

                    document.append(
                        line.rstrip()
                    )

        return


    # --------------------------------------------------------
    # HPLT PARQUET
    # --------------------------------------------------------

    if mode == "hplt":

        # HPLT directory contains multiple Parquet shards.
        parquet_files = sorted(
            path.rglob("*.parquet")
        )

        for parquet_file in parquet_files:

            print(
                f"Reading HPLT: "
                f"{parquet_file.name}"
            )

            # Read in batches instead of loading
            # the complete Parquet file into RAM.
            parquet = pq.ParquetFile(
                parquet_file
            )

            number = 0

            for batch in parquet.iter_batches(
                batch_size=10_000,
                columns=["text"]
            ):

                texts = batch.column(
                    "text"
                ).to_pylist()

                for text in texts:

                    if not text:
                        continue

                    number += 1

                    yield (
                        f"{parquet_file.stem}:{number}",
                        text
                    )

        return


# ============================================================
# OUTPUT WRITER
# ============================================================

class ShardWriter:

    def __init__(self, directory):

        self.directory = directory

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.files = {}

        self.counts = {
            "train": 0,
            "val": 0,
            "test": 0
        }


    def write(
        self,
        split,
        text
    ):

        shard = (
            self.counts[split]
            // DOCS_PER_SHARD
        )

        key = (
            split,
            shard
        )

        if key not in self.files:

            path = (
                self.directory
                / f"{split}_{shard:04d}.txt"
            )

            self.files[key] = open(
                path,
                "w",
                encoding="utf-8"
            )

        self.files[key].write(
            text
        )

        self.files[key].write(
            "\n\n"
        )

        self.counts[split] += 1


    def close(self):

        for f in self.files.values():
            f.close()


# ============================================================
# PROCESS ONE LANGUAGE
# ============================================================

def process_language(
    language,
    cfg
):

    print("\n")
    print("=" * 80)
    print(
        f"{language.upper()} DEDUP + SPLIT"
    )
    print("=" * 80)

    data_dir = (
        ROOT
        / language
        / "data"
    )

    cleaned_dir = (
        data_dir
        / "cleaned_v2"
    )

    split_dir = (
        data_dir
        / "splits_v2"
    )

    cleaned_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # RAM-BASED DEDUP
    # ========================================================

    print(
        "Creating in-memory dedup indexes..."
    )

    paragraph_hashes = set()

    document_hashes = set()

    lsh = MinHashLSH(
        threshold=MINHASH_THRESHOLD,
        num_perm=NUM_PERM
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    writer = ShardWriter(
        split_dir
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    stats = []

    total_documents = 0
    total_words = 0

    manual_words = 0

    exact_para_duplicates = 0
    near_para_duplicates = 0
    exact_doc_duplicates = 0

    source_documents = {}
    source_words = {}


    # ========================================================
    # MANUAL FIRST
    # ========================================================

    sources = (

        [
            (source, path, mode, True)
            for source, path, mode
            in cfg["manual"]
        ]

        +

        [
            (source, path, mode, False)
            for source, path, mode
            in cfg["premade"]
        ]
    )


    # ========================================================
    # SOURCE LOOP
    # ========================================================

    for source, path, mode, manual in sources:

        print("\n")
        print("-" * 80)
        print(
            f"SOURCE: {source}"
        )
        print(
            f"PATH  : {path}"
        )
        print(
            f"MANUAL: {manual}"
        )
        print("-" * 80)


        if not path.exists():

            print(
                "WARNING: path does not exist"
            )

            continue


        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        if mode == "hplt":

            files = sorted(
                path.rglob("*.parquet")
            )

        else:

            files = (
                [path]
                if path.is_file()
                else sorted(
                    path.rglob("*.txt")
                )
            )


        print(
            f"Files: {len(files):,}"
        )


        source_documents[source] = 0
        source_words[source] = 0


        # ====================================================
        # FILE LOOP
        # ====================================================

        for file_number, file in enumerate(
            files,
            start=1
        ):

            print(
                f"\n[{file_number:,}/"
                f"{len(files):,}] "
                f"{file.name}"
            )


            try:

                # HPLT needs the directory because
                # read_documents handles all shards.
                if mode == "hplt":

                    docs = read_documents(
                        path,
                        mode
                    )

                    # Only process the HPLT directory
                    # once, not once per shard.
                    if file_number > 1:
                        break

                else:

                    docs = read_documents(
                        file,
                        mode
                    )


                # ============================================
                # DOCUMENT LOOP
                # ============================================

                for doc_id, raw in docs:

                    text = clean_text(
                        raw,
                        cfg
                    )

                    if not text:
                        continue


                    if (
                        len(text.split())
                        < MIN_DOC_WORDS
                    ):
                        continue


                    kept = []


                    # ========================================
                    # PARAGRAPH DEDUP
                    # ========================================

                    for paragraph in get_paragraphs(
                        text
                    ):

                        words = paragraph.split()

                        if not words:
                            continue


                        # ------------------------------------
                        # EXACT HASH
                        # ------------------------------------

                        paragraph_hash = (
                            sha256_text(
                                paragraph
                            )
                        )


                        if paragraph_hash in paragraph_hashes:

                            exact_para_duplicates += 1

                            continue


                        # ------------------------------------
                        # SHORT PARAGRAPHS
                        # ------------------------------------

                        if len(words) < MIN_PARA_WORDS:

                            paragraph_hashes.add(
                                paragraph_hash
                            )

                            kept.append(
                                paragraph
                            )

                            continue


                        # ------------------------------------
                        # MINHASH
                        # ------------------------------------

                        minhash = make_minhash(
                            paragraph
                        )


                        if lsh.query(
                            minhash
                        ):

                            near_para_duplicates += 1

                            continue


                        # ------------------------------------
                        # UNIQUE PARAGRAPH
                        # ------------------------------------

                        paragraph_hashes.add(
                            paragraph_hash
                        )

                        lsh.insert(
                            paragraph_hash.hex(),
                            minhash
                        )

                        kept.append(
                            paragraph
                        )


                    # ========================================
                    # REBUILD DOCUMENT
                    # ========================================

                    if not kept:
                        continue


                    text = "\n\n".join(
                        kept
                    )

                    words = text.split()


                    if len(words) < MIN_DOC_WORDS:
                        continue


                    # ========================================
                    # DOCUMENT EXACT DEDUP
                    # ========================================

                    document_hash = sha256_text(
                        text
                    )


                    if document_hash in document_hashes:

                        exact_doc_duplicates += 1

                        continue


                    document_hashes.add(
                        document_hash
                    )


                    # ========================================
                    # SPLIT
                    # ========================================

                    split = choose_split(
                        document_hash
                    )


                    writer.write(
                        split,
                        text
                    )


                    # ========================================
                    # STATS
                    # ========================================

                    word_count = len(words)

                    total_documents += 1
                    total_words += word_count

                    source_documents[source] += 1
                    source_words[source] += word_count

                    if manual:
                        manual_words += word_count


            except Exception as e:

                print(
                    f"ERROR: {file}"
                )

                print(e)

                continue


    # ========================================================
    # CLOSE OUTPUT
    # ========================================================

    writer.close()


    # ========================================================
    # SOURCE STATS
    # ========================================================

    for source in source_documents:

        stats.append(
            {
                "source": source,

                "manual": int(
                    any(
                        source == s
                        for s, _, _ in cfg["manual"]
                    )
                ),

                "documents":
                    source_documents[source],

                "words":
                    source_words[source]
            }
        )


    stats_df = pd.DataFrame(
        stats
    )

    stats_df.to_csv(
        cleaned_dir
        / "source_stats.csv",
        index=False
    )


    # ========================================================
    # SPLIT STATS
    # ========================================================

    split_stats = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        words = 0
        documents = 0

        for file in sorted(
            split_dir.glob(
                f"{split}_*.txt"
            )
        ):

            text = file.read_text(
                encoding="utf-8"
            )

            documents += len(
                re.split(
                    r"\n\s*\n+",
                    text
                )
            )

            words += len(
                text.split()
            )


        split_stats.append(
            {
                "split": split,
                "documents": documents,
                "words": words
            }
        )


    split_df = pd.DataFrame(
        split_stats
    )

    split_df.to_csv(
        cleaned_dir
        / "split_stats.csv",
        index=False
    )


    # ========================================================
    # FINAL STATS
    # ========================================================

    premade_words = (
        total_words
        - manual_words
    )


    final_stats = {

        "language": language,

        "documents":
            total_documents,

        "words":
            total_words,

        "manual_words":
            manual_words,

        "premade_words":
            premade_words,

        "manual_percentage":
            (
                manual_words
                / total_words
                * 100
            ),

        "estimated_tokens":
            int(
                total_words
                * cfg["tokens_per_word"]
            ),

        "estimated_manual_tokens":
            int(
                manual_words
                * cfg["tokens_per_word"]
            ),

        "tokens_per_word_estimate":
            cfg["tokens_per_word"],

        "exact_paragraph_duplicates":
            exact_para_duplicates,

        "near_duplicate_paragraphs":
            near_para_duplicates,

        "exact_document_duplicates":
            exact_doc_duplicates,

        "unique_paragraphs":
            len(paragraph_hashes),

        "unique_documents":
            len(document_hashes),

        "split":
            {
                "train": 80,
                "validation": 10,
                "test": 10
            },

        "minhash":
            {
                "threshold":
                    MINHASH_THRESHOLD,

                "num_perm":
                    NUM_PERM,

                "shingle_size":
                    SHINGLE_SIZE
            }
    }


    with open(
        cleaned_dir
        / "corpus_stats.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_stats,
            f,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        f"{language.upper()} COMPLETE"
    )
    print("=" * 80)

    print(
        f"Documents              : "
        f"{total_documents:,}"
    )

    print(
        f"Words                  : "
        f"{total_words:,}"
    )

    print(
        f"Manual words           : "
        f"{manual_words:,}"
    )

    print(
        f"Premade words          : "
        f"{premade_words:,}"
    )

    print(
        f"Manual %               : "
        f"{manual_words / total_words * 100:.2f}%"
    )

    print(
        f"Exact paragraph dups   : "
        f"{exact_para_duplicates:,}"
    )

    print(
        f"Near paragraph dups    : "
        f"{near_para_duplicates:,}"
    )

    print(
        f"Exact document dups    : "
        f"{exact_doc_duplicates:,}"
    )

    print(
        f"Unique paragraphs      : "
        f"{len(paragraph_hashes):,}"
    )

    print(
        f"Unique documents       : "
        f"{len(document_hashes):,}"
    )

    print(
        f"Estimated tokens       : "
        f"{total_words * cfg['tokens_per_word']:,.0f}"
    )

    print(
        f"Estimated manual tokens: "
        f"{manual_words * cfg['tokens_per_word']:,.0f}"
    )

    print(
        "\nSplits:"
    )

    print(
        "  Train: 80%"
    )

    print(
        "  Val  : 10%"
    )

    print(
        "  Test : 10%"
    )

    print(
        f"\nStats:"
    )

    print(
        cleaned_dir
        / "corpus_stats.json"
    )

    print(
        "\nSplits:"
    )

    print(
        split_dir
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start = time.time()

    print("=" * 80)
    print(
        "GUJARATI + ASSAMESE "
        "DEDUPLICATION PIPELINE"
    )
    print("=" * 80)

    print(
        "\nRAM-heavy configuration:"
    )

    print(
        f"  MinHash permutations : "
        f"{NUM_PERM}"
    )

    print(
        f"  Similarity threshold : "
        f"{MINHASH_THRESHOLD}"
    )

    print(
        f"  Shingle size         : "
        f"{SHINGLE_SIZE}"
    )

    print(
        "  Split                : "
        "80 / 10 / 10"
    )

    print(
        "\nWARNING:"
    )

    print(
        "This run builds the global "
        "deduplication indexes in RAM."
    )

    print(
        "Do not run multiple copies "
        "simultaneously."
    )


    # process_language(
    #     "Gujarati",
    #     CONFIG["Gujarati"]
    # )

    process_language(
        "Assamese",
        CONFIG["Assamese"]
    )


    elapsed = (
        time.time() - start
    ) / 3600

    print("\n")
    print("=" * 80)
    print(
        "ALL CLEANING + DEDUP + SPLIT COMPLETE"
    )
    print("=" * 80)

    print(
        f"Elapsed: {elapsed:.2f} hours"
    )
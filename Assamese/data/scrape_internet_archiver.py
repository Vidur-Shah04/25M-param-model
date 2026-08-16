import requests
from pathlib import Path
import re
import time
import random
import csv


# ============================================================
# SETTINGS
# ============================================================

API_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{}"

SEARCH_QUERY = (
    '(language:asm OR language:"Assamese") '
    'AND mediatype:texts'
)

# Start small.
# Increase after you verify the pipeline.
MAX_ITEMS = 20000

BATCH_SIZE = 500

MIN_ASSAMESE_WORDS = 2000

MAX_RETRIES = 5

REQUEST_TIMEOUT = 60

DELAY_BETWEEN_ITEMS = 0.25

DELAY_BETWEEN_BATCHES = 2.0

# OUTPUT_DIR = Path(
#     r"Gujarati\data\raw\manual\archive_org"
# )
OUTPUT_DIR = Path(
    r"Assamese\data\raw\manual\archive_org"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_LOG = (
    OUTPUT_DIR / "processed.tsv"
)

FAILED_LOG = (
    OUTPUT_DIR / "failures.tsv"
)

STATS_LOG = (
    OUTPUT_DIR / "stats.tsv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}

session = requests.Session()

session.headers.update(
    HEADERS
)


# ============================================================
# GUJARATI CHARACTER FUNCTIONS
# ============================================================

# def is_gujarati_char(ch):
#     return "\u0A80" <= ch <= "\u0AFF"
def is_assamese_char(ch):
    return "\u0980" <= ch <= "\u09FF"

# def gujarati_ratio(text):

#     letters = [
#         ch
#         for ch in text
#         if ch.isalpha()
#     ]

#     if not letters:
#         return 0.0

#     gujarati_count = sum(
#         is_gujarati_char(ch)
#         for ch in letters
#     )

#     return (
#         gujarati_count
#         / len(letters)
#     )

def assamese_ratio(text):

    letters = [
        ch for ch in text
        if ch.isalpha()
    ]

    if not letters:
        return 0.0

    assamese_count = sum(
        is_assamese_char(ch)
        for ch in letters
    )

    return assamese_count / len(letters)

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove obvious OCR garbage
        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if not line:
            continue

        # ratio = gujarati_ratio(
        #     line
        # )
        ratio = assamese_ratio(line)

        # Entirely/non-substantially foreign lines
        if ratio < 0.30:
            continue

        cleaned_lines.append(
            line
        )

    return "\n".join(
        cleaned_lines
    )


# ============================================================
# SAFE FILENAME
# ============================================================

def safe_filename(identifier):

    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        identifier
    )

    return name + ".txt"


# ============================================================
# LOG HELPERS
# ============================================================

def load_processed():

    if not PROCESSED_LOG.exists():
        return set()

    processed = set()

    with open(
        PROCESSED_LOG,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            identifier = (
                line.split("\t", 1)[0]
            )

            processed.add(
                identifier
            )

    return processed


def log_processed(
    identifier,
    title,
    words,
    chars
):

    new_file = not PROCESSED_LOG.exists()

    with open(
        PROCESSED_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        if new_file:
            f.write(
                "identifier\ttitle\twords\tchars\n"
            )

        f.write(
            f"{identifier}\t"
            f"{title}\t"
            f"{words}\t"
            f"{chars}\n"
        )


def log_failed(
    identifier,
    error
):

    new_file = not FAILED_LOG.exists()

    with open(
        FAILED_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        if new_file:
            f.write(
                "identifier\terror\n"
            )

        f.write(
            f"{identifier}\t"
            f"{repr(error)}\n"
        )


# def log_stats(
#     identifier,
#     title,
#     raw_words,
#     gujarati_words,
#     chars,
#     text_url
# ):
def log_stats(
    identifier,
    title,
    raw_words,
    assamese_words,
    chars,
    text_url
):

    new_file = not STATS_LOG.exists()

    with open(
        STATS_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        if new_file:

            f.write(
                "identifier\t"
                "title\t"
                "raw_words\t"
                "assamese_words\t"
                "chars\t"
                "text_url\n"
            )

        f.write(
            f"{identifier}\t"
            f"{title}\t"
            f"{raw_words}\t"
            f"{assamese_words}\t"
            f"{chars}\t"
            f"{text_url}\n"
        )
# ============================================================
# REQUEST WITH RETRY
# ============================================================

def request_with_retry(
    url,
    params=None,
    timeout=REQUEST_TIMEOUT
):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=timeout
            )

            response.raise_for_status()

            return response

        except requests.RequestException as e:

            last_error = e

            if attempt == MAX_RETRIES:
                break

            wait = (
                2 ** (attempt - 1)
                + random.uniform(0, 1)
            )

            print(
                f"      Retry "
                f"{attempt}/{MAX_RETRIES - 1} "
                f"in {wait:.1f}s"
            )

            time.sleep(wait)

    raise last_error


# ============================================================
# SEARCH ARCHIVE.ORG
# ============================================================

def get_identifiers():

    print(
        "Searching Archive.org..."
    )

    params = {
        "q": SEARCH_QUERY,

        "fl[]": [
            "identifier",
            "title"
        ],

        "rows": BATCH_SIZE,

        "page": 1,

        "output": "json"
    }

    response = request_with_retry(
        API_URL,
        params=params
    )

    data = response.json()

    total_found = (
        data["response"]["numFound"]
    )

    print(
        f"Total matching items: "
        f"{total_found:,}"
    )

    target = min(
        total_found,
        MAX_ITEMS
    )

    print(
        f"Will process at most: "
        f"{target:,}"
    )

    identifiers = []

    page = 1

    while len(identifiers) < target:

        print(
            f"Fetching search batch "
            f"{page}..."
        )

        params = {

            "q": SEARCH_QUERY,

            "fl[]": [
                "identifier",
                "title"
            ],

            "rows": BATCH_SIZE,

            "page": page,

            "output": "json"
        }

        response = request_with_retry(
            API_URL,
            params=params
        )

        # data = response.json()

        # docs = (
        #     data["response"]["docs"]
        # )

        # if not docs:
        #     break
        try:
            data = response.json()
        except ValueError:
            print(
                "  -> Archive.org returned "
                "non-JSON response. Stopping search."
            )
            break

        if "response" not in data:
            print(
                "  -> Archive.org returned an "
                "unexpected response:"
            )
            print(data)
            break

        docs = data["response"].get(
            "docs",
            []
        )

        if not docs:
            break

        for item in docs:

            identifier = item.get(
                "identifier"
            )

            if identifier:

                identifiers.append(
                    (
                        identifier,
                        item.get(
                            "title",
                            ""
                        )
                    )
                )

                if len(identifiers) >= target:
                    break

        page += 1

        time.sleep(
            DELAY_BETWEEN_BATCHES
        )

    # Deduplicate
    seen = set()

    unique = []

    for identifier, title in identifiers:

        if identifier in seen:
            continue

        seen.add(identifier)

        unique.append(
            (
                identifier,
                title
            )
        )

    print(
        f"Unique identifiers collected: "
        f"{len(unique):,}"
    )

    return unique


# ============================================================
# GET METADATA
# ============================================================

def get_metadata(identifier):

    url = METADATA_URL.format(
        identifier
    )

    response = request_with_retry(
        url
    )

    return response.json()


# ============================================================
# FIND FULL TEXT FILE
# ============================================================

def find_full_text(
    metadata
):

    files = metadata.get(
        "files",
        []
    )

    candidates = []

    for file_info in files:

        name = file_info.get(
            "name",
            ""
        )

        lower = name.lower()

        # Prefer DjVu full text
        if lower.endswith(
            "_djvu.txt"
        ):

            candidates.append(
                name
            )

    if not candidates:
        return None

    # Usually one result.
    # If multiple exist, choose shortest path/name.
    candidates.sort(
        key=len
    )

    return candidates[0]


# ============================================================
# DOWNLOAD FULL TEXT
# ============================================================

def download_full_text(
    identifier,
    filename
):

    # Archive.org download URL.
    # URL quoting is handled by requests.
    url = (
        f"https://archive.org/download/"
        f"{identifier}/"
        f"{filename}"
    )

    response = request_with_retry(
        url,
        timeout=120
    )

    # Some files contain a BOM.
    return (
        response.content
        .decode(
            "utf-8",
            errors="replace"
        )
    ), url


# ============================================================
# SHOULD SKIP ITEM?
# ============================================================

def should_skip_identifier(
    identifier
):

    lower = identifier.lower()

    # Huge amount of low-value government gazettes
    if lower.startswith(
        "in.gazette."
    ):
        return True

    # Other obvious non-book junk
    junk_patterns = [
        "metadata",
        "catalog",
        "sample",
    ]

    for pattern in junk_patterns:

        if pattern in lower:
            return True

    return False


# ============================================================
# PROCESS ONE ITEM
# ============================================================

def process_item(
    index,
    total,
    identifier,
    search_title,
    processed
):

    print()
    print("=" * 70)

    print(
        f"ITEM {index}/{total}"
    )

    print(
        f"Identifier: {identifier}"
    )

    print(
        f"Title     : {search_title}"
    )

    # --------------------------------------------------------
    # Already processed
    # --------------------------------------------------------

    if identifier in processed:

        print(
            "  -> ALREADY PROCESSED"
        )

        return 0


    # --------------------------------------------------------
    # Skip junk
    # --------------------------------------------------------

    if should_skip_identifier(
        identifier
    ):

        print(
            "  -> SKIPPED JUNK ITEM"
        )

        log_processed(
            identifier,
            search_title,
            0,
            0
        )

        processed.add(
            identifier
        )

        return 0


    try:

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = get_metadata(
            identifier
        )

        metadata_title = (
            metadata
            .get("metadata", {})
            .get("title", "")
        )

        title = (
            metadata_title
            or search_title
            or identifier
        )

        # ----------------------------------------------------
        # Find *_djvu.txt
        # ----------------------------------------------------

        filename = find_full_text(
            metadata
        )

        if not filename:

            print(
                "  -> NO _djvu.txt"
            )

            log_processed(
                identifier,
                title,
                0,
                0
            )

            processed.add(
                identifier
            )

            return 0

        print(
            f"  FULL TEXT: {filename}"
        )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        raw_text, text_url = (
            download_full_text(
                identifier,
                filename
            )
        )

        raw_words = len(
            raw_text.split()
        )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        cleaned = clean_text(
            raw_text
        )

        cleaned_words = (
            len(cleaned.split())
        )

        cleaned_chars = len(
            cleaned
        )

        print(
            f"  Raw words     : "
            f"{raw_words:,}"
        )

        print(
            # f"  Gujarati words: "
            f"  Assamese words: "
            f"{cleaned_words:,}"
        )

        # ----------------------------------------------------
        # Reject low-value item
        # ----------------------------------------------------

        if (
            cleaned_words
            < MIN_ASSAMESE_WORDS
        ):

            print(
                f"  -> SKIPPED "
                f"(below "
                f"{MIN_ASSAMESE_WORDS:,} "
                f"words)"
            )

            log_processed(
                identifier,
                title,
                cleaned_words,
                cleaned_chars
            )

            processed.add(
                identifier
            )

            return 0

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / safe_filename(
                identifier
            )
        )

        output_file.write_text(
            cleaned,
            encoding="utf-8"
        )

        print(
            f"  SAVED: "
            f"{output_file.name}"
        )

        log_processed(
            identifier,
            title,
            cleaned_words,
            cleaned_chars
        )

        log_stats(
            identifier,
            title,
            raw_words,
            cleaned_words,
            cleaned_chars,
            text_url
        )

        processed.add(
            identifier
        )

        return cleaned_words

    except Exception as e:

        print(
            f"  -> FAILED: {e}"
        )

        log_failed(
            identifier,
            e
        )

        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        # "ARCHIVE.ORG GUJARATI CORPUS COLLECTOR"
        "ARCHIVE.ORG ASSAMESE CORPUS COLLECTOR"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Existing progress
    # --------------------------------------------------------

    processed = load_processed()

    print(
        f"Previously processed: "
        f"{len(processed):,}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    items = get_identifiers()

    total_words = 0

    successful = 0

    for index, (
        identifier,
        search_title
    ) in enumerate(
        items,
        start=1
    ):

        before = len(
            processed
        )

        words = process_item(
            index,
            len(items),
            identifier,
            search_title,
            processed
        )

        if words > 0:

            total_words += words
            successful += 1

        # Small delay
        time.sleep(
            DELAY_BETWEEN_ITEMS
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARCHIVE.ORG COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Items considered : "
        f"{len(items):,}"
    )

    print(
        f"Successful books : "
        f"{successful:,}"
    )

    print(
        f"Words this run   : "
        f"{total_words:,}"
    )

    print(
        f"Output directory  : "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()
    print(
        "Progress files:"
    )

    print(
        f"  {PROCESSED_LOG}"
    )

    print(
        f"  {FAILED_LOG}"
    )

    print(
        f"  {STATS_LOG}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
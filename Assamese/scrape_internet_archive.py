# import requests
# from pathlib import Path
# import re
# import time
# import random
# import csv


# # ============================================================
# # SETTINGS
# # ============================================================

# API_URL = "https://archive.org/advancedsearch.php"
# METADATA_URL = "https://archive.org/metadata/{}"

# SEARCH_QUERY = (
#     '(language:asm OR language:"Assamese") '
#     'AND mediatype:texts'
# )

# # Start small.
# # Increase after you verify the pipeline.
# MAX_ITEMS = 20000

# BATCH_SIZE = 500

# MIN_ASSAMESE_WORDS = 2000

# MAX_RETRIES = 5

# REQUEST_TIMEOUT = 60

# DELAY_BETWEEN_ITEMS = 0.25

# DELAY_BETWEEN_BATCHES = 2.0

# # OUTPUT_DIR = Path(
# #     r"Gujarati\data\raw\manual\archive_org"
# # )
# OUTPUT_DIR = Path(
#     r"Assamese\data\raw\manual\archive_org"
# )

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )

# PROCESSED_LOG = (
#     OUTPUT_DIR / "processed.tsv"
# )

# FAILED_LOG = (
#     OUTPUT_DIR / "failures.tsv"
# )

# STATS_LOG = (
#     OUTPUT_DIR / "stats.tsv"
# )

# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 "
#         "(Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 "
#         "(KHTML, like Gecko) "
#         "Chrome/151.0 Safari/537.36"
#     )
# }

# session = requests.Session()

# session.headers.update(
#     HEADERS
# )


# # ============================================================
# # GUJARATI CHARACTER FUNCTIONS
# # ============================================================

# # def is_gujarati_char(ch):
# #     return "\u0A80" <= ch <= "\u0AFF"
# def is_assamese_char(ch):
#     return "\u0980" <= ch <= "\u09FF"

# # def gujarati_ratio(text):

# #     letters = [
# #         ch
# #         for ch in text
# #         if ch.isalpha()
# #     ]

# #     if not letters:
# #         return 0.0

# #     gujarati_count = sum(
# #         is_gujarati_char(ch)
# #         for ch in letters
# #     )

# #     return (
# #         gujarati_count
# #         / len(letters)
# #     )

# def assamese_ratio(text):

#     letters = [
#         ch for ch in text
#         if ch.isalpha()
#     ]

#     if not letters:
#         return 0.0

#     assamese_count = sum(
#         is_assamese_char(ch)
#         for ch in letters
#     )

#     return assamese_count / len(letters)

# # ============================================================
# # TEXT CLEANING
# # ============================================================

# def clean_text(text):

#     cleaned_lines = []

#     for line in text.splitlines():

#         line = line.strip()

#         if not line:
#             continue

#         # Remove obvious OCR garbage
#         line = re.sub(
#             r"\s+",
#             " ",
#             line
#         ).strip()

#         if not line:
#             continue

#         # ratio = gujarati_ratio(
#         #     line
#         # )
#         ratio = assamese_ratio(line)

#         # Entirely/non-substantially foreign lines
#         if ratio < 0.30:
#             continue

#         cleaned_lines.append(
#             line
#         )

#     return "\n".join(
#         cleaned_lines
#     )


# # ============================================================
# # SAFE FILENAME
# # ============================================================

# def safe_filename(identifier):

#     name = re.sub(
#         r'[<>:"/\\|?*]',
#         "_",
#         identifier
#     )

#     return name + ".txt"


# # ============================================================
# # LOG HELPERS
# # ============================================================

# def load_processed():

#     if not PROCESSED_LOG.exists():
#         return set()

#     processed = set()

#     with open(
#         PROCESSED_LOG,
#         "r",
#         encoding="utf-8"
#     ) as f:

#         for line in f:

#             line = line.strip()

#             if not line:
#                 continue

#             identifier = (
#                 line.split("\t", 1)[0]
#             )

#             processed.add(
#                 identifier
#             )

#     return processed


# def log_processed(
#     identifier,
#     title,
#     words,
#     chars
# ):

#     new_file = not PROCESSED_LOG.exists()

#     with open(
#         PROCESSED_LOG,
#         "a",
#         encoding="utf-8"
#     ) as f:

#         if new_file:
#             f.write(
#                 "identifier\ttitle\twords\tchars\n"
#             )

#         f.write(
#             f"{identifier}\t"
#             f"{title}\t"
#             f"{words}\t"
#             f"{chars}\n"
#         )


# def log_failed(
#     identifier,
#     error
# ):

#     new_file = not FAILED_LOG.exists()

#     with open(
#         FAILED_LOG,
#         "a",
#         encoding="utf-8"
#     ) as f:

#         if new_file:
#             f.write(
#                 "identifier\terror\n"
#             )

#         f.write(
#             f"{identifier}\t"
#             f"{repr(error)}\n"
#         )


# # def log_stats(
# #     identifier,
# #     title,
# #     raw_words,
# #     gujarati_words,
# #     chars,
# #     text_url
# # ):
# def log_stats(
#     identifier,
#     title,
#     raw_words,
#     assamese_words,
#     chars,
#     text_url
# ):

#     new_file = not STATS_LOG.exists()

#     with open(
#         STATS_LOG,
#         "a",
#         encoding="utf-8"
#     ) as f:

#         if new_file:

#             f.write(
#                 "identifier\t"
#                 "title\t"
#                 "raw_words\t"
#                 "assamese_words\t"
#                 "chars\t"
#                 "text_url\n"
#             )

#         f.write(
#             f"{identifier}\t"
#             f"{title}\t"
#             f"{raw_words}\t"
#             f"{assamese_words}\t"
#             f"{chars}\t"
#             f"{text_url}\n"
#         )
# # ============================================================
# # REQUEST WITH RETRY
# # ============================================================

# def request_with_retry(
#     url,
#     params=None,
#     timeout=REQUEST_TIMEOUT
# ):

#     last_error = None

#     for attempt in range(
#         1,
#         MAX_RETRIES + 1
#     ):

#         try:

#             response = session.get(
#                 url,
#                 params=params,
#                 timeout=timeout
#             )

#             response.raise_for_status()

#             return response

#         except requests.RequestException as e:

#             last_error = e

#             if attempt == MAX_RETRIES:
#                 break

#             wait = (
#                 2 ** (attempt - 1)
#                 + random.uniform(0, 1)
#             )

#             print(
#                 f"      Retry "
#                 f"{attempt}/{MAX_RETRIES - 1} "
#                 f"in {wait:.1f}s"
#             )

#             time.sleep(wait)

#     raise last_error


# # ============================================================
# # SEARCH ARCHIVE.ORG
# # ============================================================

# def get_identifiers():

#     print(
#         "Searching Archive.org..."
#     )

#     params = {
#         "q": SEARCH_QUERY,

#         "fl[]": [
#             "identifier",
#             "title"
#         ],

#         "rows": BATCH_SIZE,

#         "page": 1,

#         "output": "json"
#     }

#     response = request_with_retry(
#         API_URL,
#         params=params
#     )

#     data = response.json()

#     total_found = (
#         data["response"]["numFound"]
#     )

#     print(
#         f"Total matching items: "
#         f"{total_found:,}"
#     )

#     target = min(
#         total_found,
#         MAX_ITEMS
#     )

#     print(
#         f"Will process at most: "
#         f"{target:,}"
#     )

#     identifiers = []

#     page = 1

#     while len(identifiers) < target:

#         print(
#             f"Fetching search batch "
#             f"{page}..."
#         )

#         params = {

#             "q": SEARCH_QUERY,

#             "fl[]": [
#                 "identifier",
#                 "title"
#             ],

#             "rows": BATCH_SIZE,

#             "page": page,

#             "output": "json"
#         }

#         response = request_with_retry(
#             API_URL,
#             params=params
#         )

#         # data = response.json()

#         # docs = (
#         #     data["response"]["docs"]
#         # )

#         # if not docs:
#         #     break
#         try:
#             data = response.json()
#         except ValueError:
#             print(
#                 "  -> Archive.org returned "
#                 "non-JSON response. Stopping search."
#             )
#             break

#         if "response" not in data:
#             print(
#                 "  -> Archive.org returned an "
#                 "unexpected response:"
#             )
#             print(data)
#             break

#         docs = data["response"].get(
#             "docs",
#             []
#         )

#         if not docs:
#             break

#         for item in docs:

#             identifier = item.get(
#                 "identifier"
#             )

#             if identifier:

#                 identifiers.append(
#                     (
#                         identifier,
#                         item.get(
#                             "title",
#                             ""
#                         )
#                     )
#                 )

#                 if len(identifiers) >= target:
#                     break

#         page += 1

#         time.sleep(
#             DELAY_BETWEEN_BATCHES
#         )

#     # Deduplicate
#     seen = set()

#     unique = []

#     for identifier, title in identifiers:

#         if identifier in seen:
#             continue

#         seen.add(identifier)

#         unique.append(
#             (
#                 identifier,
#                 title
#             )
#         )

#     print(
#         f"Unique identifiers collected: "
#         f"{len(unique):,}"
#     )

#     return unique


# # ============================================================
# # GET METADATA
# # ============================================================

# def get_metadata(identifier):

#     url = METADATA_URL.format(
#         identifier
#     )

#     response = request_with_retry(
#         url
#     )

#     return response.json()


# # ============================================================
# # FIND FULL TEXT FILE
# # ============================================================

# def find_full_text(
#     metadata
# ):

#     files = metadata.get(
#         "files",
#         []
#     )

#     candidates = []

#     for file_info in files:

#         name = file_info.get(
#             "name",
#             ""
#         )

#         lower = name.lower()

#         # Prefer DjVu full text
#         if lower.endswith(
#             "_djvu.txt"
#         ):

#             candidates.append(
#                 name
#             )

#     if not candidates:
#         return None

#     # Usually one result.
#     # If multiple exist, choose shortest path/name.
#     candidates.sort(
#         key=len
#     )

#     return candidates[0]


# # ============================================================
# # DOWNLOAD FULL TEXT
# # ============================================================

# def download_full_text(
#     identifier,
#     filename
# ):

#     # Archive.org download URL.
#     # URL quoting is handled by requests.
#     url = (
#         f"https://archive.org/download/"
#         f"{identifier}/"
#         f"{filename}"
#     )

#     response = request_with_retry(
#         url,
#         timeout=120
#     )

#     # Some files contain a BOM.
#     return (
#         response.content
#         .decode(
#             "utf-8",
#             errors="replace"
#         )
#     ), url


# # ============================================================
# # SHOULD SKIP ITEM?
# # ============================================================

# def should_skip_identifier(
#     identifier
# ):

#     lower = identifier.lower()

#     # Huge amount of low-value government gazettes
#     if lower.startswith(
#         "in.gazette."
#     ):
#         return True

#     # Other obvious non-book junk
#     junk_patterns = [
#         "metadata",
#         "catalog",
#         "sample",
#     ]

#     for pattern in junk_patterns:

#         if pattern in lower:
#             return True

#     return False


# # ============================================================
# # PROCESS ONE ITEM
# # ============================================================

# def process_item(
#     index,
#     total,
#     identifier,
#     search_title,
#     processed
# ):

#     print()
#     print("=" * 70)

#     print(
#         f"ITEM {index}/{total}"
#     )

#     print(
#         f"Identifier: {identifier}"
#     )

#     print(
#         f"Title     : {search_title}"
#     )

#     # --------------------------------------------------------
#     # Already processed
#     # --------------------------------------------------------

#     if identifier in processed:

#         print(
#             "  -> ALREADY PROCESSED"
#         )

#         return 0


#     # --------------------------------------------------------
#     # Skip junk
#     # --------------------------------------------------------

#     if should_skip_identifier(
#         identifier
#     ):

#         print(
#             "  -> SKIPPED JUNK ITEM"
#         )

#         log_processed(
#             identifier,
#             search_title,
#             0,
#             0
#         )

#         processed.add(
#             identifier
#         )

#         return 0


#     try:

#         # ----------------------------------------------------
#         # Metadata
#         # ----------------------------------------------------

#         metadata = get_metadata(
#             identifier
#         )

#         metadata_title = (
#             metadata
#             .get("metadata", {})
#             .get("title", "")
#         )

#         title = (
#             metadata_title
#             or search_title
#             or identifier
#         )

#         # ----------------------------------------------------
#         # Find *_djvu.txt
#         # ----------------------------------------------------

#         filename = find_full_text(
#             metadata
#         )

#         if not filename:

#             print(
#                 "  -> NO _djvu.txt"
#             )

#             log_processed(
#                 identifier,
#                 title,
#                 0,
#                 0
#             )

#             processed.add(
#                 identifier
#             )

#             return 0

#         print(
#             f"  FULL TEXT: {filename}"
#         )

#         # ----------------------------------------------------
#         # Download
#         # ----------------------------------------------------

#         raw_text, text_url = (
#             download_full_text(
#                 identifier,
#                 filename
#             )
#         )

#         raw_words = len(
#             raw_text.split()
#         )

#         # ----------------------------------------------------
#         # Clean
#         # ----------------------------------------------------

#         cleaned = clean_text(
#             raw_text
#         )

#         cleaned_words = (
#             len(cleaned.split())
#         )

#         cleaned_chars = len(
#             cleaned
#         )

#         print(
#             f"  Raw words     : "
#             f"{raw_words:,}"
#         )

#         print(
#             # f"  Gujarati words: "
#             f"  Assamese words: "
#             f"{cleaned_words:,}"
#         )

#         # ----------------------------------------------------
#         # Reject low-value item
#         # ----------------------------------------------------

#         if (
#             cleaned_words
#             < MIN_ASSAMESE_WORDS
#         ):

#             print(
#                 f"  -> SKIPPED "
#                 f"(below "
#                 f"{MIN_ASSAMESE_WORDS:,} "
#                 f"words)"
#             )

#             log_processed(
#                 identifier,
#                 title,
#                 cleaned_words,
#                 cleaned_chars
#             )

#             processed.add(
#                 identifier
#             )

#             return 0

#         # ----------------------------------------------------
#         # Save
#         # ----------------------------------------------------

#         output_file = (
#             OUTPUT_DIR
#             / safe_filename(
#                 identifier
#             )
#         )

#         output_file.write_text(
#             cleaned,
#             encoding="utf-8"
#         )

#         print(
#             f"  SAVED: "
#             f"{output_file.name}"
#         )

#         log_processed(
#             identifier,
#             title,
#             cleaned_words,
#             cleaned_chars
#         )

#         log_stats(
#             identifier,
#             title,
#             raw_words,
#             cleaned_words,
#             cleaned_chars,
#             text_url
#         )

#         processed.add(
#             identifier
#         )

#         return cleaned_words

#     except Exception as e:

#         print(
#             f"  -> FAILED: {e}"
#         )

#         log_failed(
#             identifier,
#             e
#         )

#         return 0


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 70)
#     print(
#         # "ARCHIVE.ORG GUJARATI CORPUS COLLECTOR"
#         "ARCHIVE.ORG ASSAMESE CORPUS COLLECTOR"
#     )
#     print("=" * 70)

#     # --------------------------------------------------------
#     # Existing progress
#     # --------------------------------------------------------

#     processed = load_processed()

#     print(
#         f"Previously processed: "
#         f"{len(processed):,}"
#     )

#     # --------------------------------------------------------
#     # Search
#     # --------------------------------------------------------

#     items = get_identifiers()

#     total_words = 0

#     successful = 0

#     for index, (
#         identifier,
#         search_title
#     ) in enumerate(
#         items,
#         start=1
#     ):

#         before = len(
#             processed
#         )

#         words = process_item(
#             index,
#             len(items),
#             identifier,
#             search_title,
#             processed
#         )

#         if words > 0:

#             total_words += words
#             successful += 1

#         # Small delay
#         time.sleep(
#             DELAY_BETWEEN_ITEMS
#         )

#     # --------------------------------------------------------
#     # Final summary
#     # --------------------------------------------------------

#     print()
#     print("=" * 70)
#     print("ARCHIVE.ORG COLLECTION COMPLETE")
#     print("=" * 70)

#     print(
#         f"Items considered : "
#         f"{len(items):,}"
#     )

#     print(
#         f"Successful books : "
#         f"{successful:,}"
#     )

#     print(
#         f"Words this run   : "
#         f"{total_words:,}"
#     )

#     print(
#         f"Output directory  : "
#         f"{OUTPUT_DIR.resolve()}"
#     )

#     print()
#     print(
#         "Progress files:"
#     )

#     print(
#         f"  {PROCESSED_LOG}"
#     )

#     print(
#         f"  {FAILED_LOG}"
#     )

#     print(
#         f"  {STATS_LOG}"
#     )

#     print("=" * 70)


# if __name__ == "__main__":
#     main()
# # import requests
# # import re
# # import time
# # import random
# # from pathlib import Path
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # from threading import local
# # import json

# # # ============================================================
# # # SETTINGS
# # # ============================================================

# # API_URL = "https://archive.org/advancedsearch.php"
# # METADATA_URL = "https://archive.org/metadata/{}"

# # SEARCH_QUERY = (
# #     '(language:asm OR language:"Assamese") '
# #     'AND mediatype:texts'
# # )

# # MAX_ITEMS = 20000
# # BATCH_SIZE = 500

# # MIN_ASSAMESE_WORDS = 2000
# # MAX_RETRIES = 5
# # REQUEST_TIMEOUT = 60

# # # Main speed setting
# # WORKERS = 24


# # # ============================================================
# # # OUTPUT
# # # ============================================================

# # OUTPUT_DIR = Path(
# #     r"Assamese\data\raw\manual\archive_org"
# # )

# # OUTPUT_DIR.mkdir(
# #     parents=True,
# #     exist_ok=True
# # )

# # PROCESSED_LOG = OUTPUT_DIR / "processed.tsv"
# # FAILED_LOG = OUTPUT_DIR / "failures.tsv"
# # STATS_LOG = OUTPUT_DIR / "stats.tsv"


# # # ============================================================
# # # THREAD-LOCAL HTTP SESSION
# # # ============================================================

# # thread_local = local()

# # HEADERS = {
# #     "User-Agent": (
# #         "Mozilla/5.0 "
# #         "(Windows NT 10.0; Win64; x64) "
# #         "AppleWebKit/537.36 "
# #         "(KHTML, like Gecko) "
# #         "Chrome/151.0 Safari/537.36"
# #     )
# # }


# # def get_session():

# #     if not hasattr(thread_local, "session"):

# #         session = requests.Session()
# #         session.headers.update(HEADERS)

# #         thread_local.session = session

# #     return thread_local.session


# # # ============================================================
# # # ASSAMESE
# # # ============================================================

# # def is_assamese_char(ch):
# #     return "\u0980" <= ch <= "\u09FF"


# # def assamese_ratio(text):

# #     letters = [
# #         ch for ch in text
# #         if ch.isalpha()
# #     ]

# #     if not letters:
# #         return 0.0

# #     assamese_count = sum(
# #         is_assamese_char(ch)
# #         for ch in letters
# #     )

# #     return assamese_count / len(letters)


# # # ============================================================
# # # CLEANING
# # # ============================================================

# # def clean_text(text):

# #     cleaned_lines = []

# #     for line in text.splitlines():

# #         line = re.sub(
# #             r"\s+",
# #             " ",
# #             line.strip()
# #         )

# #         if not line:
# #             continue

# #         if assamese_ratio(line) < 0.30:
# #             continue

# #         cleaned_lines.append(line)

# #     return "\n".join(cleaned_lines)


# # # ============================================================
# # # FILENAME
# # # ============================================================

# # def safe_filename(identifier):

# #     name = re.sub(
# #         r'[<>:"/\\|?*]',
# #         "_",
# #         identifier
# #     )

# #     return name + ".txt"


# # # ============================================================
# # # PROCESSED LOG
# # # ============================================================

# # def load_processed():

# #     if not PROCESSED_LOG.exists():
# #         return set()

# #     processed = set()

# #     with open(
# #         PROCESSED_LOG,
# #         encoding="utf-8"
# #     ) as f:

# #         next(f, None)

# #         for line in f:

# #             identifier = line.split(
# #                 "\t",
# #                 1
# #             )[0]

# #             if identifier:
# #                 processed.add(identifier)

# #     return processed


# # # ============================================================
# # # REQUEST WITH RETRY
# # # ============================================================

# # def request_with_retry(
# #     url,
# #     params=None,
# #     timeout=REQUEST_TIMEOUT
# # ):

# #     last_error = None

# #     for attempt in range(
# #         1,
# #         MAX_RETRIES + 1
# #     ):

# #         try:

# #             response = get_session().get(
# #                 url,
# #                 params=params,
# #                 timeout=timeout
# #             )

# #             response.raise_for_status()

# #             return response

# #         except requests.RequestException as e:

# #             last_error = e

# #             if attempt == MAX_RETRIES:
# #                 break

# #             wait = (
# #                 2 ** (attempt - 1)
# #                 + random.uniform(0, 1)
# #             )

# #             time.sleep(wait)

# #     raise last_error


# # # ============================================================
# # # SEARCH ARCHIVE.ORG
# # # ============================================================

# # # def get_identifiers():

# # #     print("Searching Archive.org...")

# # #     first = request_with_retry(
# # #         API_URL,
# # #         params={
# # #             "q": SEARCH_QUERY,
# # #             "fl[]": ["identifier", "title"],
# # #             "rows": BATCH_SIZE,
# # #             "page": 1,
# # #             "output": "json"
# # #         }
# # #     ).json()

# # #     total_found = (
# # #         first["response"]["numFound"]
# # #     )

# # #     target = min(
# # #         total_found,
# # #         MAX_ITEMS
# # #     )

# # #     print(
# # #         f"Total matching items: {total_found:,}"
# # #     )

# # #     print(
# # #         f"Will process at most: {target:,}"
# # #     )

# # #     identifiers = []

# # #     page = 1

# # #     while len(identifiers) < target:

# # #         print(
# # #             f"Fetching search batch {page}..."
# # #         )

# # #         data = request_with_retry(
# # #             API_URL,
# # #             params={
# # #                 "q": SEARCH_QUERY,
# # #                 "fl[]": ["identifier", "title"],
# # #                 "rows": BATCH_SIZE,
# # #                 "page": page,
# # #                 "output": "json"
# # #             }
# # #         ).json()

# # #         docs = data["response"].get(
# # #             "docs",
# # #             []
# # #         )

# # #         if not docs:
# # #             break

# # #         for item in docs:

# # #             identifier = item.get(
# # #                 "identifier"
# # #             )

# # #             if identifier:

# # #                 identifiers.append(
# # #                     (
# # #                         identifier,
# # #                         item.get("title", "")
# # #                     )
# # #                 )

# # #                 if len(identifiers) >= target:
# # #                     break

# # #         page += 1

# # #         time.sleep(0.5)

# # #     # Deduplicate
# # #     seen = set()
# # #     unique = []

# # #     for identifier, title in identifiers:

# # #         if identifier in seen:
# # #             continue

# # #         seen.add(identifier)

# # #         unique.append(
# # #             (identifier, title)
# # #         )

# # #     print(
# # #         f"Unique identifiers: {len(unique):,}"
# # #     )

# # #     return unique


# # def get_identifiers():

# #     print("Searching Archive.org...")

# #     identifiers = []
# #     page = 1

# #     while len(identifiers) < MAX_ITEMS:

# #         print(f"Fetching search batch {page}...")

# #         params = {
# #             "q": SEARCH_QUERY,
# #             "fl[]": ["identifier", "title"],
# #             "rows": BATCH_SIZE,
# #             "page": page,
# #             "output": "json"
# #         }

# #         # Retry the SEARCH itself if Archive.org
# #         # gives a bad/rate-limited response
# #         data = None

# #         for attempt in range(1, MAX_RETRIES + 1):

# #             try:

# #                 response = request_with_retry(
# #                     API_URL,
# #                     params=params,
# #                     timeout=REQUEST_TIMEOUT
# #                 )

# #                 data = response.json()

# #                 if "response" in data:
# #                     break

# #                 print(
# #                     f"  Bad Archive.org response "
# #                     f"(attempt {attempt})"
# #                 )

# #             except Exception as e:

# #                 print(
# #                     f"  Search error "
# #                     f"(attempt {attempt}): {e}"
# #                 )

# #             time.sleep(
# #                 min(30, 2 ** attempt)
# #                 + random.uniform(0, 2)
# #             )

# #         if not data or "response" not in data:

# #             print(
# #                 "  Could not get valid response. "
# #                 "Stopping search."
# #             )

# #             break

# #         docs = data["response"].get(
# #             "docs",
# #             []
# #         )

# #         if not docs:
# #             break

# #         for item in docs:

# #             identifier = item.get(
# #                 "identifier"
# #             )

# #             if identifier:

# #                 identifiers.append(
# #                     (
# #                         identifier,
# #                         item.get("title", "")
# #                     )
# #                 )

# #                 if len(identifiers) >= MAX_ITEMS:
# #                     break

# #         page += 1

# #         # Don't hammer Archive.org
# #         time.sleep(1)

# #     # Deduplicate
# #     seen = set()
# #     unique = []

# #     for identifier, title in identifiers:

# #         if identifier in seen:
# #             continue

# #         seen.add(identifier)

# #         unique.append(
# #             (identifier, title)
# #         )

# #     print(
# #         f"Unique identifiers: "
# #         f"{len(unique):,}"
# #     )

# #     return unique
# # # ============================================================
# # # METADATA
# # # ============================================================

# # def get_metadata(identifier):

# #     return request_with_retry(
# #         METADATA_URL.format(identifier)
# #     ).json()


# # # ============================================================
# # # FIND FULL TEXT
# # # ============================================================

# # def find_full_text(metadata):

# #     candidates = []

# #     for file_info in metadata.get(
# #         "files",
# #         []
# #     ):

# #         name = file_info.get(
# #             "name",
# #             ""
# #         )

# #         if name.lower().endswith(
# #             "_djvu.txt"
# #         ):

# #             candidates.append(name)

# #     if not candidates:
# #         return None

# #     return min(
# #         candidates,
# #         key=len
# #     )


# # # ============================================================
# # # DOWNLOAD
# # # ============================================================

# # def download_full_text(
# #     identifier,
# #     filename
# # ):

# #     url = (
# #         f"https://archive.org/download/"
# #         f"{identifier}/"
# #         f"{filename}"
# #     )

# #     response = request_with_retry(
# #         url,
# #         timeout=120
# #     )

# #     return (
# #         response.content.decode(
# #             "utf-8",
# #             errors="replace"
# #         ),
# #         url
# #     )


# # # ============================================================
# # # SKIP JUNK
# # # ============================================================

# # def should_skip(identifier):

# #     lower = identifier.lower()

# #     if lower.startswith("in.gazette."):
# #         return True

# #     for pattern in (
# #         "metadata",
# #         "catalog",
# #         "sample"
# #     ):

# #         if pattern in lower:
# #             return True

# #     return False


# # # ============================================================
# # # PROCESS ONE BOOK
# # # ============================================================

# # def process_item(
# #     identifier,
# #     search_title
# # ):

# #     try:

# #         if should_skip(identifier):

# #             return {
# #                 "status": "skip",
# #                 "identifier": identifier,
# #                 "title": search_title,
# #                 "words": 0,
# #                 "chars": 0
# #             }

# #         # Metadata
# #         metadata = get_metadata(
# #             identifier
# #         )

# #         title = (
# #             metadata
# #             .get("metadata", {})
# #             .get("title", "")
# #             or search_title
# #             or identifier
# #         )

# #         # Find text
# #         filename = find_full_text(
# #             metadata
# #         )

# #         if not filename:

# #             return {
# #                 "status": "skip",
# #                 "identifier": identifier,
# #                 "title": title,
# #                 "words": 0,
# #                 "chars": 0
# #             }

# #         # Download
# #         raw_text, text_url = (
# #             download_full_text(
# #                 identifier,
# #                 filename
# #             )
# #         )

# #         raw_words = len(
# #             raw_text.split()
# #         )

# #         # Clean
# #         cleaned = clean_text(
# #             raw_text
# #         )

# #         words = len(
# #             cleaned.split()
# #         )

# #         chars = len(cleaned)

# #         # Too small
# #         if words < MIN_ASSAMESE_WORDS:

# #             return {
# #                 "status": "skip",
# #                 "identifier": identifier,
# #                 "title": title,
# #                 "words": words,
# #                 "chars": chars
# #             }

# #         # Save
# #         output_file = (
# #             OUTPUT_DIR
# #             / safe_filename(identifier)
# #         )

# #         output_file.write_text(
# #             cleaned,
# #             encoding="utf-8"
# #         )

# #         return {
# #             "status": "success",
# #             "identifier": identifier,
# #             "title": title,
# #             "words": words,
# #             "chars": chars,
# #             "raw_words": raw_words,
# #             "url": text_url
# #         }

# #     except Exception as e:

# #         return {
# #             "status": "failed",
# #             "identifier": identifier,
# #             "title": search_title,
# #             "error": repr(e)
# #         }


# # # ============================================================
# # # LOGGING
# # # ============================================================

# # def log_result(result):

# #     identifier = result["identifier"]
# #     title = result.get("title", "")

# #     if result["status"] in (
# #         "success",
# #         "skip"
# #     ):

# #         new_file = not PROCESSED_LOG.exists()

# #         with open(
# #             PROCESSED_LOG,
# #             "a",
# #             encoding="utf-8"
# #         ) as f:

# #             if new_file:
# #                 f.write(
# #                     "identifier\ttitle\twords\tchars\n"
# #                 )

# #             f.write(
# #                 f"{identifier}\t"
# #                 f"{title}\t"
# #                 f"{result.get('words', 0)}\t"
# #                 f"{result.get('chars', 0)}\n"
# #             )

# #     if result["status"] == "success":

# #         new_file = not STATS_LOG.exists()

# #         with open(
# #             STATS_LOG,
# #             "a",
# #             encoding="utf-8"
# #         ) as f:

# #             if new_file:

# #                 f.write(
# #                     "identifier\t"
# #                     "title\t"
# #                     "raw_words\t"
# #                     "assamese_words\t"
# #                     "chars\t"
# #                     "text_url\n"
# #                 )

# #             f.write(
# #                 f"{identifier}\t"
# #                 f"{title}\t"
# #                 f"{result['raw_words']}\t"
# #                 f"{result['words']}\t"
# #                 f"{result['chars']}\t"
# #                 f"{result['url']}\n"
# #             )

# #     if result["status"] == "failed":

# #         new_file = not FAILED_LOG.exists()

# #         with open(
# #             FAILED_LOG,
# #             "a",
# #             encoding="utf-8"
# #         ) as f:

# #             if new_file:
# #                 f.write(
# #                     "identifier\terror\n"
# #                 )

# #             f.write(
# #                 f"{identifier}\t"
# #                 f"{result['error']}\n"
# #             )


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main():

# #     print("=" * 70)
# #     print("ARCHIVE.ORG ASSAMESE CORPUS COLLECTOR")
# #     print("=" * 70)

# #     processed = load_processed()
# #     existing_files = {
# #         p.stem
# #         for p in OUTPUT_DIR.glob("*.txt")
# #     }

# #     print(
# #         f"Existing text files: "
# #         f"{len(existing_files):,}"
# #     )
# #     print(
# #         f"Previously processed: "
# #         f"{len(processed):,}"
# #     )

# #     items = get_identifiers()

# #     # Remove already processed
# #     # items = [
# #     #     item
# #     #     for item in items
# #     #     if item[0] not in processed
# #     # ]
# #     items = [
# #     item
# #     for item in items
# #     if (
# #         item[0] not in processed
# #         and item[0] not in existing_files
# #     )
# # ]

# #     print(
# #         f"Remaining items: "
# #         f"{len(items):,}"
# #     )

# #     total_words = 0
# #     successful = 0
# #     failed = 0

# #     start = time.time()

# #     # ========================================================
# #     # MULTITHREADED DOWNLOAD + PROCESSING
# #     # ========================================================

# #     with ThreadPoolExecutor(
# #         max_workers=WORKERS
# #     ) as executor:

# #         futures = {
# #             executor.submit(
# #                 process_item,
# #                 identifier,
# #                 title
# #             ): identifier

# #             for identifier, title in items
# #         }

# #         completed = 0

# #         for future in as_completed(
# #             futures
# #         ):

# #             result = future.result()

# #             completed += 1

# #             log_result(result)

# #             if result["status"] == "success":

# #                 successful += 1
# #                 total_words += result["words"]

# #             elif result["status"] == "failed":

# #                 failed += 1

# #             print(
# #                 f"[{completed:,}/{len(items):,}] "
# #                 f"{result['identifier']} | "
# #                 f"{result['status']} | "
# #                 f"{result.get('words', 0):,} words"
# #             )

# #     elapsed = time.time() - start

# #     print()
# #     print("=" * 70)
# #     print("COLLECTION COMPLETE")
# #     print("=" * 70)

# #     print(
# #         f"Processed : {len(items):,}"
# #     )

# #     print(
# #         f"Successful: {successful:,}"
# #     )

# #     print(
# #         f"Failed    : {failed:,}"
# #     )

# #     print(
# #         f"Words     : {total_words:,}"
# #     )

# #     print(
# #         f"Time      : {elapsed / 3600:.2f} hours"
# #     )

# #     print(
# #         f"Output    : {OUTPUT_DIR.resolve()}"
# #     )

# #     print("=" * 70)


# # if __name__ == "__main__":
# #     main()
import requests
import re
import time
import random

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local


# ============================================================
# SETTINGS
# ============================================================

# Archive.org Scraping API
SCRAPE_API_URL = (
    "https://archive.org/services/search/v1/scrape"
)

METADATA_URL = (
    "https://archive.org/metadata/{}"
)

SEARCH_QUERY = (
    '(language:asm OR language:"Assamese") '
    'AND mediatype:texts'
)

# Maximum number of items to collect
MAX_ITEMS = 20000

# Scraping API supports large counts.
# 10,000 is the maximum documented by current API implementations.
SCRAPE_COUNT = 10000

# Minimum Assamese words required
MIN_ASSAMESE_WORDS = 2000

# Request settings
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60

# Number of simultaneous metadata/text downloads
WORKERS = 24


# ============================================================
# OUTPUT
# ============================================================

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


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


thread_local = local()


def get_session():

    if not hasattr(
        thread_local,
        "session"
    ):

        session = requests.Session()

        session.headers.update(
            HEADERS
        )

        thread_local.session = session

    return thread_local.session


# ============================================================
# ASSAMESE CHARACTER FUNCTIONS
# ============================================================

def is_assamese_char(ch):

    return (
        "\u0980"
        <= ch
        <= "\u09FF"
    )


def assamese_ratio(text):

    letters = [
        ch
        for ch in text
        if ch.isalpha()
    ]

    if not letters:
        return 0.0

    assamese_count = sum(
        is_assamese_char(ch)
        for ch in letters
    )

    return (
        assamese_count
        / len(letters)
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Normalize whitespace
        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if not line:
            continue

        ratio = assamese_ratio(
            line
        )

        # Remove mostly foreign / garbage lines
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
# LOAD PROCESSED IDS
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
                line.split(
                    "\t",
                    1
                )[0]
            )

            # Skip header
            if identifier == "identifier":
                continue

            processed.add(
                identifier
            )

    return processed


# ============================================================
# LOGGING
# ============================================================

def log_processed(
    identifier,
    title,
    words,
    chars
):

    new_file = (
        not PROCESSED_LOG.exists()
    )

    with open(
        PROCESSED_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        if new_file:

            f.write(
                "identifier\t"
                "title\t"
                "words\t"
                "chars\n"
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

    new_file = (
        not FAILED_LOG.exists()
    )

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


def log_stats(
    identifier,
    title,
    raw_words,
    assamese_words,
    chars,
    text_url
):

    new_file = (
        not STATS_LOG.exists()
    )

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

            response = get_session().get(
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
                f"{attempt}/{MAX_RETRIES} "
                f"in {wait:.1f}s"
            )

            time.sleep(wait)

    raise last_error


# ============================================================
# ARCHIVE.ORG SCRAPING API
# ============================================================

def get_identifiers():

    print(
        "Searching Archive.org "
        "using Scraping API..."
    )

    identifiers = []

    cursor = None

    batch_number = 0

    total_found = None

    while len(identifiers) < MAX_ITEMS:

        batch_number += 1

        print(
            f"Fetching scrape batch "
            f"{batch_number}..."
        )

        params = {

            "q": SEARCH_QUERY,

            "fields": (
                "identifier,title"
            ),

            "count": SCRAPE_COUNT
        }

        # Cursor only exists after first request
        if cursor:

            params["cursor"] = cursor

        data = None

        # Retry malformed/API responses
        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                response = request_with_retry(
                    SCRAPE_API_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT
                )

                data = response.json()

                if "items" in data:

                    break

                print(
                    "  -> Unexpected "
                    "Scraping API response:"
                )

                print(data)

            except Exception as e:

                print(
                    f"  -> Scrape error "
                    f"attempt {attempt}: "
                    f"{e}"
                )

            if attempt < MAX_RETRIES:

                wait = (
                    5 * attempt
                    + random.uniform(1, 3)
                )

                print(
                    f"  Waiting "
                    f"{wait:.1f}s..."
                )

                time.sleep(
                    wait
                )

        if not data:

            print(
                "  -> Could not obtain "
                "valid scrape response."
            )

            break

        items = data.get(
            "items",
            []
        )

        if total_found is None:

            total_found = data.get(
                "total",
                0
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

        if not items:

            print(
                "  -> No more items."
            )

            break

        for item in items:

            identifier = item.get(
                "identifier"
            )

            if not identifier:
                continue

            identifiers.append(
                (
                    identifier,
                    item.get(
                        "title",
                        ""
                    )
                )
            )

            if len(identifiers) >= MAX_ITEMS:
                break

        print(
            f"  Received: "
            f"{len(items):,} items"
        )

        print(
            f"  Total collected: "
            f"{len(identifiers):,}"
        )

        # Get cursor for next batch
        cursor = data.get(
            "cursor"
        )

        # No cursor = finished
        if not cursor:

            print(
                "  -> Scraping API "
                "returned no cursor. "
                "Search complete."
            )

            break

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    seen = set()

    unique = []

    for identifier, title in identifiers:

        if identifier in seen:
            continue

        seen.add(
            identifier
        )

        unique.append(
            (
                identifier,
                title
            )
        )

    print(
        f"Unique identifiers "
        f"collected: "
        f"{len(unique):,}"
    )

    return unique


# ============================================================
# GET METADATA
# ============================================================

def get_metadata(
    identifier
):

    url = METADATA_URL.format(
        identifier
    )

    response = request_with_retry(
        url
    )

    return response.json()


# ============================================================
# FIND FULL TEXT
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

        if name.lower().endswith(
            "_djvu.txt"
        ):

            candidates.append(
                name
            )

    if not candidates:
        return None

    return min(
        candidates,
        key=len
    )


# ============================================================
# DOWNLOAD FULL TEXT
# ============================================================

def download_full_text(
    identifier,
    filename
):

    url = (
        f"https://archive.org/download/"
        f"{identifier}/"
        f"{filename}"
    )

    response = request_with_retry(
        url,
        timeout=120
    )

    text = response.content.decode(
        "utf-8",
        errors="replace"
    )

    return text, url


# ============================================================
# SHOULD SKIP ITEM
# ============================================================

def should_skip_identifier(
    identifier
):

    lower = identifier.lower()

    # Government gazettes
    if lower.startswith(
        "in.gazette."
    ):

        return True

    junk_patterns = [
        "metadata",
        "catalog",
        "sample"
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
    search_title
):

    result = {
        "status": "failed",
        "identifier": identifier,
        "title": search_title,
        "words": 0,
        "chars": 0
    }

    try:

        # ----------------------------------------------------
        # Skip junk
        # ----------------------------------------------------

        if should_skip_identifier(
            identifier
        ):

            result["status"] = "skip"

            return result


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

        result["title"] = title


        # ----------------------------------------------------
        # Find full text
        # ----------------------------------------------------

        filename = find_full_text(
            metadata
        )

        if not filename:

            result["status"] = "skip"

            return result


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

        cleaned_words = len(
            cleaned.split()
        )

        cleaned_chars = len(
            cleaned
        )


        # ----------------------------------------------------
        # Reject small documents
        # ----------------------------------------------------

        if (
            cleaned_words
            < MIN_ASSAMESE_WORDS
        ):

            result.update({
                "status": "skip",
                "words": cleaned_words,
                "chars": cleaned_chars
            })

            return result


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


        # ----------------------------------------------------
        # Return result
        # Main thread will log it
        # ----------------------------------------------------

        result.update({

            "status": "success",

            "words": cleaned_words,

            "chars": cleaned_chars,

            "raw_words": raw_words,

            "url": text_url
        })

        return result


    except Exception as e:

        result.update({

            "status": "failed",

            "error": repr(e)
        })

        return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "ARCHIVE.ORG ASSAMESE "
        "CORPUS COLLECTOR"
    )

    print("=" * 70)


    # ========================================================
    # EXISTING DATA
    # ========================================================

    processed = load_processed()

    existing_files = {
        p.stem
        for p in OUTPUT_DIR.glob(
            "*.txt"
        )
    }

    print(
        f"Existing text files: "
        f"{len(existing_files):,}"
    )

    print(
        f"Previously processed: "
        f"{len(processed):,}"
    )


    # ========================================================
    # SEARCH
    # ========================================================

    items = get_identifiers()


    # ========================================================
    # REMOVE ALREADY PROCESSED
    # ========================================================

    remaining_items = []

    for identifier, title in items:

        if identifier in processed:
            continue

        if identifier in existing_files:
            continue

        remaining_items.append(
            (
                identifier,
                title
            )
        )

    items = remaining_items

    print()

    print(
        f"Remaining items to process: "
        f"{len(items):,}"
    )

    print(
        f"Workers: {WORKERS}"
    )

    print()


    # ========================================================
    # MULTITHREADED PROCESSING
    # ========================================================

    total_words = 0

    successful = 0

    skipped = 0

    failed = 0

    completed = 0

    start = time.time()


    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {

            executor.submit(
                process_item,

                index,

                len(items),

                identifier,

                title

            ): identifier

            for index, (
                identifier,
                title
            )
            in enumerate(
                items,
                start=1
            )
        }


        for future in as_completed(
            futures
        ):

            result = future.result()

            completed += 1

            identifier = (
                result["identifier"]
            )

            title = result.get(
                "title",
                ""
            )

            status = result[
                "status"
            ]


            # =================================================
            # LOG IN MAIN THREAD
            # =================================================

            if status == "success":

                words = result[
                    "words"
                ]

                chars = result[
                    "chars"
                ]

                total_words += words

                successful += 1

                log_processed(
                    identifier,
                    title,
                    words,
                    chars
                )

                log_stats(
                    identifier,
                    title,
                    result[
                        "raw_words"
                    ],
                    words,
                    chars,
                    result["url"]
                )

                processed.add(
                    identifier
                )

                existing_files.add(
                    Path(
                        safe_filename(
                            identifier
                        )
                    ).stem
                )

            elif status == "skip":

                skipped += 1

                log_processed(
                    identifier,
                    title,
                    result.get(
                        "words",
                        0
                    ),
                    result.get(
                        "chars",
                        0
                    )
                )

                processed.add(
                    identifier
                )

            else:

                failed += 1

                log_failed(
                    identifier,
                    result.get(
                        "error",
                        "Unknown error"
                    )
                )


            # =================================================
            # PROGRESS
            # =================================================

            print(
                f"[{completed:,}/"
                f"{len(items):,}] "
                f"{identifier} | "
                f"{status} | "
                f"{result.get('words', 0):,} words"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = (
        time.time()
        - start
    )

    print()

    print("=" * 70)

    print(
        "ARCHIVE.ORG COLLECTION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Items processed : "
        f"{len(items):,}"
    )

    print(
        f"Successful      : "
        f"{successful:,}"
    )

    print(
        f"Skipped         : "
        f"{skipped:,}"
    )

    print(
        f"Failed          : "
        f"{failed:,}"
    )

    print(
        f"Words this run  : "
        f"{total_words:,}"
    )

    print(
        f"Time            : "
        f"{elapsed / 3600:.2f} hours"
    )

    if elapsed > 0:

        print(
            f"Speed           : "
            f"{successful / elapsed * 3600:.1f} "
            f"successful items/hour"
        )

    print()

    print(
        f"Output: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()

    print(
        "Progress logs:"
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
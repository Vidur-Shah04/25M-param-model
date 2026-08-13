# # import requests
# # from bs4 import BeautifulSoup
# # from urllib.parse import urljoin
# # from pathlib import Path
# # import time


# # # ============================================================
# # # INDEX PAGES
# # # ============================================================

# INDEX_URLS = [
#     "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A7/",
#     "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A8/",
#     "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A9/",
#     "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%AA/",
#     "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%AB/",
# ]

# # # Currently testing Part 1 only
# # INDEX_URLS = INDEX_URLS[:1]


# # # ============================================================
# # # OUTPUT
# # # ============================================================

# # OUTPUT_DIR = Path("Gujarati/data/raw/manual/vishwakosh")
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # # ============================================================
# # # REQUEST SETTINGS
# # # ============================================================

# # HEADERS = {
# #     "User-Agent": "Mozilla/5.0"
# # }


# # # ============================================================
# # # GET ARTICLE LINKS FROM INDEX PAGE
# # # ============================================================

# # def get_article_links(index_url):

# #     response = requests.get(
# #         index_url,
# #         headers=HEADERS,
# #         timeout=30
# #     )

# #     response.raise_for_status()

# #     soup = BeautifulSoup(
# #         response.text,
# #         "html.parser"
# #     )

# #     links = []

# #     # The "વધુ વાંચો" links point to the actual article pages
# #     for tag in soup.find_all(
# #         "a",
# #         class_="fl-post-grid-more"
# #     ):

# #         href = tag.get("href")

# #         if href:
# #             links.append(
# #                 urljoin(index_url, href)
# #             )

# #     return links


# # # ============================================================
# # # SCRAPE ONE ARTICLE
# # # ============================================================

# # def scrape_article(url):

# #     response = requests.get(
# #         url,
# #         headers=HEADERS,
# #         timeout=30
# #     )

# #     response.raise_for_status()

# #     soup = BeautifulSoup(
# #         response.text,
# #         "html.parser"
# #     )

# #     # There are multiple divs with this class.
# #     # From our inspection:
# #     #
# #     # DIV 0 -> title
# #     # DIV 1 -> date
# #     # DIV 2 -> navigation/breadcrumb
# #     # DIV 3 -> ACTUAL ARTICLE
# #     # DIV 4 -> tags/author metadata
# #     # DIV 5 -> empty
# #     # DIV 6 -> post navigation

# #     articles = soup.find_all(
# #         "div",
# #         class_="fl-module-content fl-node-content"
# #     )

# #     if len(articles) <= 3:
# #         return None

# #     # Actual article body
# #     article = articles[3]

# #     # Remove author/credit elements.
# #     # These have style="text-align: right;"
# #     for element in article.find_all(
# #         style=lambda value:
# #             value and "text-align: right" in value
# #     ):
# #         element.decompose()

# #     # Extract text while preserving paragraph boundaries
# #     text = article.get_text(
# #         "\n",
# #         strip=True
# #     )

# #     return text


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main():

# #     # --------------------------------------------------------
# #     # 1. Collect article URLs
# #     # --------------------------------------------------------

# #     article_urls = []

# #     for index_url in INDEX_URLS:

# #         print(f"Reading index: {index_url}")

# #         links = get_article_links(index_url)

# #         print(f"Found {len(links)} article links")

# #         article_urls.extend(links)


# #     print(
# #         f"\nTotal article URLs before deduplication: "
# #         f"{len(article_urls)}"
# #     )


# #     # --------------------------------------------------------
# #     # 2. Remove duplicate URLs
# #     # --------------------------------------------------------

# #     article_urls = list(
# #         dict.fromkeys(article_urls)
# #     )


# #     print(
# #         f"Total unique article URLs: "
# #         f"{len(article_urls)}"
# #     )


# #     # --------------------------------------------------------
# #     # 3. Scrape articles
# #     # --------------------------------------------------------

# #     successful = 0
# #     failed = 0

# #     for i, url in enumerate(
# #         article_urls,
# #         start=1
# #     ):

# #         print(
# #             f"[{i}/{len(article_urls)}] {url}"
# #         )

# #         try:

# #             text = scrape_article(url)

# #             if text is None:

# #                 print(
# #                     "  -> article body not found"
# #                 )

# #                 failed += 1
# #                 continue


# #             # ------------------------------------------------
# #             # Save article
# #             # ------------------------------------------------

# #             output_file = (
# #                 OUTPUT_DIR /
# #                 f"article_{i:06d}.txt"
# #             )

# #             output_file.write_text(
# #                 text,
# #                 encoding="utf-8"
# #             )


# #             print(
# #                 f"  -> saved "
# #                 f"({len(text):,} characters)"
# #             )

# #             successful += 1


# #             # Don't hammer the website
# #             time.sleep(0.2)


# #         except Exception as e:

# #             print(
# #                 f"  -> FAILED: {e}"
# #             )

# #             failed += 1


# #     # --------------------------------------------------------
# #     # 4. Summary
# #     # --------------------------------------------------------

# #     print("\nDone.")

# #     print(
# #         f"Successful: {successful}"
# #     )

# #     print(
# #         f"Failed:     {failed}"
# #     )

# #     print(
# #         f"Output directory: "
# #         f"{OUTPUT_DIR.resolve()}"
# #     )


# # # ============================================================
# # # RUN
# # # ============================================================

# # if __name__ == "__main__":
# #     main()
# import requests
# from bs4 import BeautifulSoup


# # INDEX_URL = (
# #     "https://gujarativishwakosh.org/"
# #     "%E0%AA%96%E0%AA%82%E0%AA%A1/"
# #     "%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A7/"
# # )

# HEADERS = {
#     "User-Agent": "Mozilla/5.0"
# }
# pagenum=0
# for INDEX_URL in INDEX_URLS:
#     pagenum+=1
#     article=[]
#     for page in range(1, 1000):

#         if page == 1:
#             url = INDEX_URL
#         else:
#             url = INDEX_URL.rstrip("/") + f"/page/{page}/"

#         response = requests.get(
#             url,
#             headers=HEADERS,
#             timeout=30
#         )

#         soup = BeautifulSoup(response.text, "html.parser")

#         links = soup.find_all(
#             "a",
#             class_="fl-post-grid-more"
#         )

#         # print(
#         #     f"Page {page}: "
#         #     f"status={response.status_code}, "
#         #     f"articles={len(links)}"
#         # )
#         # No articles = reached the end
#         article.append(len(links))
#         if response.status_code == 404 or not links:
#             break
#     print(f"page {pagenum} has {page} pages with {sum(article)} articles \n {article}\n ")

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# ============================================================
# VOLUMES
# ============================================================

INDEX_URLS = [
    "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A7/",
    "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A8/",
    "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A9/",
    "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%AA/",
    "https://gujarativishwakosh.org/%E0%AA%96%E0%AA%82%E0%AA%A1-%E0%AB%A5/",
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "Gujarati/data/raw/manual/vishwakosh"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

URL_FILE = OUTPUT_DIR / "article_urls.txt"

FAILED_FILE = OUTPUT_DIR / "failed_urls.txt"


# ============================================================
# SETTINGS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}

# Number of simultaneous article downloads
WORKERS = 8

# Maximum number of pagination pages
MAX_PAGES = 1000

# Retry failed requests
RETRIES = 3


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update(HEADERS)


# ============================================================
# GET ARTICLE LINKS FROM ONE INDEX PAGE
# ============================================================

def get_article_links(url):

    for attempt in range(RETRIES):

        try:

            response = session.get(
                url,
                timeout=30
            )

            if response.status_code == 404:
                return []

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            links = []

            for tag in soup.find_all(
                "a",
                class_="fl-post-grid-more"
            ):

                href = tag.get("href")

                if href:

                    links.append(
                        urljoin(url, href)
                    )

            return links

        except Exception as e:

            if attempt == RETRIES - 1:

                print(
                    f"FAILED INDEX: {url} | {e}"
                )

                return []

            time.sleep(1)


# ============================================================
# COLLECT ALL ARTICLE URLS
# ============================================================

def collect_urls():

    # If URL file already exists, don't download indexes again

    if URL_FILE.exists():

        urls = [
            x.strip()
            for x in URL_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

        print(
            f"Existing URL list found: {len(urls)} URLs"
        )

        return urls


    all_urls = []

    for volume_number, index_url in enumerate(
        INDEX_URLS,
        start=1
    ):

        print()
        print("=" * 60)
        print(f"VOLUME {volume_number}")
        print("=" * 60)

        volume_urls = []

        for page in range(1, MAX_PAGES + 1):

            if page == 1:

                url = index_url

            else:

                url = (
                    index_url.rstrip("/")
                    + f"/page/{page}/"
                )

            links = get_article_links(url)

            print(
                f"Volume {volume_number} | "
                f"Page {page} | "
                f"Articles: {len(links)}"
            )

            if not links:
                break

            volume_urls.extend(links)

        # Deduplicate this volume

        volume_urls = list(
            dict.fromkeys(volume_urls)
        )

        print(
            f"Volume {volume_number}: "
            f"{len(volume_urls)} unique articles"
        )

        all_urls.extend(volume_urls)


    # Deduplicate everything

    all_urls = list(
        dict.fromkeys(all_urls)
    )


    # Save URL list

    URL_FILE.write_text(
        "\n".join(all_urls),
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"TOTAL UNIQUE ARTICLES: {len(all_urls)}")
    print(f"Saved URL list to: {URL_FILE}")
    print("=" * 60)

    return all_urls


# ============================================================
# SCRAPE ONE ARTICLE
# ============================================================

def scrape_article(url):

    for attempt in range(RETRIES):

        try:

            response = session.get(
                url,
                timeout=30
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            articles = soup.find_all(
                "div",
                class_="fl-module-content fl-node-content"
            )

            if len(articles) <= 3:

                return None, "ARTICLE_BODY_NOT_FOUND"


            # Based on our inspection:
            # DIV 3 = actual article

            article = articles[3]


            # Remove author / credit elements

            for element in article.find_all(
                style=lambda value:
                    value and
                    "text-align: right" in value
            ):

                element.decompose()


            text = article.get_text(
                "\n",
                strip=True
            )


            if not text:

                return None, "EMPTY_ARTICLE"


            return text, None


        except Exception as e:

            if attempt == RETRIES - 1:

                return None, str(e)

            time.sleep(1)


# ============================================================
# SAVE ONE ARTICLE
# ============================================================

def process_article(args):

    number, url = args

    output_file = (
        OUTPUT_DIR /
        f"article_{number:06d}.txt"
    )

    # --------------------------------------------------------
    # CHECKPOINT
    #
    # If already downloaded, skip it.
    # --------------------------------------------------------

    if output_file.exists():

        return number, url, "SKIPPED", None


    text, error = scrape_article(url)


    if text is None:

        return number, url, "FAILED", error


    output_file.write_text(
        text,
        encoding="utf-8"
    )


    return number, url, "SAVED", len(text)


# ============================================================
# SCRAPE ALL ARTICLES
# ============================================================

def scrape_all(urls):

    print()
    print("=" * 60)
    print("STARTING ARTICLE SCRAPE")
    print("=" * 60)

    total = len(urls)

    successful = 0
    failed = 0
    skipped = 0

    failed_urls = []


    jobs = [
        (i, url)
        for i, url in enumerate(
            urls,
            start=1
        )
    ]


    # --------------------------------------------------------
    # Concurrent downloading
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_article,
                job
            )
            for job in jobs
        ]


        for completed, future in enumerate(
            as_completed(futures),
            start=1
        ):

            number, url, status, info = (
                future.result()
            )


            if status == "SAVED":

                successful += 1

                print(
                    f"[{completed}/{total}] "
                    f"SAVED article_{number:06d}.txt "
                    f"({info:,} chars)"
                )


            elif status == "SKIPPED":

                skipped += 1

                print(
                    f"[{completed}/{total}] "
                    f"SKIP article_{number:06d}.txt"
                )


            else:

                failed += 1

                failed_urls.append(url)

                print(
                    f"[{completed}/{total}] "
                    f"FAILED article_{number:06d} "
                    f"| {info}"
                )


    # --------------------------------------------------------
    # Save failed URLs
    # --------------------------------------------------------

    if failed_urls:

        FAILED_FILE.write_text(
            "\n".join(failed_urls),
            encoding="utf-8"
        )


    print()
    print("=" * 60)
    print("SCRAPE FINISHED")
    print("=" * 60)

    print(
        f"Successful : {successful}"
    )

    print(
        f"Skipped    : {skipped}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"Output     : {OUTPUT_DIR.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("STEP 1: Collecting article URLs")

    urls = collect_urls()


    print()
    print(
        f"STEP 2: Scraping {len(urls)} articles"
    )


    scrape_all(urls)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
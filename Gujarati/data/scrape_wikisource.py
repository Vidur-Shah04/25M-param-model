import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from pathlib import Path
import re
import time


# ============================================================
# SETTINGS
# ============================================================

BOOK_INDEX_URL = (
    # "https://gu.wikisource.org/wiki/"
    # "%E0%AA%B5%E0%AA%BF%E0%AA%95%E0%AA%BF%E0%AA%B8%E0%AB%8D%E0%AA%B0%E0%AB%8B%E0%AA%A4%3A%E0%AA%AA%E0%AC%81%E0%AA%B8%E0%AB%8D%E0%AA%A4%E0%AA%95%E0%AB%8B"
    "https://gu.wikisource.org/wiki/%E0%AA%B5%E0%AA%BF%E0%AA%95%E0%AA%BF%E0%AA%B8%E0%AB%8D%E0%AA%B0%E0%AB%8B%E0%AA%A4%3A%E0%AA%AA%E0%AB%81%E0%AA%B8%E0%AB%8D%E0%AA%A4%E0%AA%95%E0%AB%8B"
    )

# IMPORTANT:
# If the encoded URL above gives an error, use your working
# BOOK_INDEX_URL from the previous script.

OUTPUT_DIR = Path(
    r"Gujarati\data\raw\manual\wikisource"
)

URL_LOG = OUTPUT_DIR / "books_and_chapters.txt"
FAILED_LOG = OUTPUT_DIR / "failed.txt"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)

DELAY = 0.01


# ============================================================
# HELPERS
# ============================================================

def clean_filename(name):

    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip(" .")

# ============================================================
# GET BOOK URLS
# ============================================================

def get_book_urls():

    response = session.get(
        BOOK_INDEX_URL,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    table = soup.find(
        "table",
        class_="wikitable sortable jquery-tablesorter"
    )

    if table is None:

        table = soup.find(
            "table",
            class_="wikitable"
        )

    if table is None:

        raise RuntimeError(
            "Could not find Wikisource book table"
        )

    book_urls = []

    for row in table.find_all("tr"):

        cells = row.find_all("td")

        if len(cells) < 2:
            continue

        link = cells[1].find(
            "a",
            href=True
        )

        if link is None:
            continue

        href = link["href"]

        if not href.startswith("/wiki/"):
            continue

        decoded_href = unquote(href)

        # Ignore creator/category/project pages
        if (
            decoded_href.startswith("/wiki/સર્જક:")
            or decoded_href.startswith("/wiki/શ્રેણી:")
            or decoded_href.startswith("/wiki/વિકિસ્રોત:")
        ):
            continue

        book_urls.append(
            urljoin(
                BOOK_INDEX_URL,
                href
            )
        )

    return list(
        dict.fromkeys(book_urls)
    )


# ============================================================
# GET CHAPTER / SUBPAGE / PROOFREAD PAGE URLS
# ============================================================

def get_chapter_urls(book_url):

    response = session.get(
        book_url,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    html = response.text

    # ========================================================
    # CASE 1
    # PROOFREADPAGE BOOK
    # ========================================================

    # Example:
    #
    # "prpSourceIndexPage":"સૂચિ:Rachanatmak Karyakram.pdf"
    #
    # These books have their actual text in:
    #
    # /wiki/પૃષ્ઠ:Rachanatmak_Karyakram.pdf/૧
    # /wiki/પૃષ્ઠ:Rachanatmak_Karyakram.pdf/૨
    # etc.

    match = re.search(
        r'"prpSourceIndexPage":"([^"]+)"',
        html
    )

    if match:

        index_title = match.group(1)

        print(
            f"  Proofread source index found: "
            f"{index_title}"
        )

        index_title = index_title.replace(
            " ",
            "_"
        )

        index_url = (
            "https://gu.wikisource.org/wiki/"
            + index_title
        )

        try:

            index_response = session.get(
                index_url,
                timeout=30
            )

            index_response.raise_for_status()

            index_soup = BeautifulSoup(
                index_response.text,
                "html.parser"
            )

            page_urls = []

            for link in index_soup.find_all(
                "a",
                href=True
            ):

                href = link["href"]

                # Decode URL before checking namespace
                decoded_href = unquote(
                    href
                )

                # Remove fragment
                if "#" in decoded_href:

                    decoded_href = (
                        decoded_href.split(
                            "#",
                            1
                        )[0]
                    )

                # Actual ProofreadPage pages
                if decoded_href.startswith(
                    "/wiki/પૃષ્ઠ:"
                ):

                    page_urls.append(
                        urljoin(
                            index_url,
                            href.split("#")[0]
                        )
                    )

            page_urls = list(
                dict.fromkeys(
                    page_urls
                )
            )

            if page_urls:

                print(
                    f"  Found "
                    f"{len(page_urls)} "
                    f"ProofreadPage URLs"
                )

                return page_urls

        except Exception as e:

            print(
                f"  Proofread index failed: "
                f"{e}"
            )


    # ========================================================
    # CASE 2
    # NORMAL WIKISOURCE BOOK WITH SUBPAGES
    # ========================================================

    parsed = urlparse(
        book_url
    )

    book_path = unquote(
        parsed.path
    )

    if not book_path.startswith(
        "/wiki/"
    ):

        return []


    prefix = (
        book_path.rstrip("/")
        + "/"
    )

    content = soup.select_one(
        "#mw-content-text .mw-parser-output"
    )

    if content is None:

        content = soup.select_one(
            "#mw-content-text"
        )

    if content is None:

        return []


    chapter_urls = []


    for link in content.find_all(
        "a",
        href=True
    ):

        href = link["href"]

        # Ignore external URLs
        if href.startswith(
            "http"
        ):

            continue


        # Remove fragments
        if "#" in href:

            href = href.split(
                "#",
                1
            )[0]


        # Only wiki pages
        if not href.startswith(
            "/wiki/"
        ):

            continue


        decoded_href = unquote(
            href
        )


        # ----------------------------------------------------
        # Ignore special namespaces
        # ----------------------------------------------------

        ignored_namespaces = [
            "/wiki/વિશેષ:",
            "/wiki/શ્રેણી:",
            "/wiki/વિકિસ્રોત:",
            "/wiki/સદસ્ય:",
            "/wiki/સંચિકા:",
            "/wiki/પૃષ્ઠ:",
            "/wiki/સૂચિ:",
        ]

        if any(
            decoded_href.startswith(
                namespace
            )
            for namespace in ignored_namespaces
        ):

            continue


        # ----------------------------------------------------
        # Only subpages belonging to this book
        # ----------------------------------------------------

        if decoded_href.startswith(
            prefix
        ):

            full_url = urljoin(
                book_url,
                href
            )

            chapter_urls.append(
                full_url
            )


    return list(
        dict.fromkeys(
            chapter_urls
        )
    )


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_page_text(url):

    response = session.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    content = soup.select_one(
        "#mw-content-text .mw-parser-output"
    )

    if content is None:

        content = soup.select_one(
            "#mw-content-text"
        )

    if content is None:

        raise RuntimeError(
            "Main content not found"
        )


    # ========================================================
    # REMOVE UI / NAVIGATION
    # ========================================================

    selectors_to_remove = [

        ".mw-editsection",

        ".navbox",
        ".vertical-navbox",

        ".metadata",

        ".ambox",
        ".tmbox",
        ".messagebox",

        ".toc",

        ".printfooter",
        ".catlinks",

        ".reference",
        ".mw-references-wrap",

        ".portal",
        ".plainlinks",

        # ProofreadPage UI
        ".prp-page-quality",
        ".prp-pageheader",
        ".prp-pagefooter",
        ".prp-page-image",
        ".prp-pagequality",
    ]


    for selector in selectors_to_remove:

        for element in content.select(
            selector
        ):

            element.decompose()


    # ========================================================
    # REMOVE REFERENCES / FOOTNOTES
    # ========================================================

    for element in content.find_all(
        "sup"
    ):

        element.decompose()


    # ========================================================
    # REMOVE IMAGES / SCRIPTS / STYLES
    # ========================================================

    for element in content.find_all(
        [
            "img",
            "script",
            "style"
        ]
    ):

        element.decompose()


    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    text = content.get_text(
        "\n",
        strip=True
    )


    # ========================================================
    # NORMALIZE WHITESPACE
    # ========================================================

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if line:

            lines.append(
                line
            )


    return "\n".join(
        lines
    )


# ============================================================
# SCRAPE ONE BOOK
# ============================================================

def scrape_book(
    book_number,
    book_url
):

    # --------------------------------------------------------
    # Get readable title
    # --------------------------------------------------------

    title = unquote(
        urlparse(book_url)
        .path
        .replace(
            "/wiki/",
            "",
            1
        )
    )

    title = title.replace(
        "_",
        " "
    )

    safe_title = clean_filename(
        title
    )

    output_file = (
        OUTPUT_DIR
        / f"{book_number:04d}_{safe_title}.txt"
    )


    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"[{book_number}] SKIP "
            f"(already exists): {title}"
        )

        return {

            "book": title,

            "chapters": 0,

            "words": len(
                output_file.read_text(
                    encoding="utf-8"
                ).split()
            ),

            "status": "skipped"
        }


    print()
    print("=" * 70)

    print(
        f"BOOK {book_number}: "
        f"{title}"
    )

    print(book_url)

    print("=" * 70)


    # --------------------------------------------------------
    # Find chapters / ProofreadPage pages
    # --------------------------------------------------------

    chapter_urls = get_chapter_urls(
        book_url
    )


    # --------------------------------------------------------
    # No subpages
    # --------------------------------------------------------

    if not chapter_urls:

        chapter_urls = [
            book_url
        ]


    print(
        f"Chapter/page URLs: "
        f"{len(chapter_urls)}"
    )


    all_text = []

    successful = 0


    # --------------------------------------------------------
    # Scrape
    # --------------------------------------------------------

    for i, chapter_url in enumerate(
        chapter_urls,
        start=1
    ):

        print(
            f"  [{i}/{len(chapter_urls)}] "
            f"{chapter_url}"
        )

        try:

            text = extract_page_text(
                chapter_url
            )

            if text:

                all_text.append(
                    text
                )

                successful += 1

        except Exception as e:

            print(
                f"      FAILED: {e}"
            )

            with open(
                FAILED_LOG,
                "a",
                encoding="utf-8"
            ) as failed:

                failed.write(
                    f"{title}\t"
                    f"{chapter_url}\t"
                    f"{repr(e)}\n"
                )


        time.sleep(
            DELAY
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_text = "\n\n".join(
        all_text
    )


    if not final_text.strip():

        print(
            "  -> NO TEXT EXTRACTED"
        )

        return {

            "book": title,

            "chapters": len(
                chapter_urls
            ),

            "words": 0,

            "status": "failed"
        }


    output_file.write_text(
        final_text,
        encoding="utf-8"
    )


    word_count = len(
        final_text.split()
    )


    print(
        f"  -> SAVED "
        f"{word_count:,} words"
    )


    return {

        "book": title,

        "chapters": successful,

        "words": word_count,

        "status": "saved"
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Getting Gujarati Wikisource books..."
    )


    # books = get_book_urls()
    books = get_book_urls()

    # books = [
    #     books[0]
    # ]


    print(
        f"Found {len(books)} books"
    )


    # --------------------------------------------------------
    # Save book URLs
    # --------------------------------------------------------

    with open(
        URL_LOG,
        "w",
        encoding="utf-8"
    ) as f:

        for i, url in enumerate(
            books,
            start=1
        ):

            f.write(
                f"{i}\t{url}\n"
            )


    # --------------------------------------------------------
    # Scrape books
    # --------------------------------------------------------

    total_words = 0

    saved_books = 0

    failed_books = 0


    for i, book_url in enumerate(
        books,
        start=1
    ):

        try:

            result = scrape_book(
                i,
                book_url
            )


            total_words += result[
                "words"
            ]


            if result["status"] in {
                "saved",
                "skipped"
            }:

                saved_books += 1

            else:

                failed_books += 1


        except Exception as e:

            failed_books += 1


            print(
                f"BOOK FAILED: "
                f"{book_url}"
            )


            with open(
                FAILED_LOG,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"BOOK\t"
                    f"{book_url}\t"
                    f"{repr(e)}\n"
                )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "WIKISOURCE SCRAPE COMPLETE"
    )

    print("=" * 70)

    print(
        f"Books found    : "
        f"{len(books):,}"
    )

    print(
        f"Books processed: "
        f"{saved_books:,}"
    )

    print(
        f"Books failed   : "
        f"{failed_books:,}"
    )

    print(
        f"Words collected: "
        f"{total_words:,}"
    )

    print(
        f"Output         : "
        f"{OUTPUT_DIR.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
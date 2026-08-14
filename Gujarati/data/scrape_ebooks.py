# from zipfile import ZipFile
# from bs4 import BeautifulSoup
# from pathlib import Path


# # EPUB_PATH = Path("book.epub")
# EPUB_PATH = Path(r"Gujarati\data\raw\manual\ebooks\આકાશદીપ.epub")

# # OUTPUT_PATH = Path("book.txt")
# OUTPUT_PATH = Path(r"Gujarati\data\raw\manual\ebooks\આકાશદીપ.txt")


# with ZipFile(EPUB_PATH, "r") as epub:

#     html_files = [
#         name
#         for name in epub.namelist()
#         if name.lower().endswith((".html", ".xhtml", ".htm"))
#     ]

#     # Keep only actual chapters
#     chapter_files = [
#         name
#         for name in html_files
#         if "/c" in name.lower()
#     ]

#     print(f"Chapter files: {len(chapter_files)}")

#     all_text = []

#     for i, filename in enumerate(chapter_files, start=1):

#         print(f"[{i}/{len(chapter_files)}] {filename}")

#         html = epub.read(filename)

#         soup = BeautifulSoup(
#             html,
#             "html.parser"
#         )

#         # Remove non-content elements
#         for tag in soup(["script", "style", "nav"]):
#             tag.decompose()

#         text = soup.get_text(
#             "\n",
#             strip=True
#         )

#         if text:
#             all_text.append(text)


# # Save corpus
# OUTPUT_PATH.write_text(
#     "\n\n".join(all_text),
#     encoding="utf-8"
# )


# # Stats
# words = OUTPUT_PATH.read_text(
#     encoding="utf-8"
# ).split()

# characters = OUTPUT_PATH.stat().st_size

# print("\n" + "=" * 50)
# print("DONE")
# print("=" * 50)
# print(f"Chapters : {len(chapter_files)}")
# print(f"Words    : {len(words):,}")
# print(f"Characters: {characters:,}")
# print(f"Output   : {OUTPUT_PATH.resolve()}")
from zipfile import ZipFile
from bs4 import BeautifulSoup
from pathlib import Path


# ============================================================
# FOLDER
# ============================================================

EPUB_DIR = Path(
    r"Gujarati\data\raw\manual\ebooks"
)


# ============================================================
# FIND ALL EPUBS
# ============================================================

epub_files = sorted(
    EPUB_DIR.glob("*.epub")
)

print(f"Found {len(epub_files)} EPUB files.\n")


# ============================================================
# PROCESS EVERY EPUB
# ============================================================

for book_num, EPUB_PATH in enumerate(
    epub_files,
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"BOOK {book_num}/{len(epub_files)}")
    print(f"{EPUB_PATH.name}")
    print("=" * 70)


    # --------------------------------------------------------
    # Open EPUB
    # --------------------------------------------------------

    with ZipFile(EPUB_PATH, "r") as epub:

        html_files = [
            name
            for name in epub.namelist()
            if name.lower().endswith(
                (".html", ".xhtml", ".htm")
            )
        ]


        # ----------------------------------------------------
        # Show numbered HTML files
        # ----------------------------------------------------

        print("\nHTML/XHTML files:\n")

        for i, filename in enumerate(
            html_files,
            start=1
        ):
            print(f"{i}. {filename}")


        # ----------------------------------------------------
        # Ask range
        # ----------------------------------------------------

        while True:

            try:

                start_index = int(
                    input("\nEnter START index: ")
                )

                end_index = int(
                    input("Enter END index: ")
                )

                if (
                    start_index < 1
                    or end_index > len(html_files)
                    or start_index > end_index
                ):
                    print(
                        "Invalid range. Try again."
                    )
                    continue

                break

            except ValueError:

                print(
                    "Please enter integers."
                )


        # ----------------------------------------------------
        # Select files
        # 1-indexed + inclusive
        # ----------------------------------------------------

        selected_files = html_files[
            start_index - 1 : end_index
        ]


        print("\nSelected files:\n")

        for i, filename in enumerate(
            selected_files,
            start=start_index
        ):
            print(f"{i}. {filename}")


        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        all_text = []

        for i, filename in enumerate(
            selected_files,
            start=1
        ):

            print(
                f"Extracting "
                f"{i}/{len(selected_files)}: "
                f"{filename}"
            )

            html = epub.read(filename)

            soup = BeautifulSoup(
                html,
                "html.parser"
            )


            # Remove non-content elements
            for tag in soup(
                ["script", "style", "nav"]
            ):
                tag.decompose()


            text = soup.get_text(
                "\n",
                strip=True
            )


            if text:
                all_text.append(text)


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH = EPUB_PATH.with_suffix(".txt")

    OUTPUT_PATH.write_text(
        "\n\n".join(all_text),
        encoding="utf-8"
    )


    # ========================================================
    # STATS
    # ========================================================

    words = OUTPUT_PATH.read_text(
        encoding="utf-8"
    ).split()

    characters = OUTPUT_PATH.stat().st_size


    print("\n" + "-" * 60)
    print("SAVED")
    print("-" * 60)

    print(
        f"Book       : {EPUB_PATH.name}"
    )

    print(
        f"Files used : {len(selected_files)}"
    )

    print(
        f"Words      : {len(words):,}"
    )

    print(
        f"Characters : {characters:,}"
    )

    print(
        f"Output     : {OUTPUT_PATH.resolve()}"
    )


print("\n" + "=" * 70)
print("ALL EPUBS PROCESSED")
print("=" * 70)
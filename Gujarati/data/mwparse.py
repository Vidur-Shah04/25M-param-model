import bz2
import xml.etree.ElementTree as ET
import mwparserfromhell
from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

XML_PATH = Path(
    r"Gujarati\data\raw\manual\wiki dumps\guwiki-20260101-pages-articles.xml.bz2"
)

OUTPUT_DIR = Path(
    r"Gujarati\data\raw\manual\wiki dumps\gujarati_wikipedia"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLEAN WIKITEXT
# ============================================================

# def clean_wikitext(wikitext):

#     code = mwparserfromhell.parse(wikitext)

#     # --------------------------------------------------------
#     # Remove templates
#     # --------------------------------------------------------

#     for template in code.filter_templates():
#         code.remove(template)

#     # --------------------------------------------------------
#     # Remove unwanted HTML tags
#     # --------------------------------------------------------

#     for tag in code.filter_tags():

#         name = str(tag.name).lower().strip()

#         if name in {
#             "ref",
#             "references",
#             "gallery",
#             "timeline",
#             "math",
#             "table"
#         }:
#             code.remove(tag)

#     # --------------------------------------------------------
#     # Convert MediaWiki markup to plain text
#     # --------------------------------------------------------

#     text = code.strip_code(
#         normalize=True,
#         collapse=True
#     )

#     # --------------------------------------------------------
#     # Remove URLs
#     # --------------------------------------------------------

#     text = re.sub(
#         r"https?://\S+",
#         " ",
#         text
#     )

#     # --------------------------------------------------------
#     # Normalize whitespace
#     # --------------------------------------------------------

#     text = re.sub(
#         r"\s+",
#         " ",
#         text
#     )

#     return text.strip()

def clean_wikitext(wikitext):

    # ========================================================
    # 1. Remove templates
    # ========================================================

    code = mwparserfromhell.parse(wikitext)

    for template in list(code.filter_templates(recursive=True)):
        try:
            code.replace(template, "")
        except ValueError:
            pass

    text = str(code)

    # ========================================================
    # 2. Remove references and other non-text blocks
    # ========================================================

    text = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"<ref\b[^>]*/>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"<gallery\b[^>]*>.*?</gallery\s*>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"<timeline\b[^>]*>.*?</timeline\s*>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # ========================================================
    # 3. Remove wiki tables
    # ========================================================

    text = re.sub(
        r"\{\|.*?\|\}",
        " ",
        text,
        flags=re.DOTALL
    )

    # ========================================================
    # 4. Parse again
    # ========================================================

    code = mwparserfromhell.parse(text)

    # ========================================================
    # 5. Convert wiki markup to plain text
    # ========================================================

    text = code.strip_code(
        normalize=True,
        collapse=True
    )

    # ========================================================
    # 6. Remove URLs
    # ========================================================

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # ========================================================
    # 7. Normalize whitespace
    # ========================================================

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
# ============================================================
# MAIN
# ============================================================

def main():

    article_count = 0
    skipped = 0

    print("Starting Gujarati Wikipedia extraction...")
    print(f"Input : {XML_PATH.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print()

    # --------------------------------------------------------
    # Open compressed XML
    # --------------------------------------------------------

    with bz2.open(
        XML_PATH,
        "rb"
    ) as xml_file:

        # ----------------------------------------------------
        # Stream XML
        # ----------------------------------------------------

        for event, elem in ET.iterparse(
            xml_file,
            events=("end",)
        ):

            # ------------------------------------------------
            # Only process <page>
            # ------------------------------------------------

            if not elem.tag.endswith("page"):
                continue

            title = ""
            namespace = None
            text = ""
            redirect = False

            # ------------------------------------------------
            # Read page information
            # ------------------------------------------------

            for child in elem:

                tag = child.tag.split("}")[-1]

                if tag == "title":

                    title = child.text or ""

                elif tag == "ns":

                    namespace = child.text

                elif tag == "redirect":

                    redirect = True

                elif tag == "revision":

                    for revision_child in child:

                        revision_tag = (
                            revision_child.tag
                            .split("}")[-1]
                        )

                        if revision_tag == "text":

                            text = (
                                revision_child.text
                                or ""
                            )

            # ------------------------------------------------
            # Only namespace 0
            # ------------------------------------------------

            if namespace != "0":

                skipped += 1
                elem.clear()
                continue

            # ------------------------------------------------
            # Skip redirects
            # ------------------------------------------------

            if redirect or not text:

                skipped += 1
                elem.clear()
                continue

            # ------------------------------------------------
            # Clean article
            # ------------------------------------------------

            cleaned = clean_wikitext(text)

            if not cleaned:

                skipped += 1
                elem.clear()
                continue

            # ------------------------------------------------
            # Save article
            # ------------------------------------------------

            article_count += 1

            output_file = (
                OUTPUT_DIR /
                f"article_{article_count:06d}.txt"
            )

            output_file.write_text(
                cleaned,
                encoding="utf-8"
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if article_count % 100 == 0:

                print(
                    f"Articles processed: "
                    f"{article_count:,}"
                )

            # ------------------------------------------------
            # Free memory
            # ------------------------------------------------

            elem.clear()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Articles extracted : "
        f"{article_count:,}"
    )

    print(
        f"Pages skipped      : "
        f"{skipped:,}"
    )

    print(
        f"Output directory   : "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
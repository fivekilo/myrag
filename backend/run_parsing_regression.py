import json
from pathlib import Path

import fitz

from services.parsing_service import ParsingService


def build_page_map(pdf_path: Path) -> tuple[str, list[dict]]:
    text_blocks = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                text_blocks.append({"text": text.strip(), "page": page_num})
    return "\n".join(block["text"] for block in text_blocks), text_blocks


def run() -> int:
    workspace_dir = Path(__file__).resolve().parent.parent
    sample_dir = workspace_dir / "sample_papers_2026"
    parser = ParsingService()
    failures = []
    results = []

    for pdf_path in sorted(sample_dir.glob("*.pdf")):
        raw_text, page_map = build_page_map(pdf_path)
        parsed = parser.parse_pdf(
            raw_text,
            "structured_blocks",
            {
                "filename": pdf_path.name,
                "loading_method": "pymupdf",
            },
            page_map=page_map,
            file_path=pdf_path,
        )

        blocks = parsed.get("blocks", [])
        diagnostics = parsed.get("diagnostics", {})
        title_texts = [block["text"] for block in blocks if block["type"] == "title"]
        paragraph_texts = [block["text"] for block in blocks if block["type"] == "paragraph"]
        caption_count = sum(1 for block in blocks if block["type"] == "caption")
        table_count = sum(1 for block in blocks if block["type"] == "table")
        has_abstract = "Abstract" in title_texts
        has_references = "References" in title_texts
        intro_index = next(
            (idx for idx, block in enumerate(blocks) if block["type"] == "title" and "Introduction" in block["text"]),
            None,
        )
        intro_followed_by_paragraph = (
            intro_index is not None
            and intro_index + 1 < len(blocks)
            and blocks[intro_index + 1]["type"] in {"paragraph", "abstract_body"}
        )
        header_noise_in_paragraphs = any(
            "Association for Computational Linguistics" in text for text in paragraph_texts
        )

        paper_result = {
            "file": pdf_path.name,
            "total_blocks": len(blocks),
            "title_count": len(title_texts),
            "table_count": table_count,
            "caption_count": caption_count,
            "reference_start_page": diagnostics.get("reference_start_page"),
            "header_footer_candidates": len(diagnostics.get("header_footer_candidates", [])),
            "has_abstract": has_abstract,
            "has_references": has_references,
            "intro_followed_by_paragraph": intro_followed_by_paragraph,
            "section_path_examples": [
                block.get("section_path", [])
                for block in blocks
                if block["type"] in {"title", "paragraph"} and block.get("section_path")
            ][:5],
        }
        results.append(paper_result)

        if not has_abstract:
            failures.append(f"{pdf_path.name}: missing Abstract title")
        if not has_references:
            failures.append(f"{pdf_path.name}: missing References title")
        if intro_index is not None and not intro_followed_by_paragraph:
            failures.append(f"{pdf_path.name}: introduction is not followed by paragraph-like content")
        if header_noise_in_paragraphs:
            failures.append(f"{pdf_path.name}: repeated ACL header leaked into paragraph blocks")

    total_caption_or_table_hits = sum(
        1 for result in results if (result["caption_count"] + result["table_count"]) > 0
    )
    if total_caption_or_table_hits < 2:
        failures.append("Expected at least two papers to contain caption or table blocks")

    long_paper = next(
        (result for result in results if "nlp_for_social_good_survey" in result["file"]),
        None,
    )
    if long_paper and not any(len(path) >= 1 for path in long_paper["section_path_examples"]):
        failures.append("Long survey paper did not retain section paths")

    print(json.dumps({"results": results, "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())

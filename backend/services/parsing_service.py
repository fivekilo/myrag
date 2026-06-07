import logging
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency in current workspace
    pdfplumber = None

logger = logging.getLogger(__name__)


class ParsingService:
    """PDF parsing service with both legacy and structured parsing modes."""

    TITLE_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+")
    CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s+\d+[A-Za-z]?\s*:", re.IGNORECASE)
    REFERENCE_ITEM_RE = re.compile(r"^\[?\d+\]?[\.\)]?\s+")

    def parse_pdf(
        self,
        text: str,
        method: str,
        metadata: dict,
        page_map: list | None = None,
        file_path: str | Path | None = None,
    ) -> dict:
        """Parse a PDF document with the specified method."""
        try:
            if not page_map:
                raise ValueError("Page map is required for parsing.")

            total_pages = len(page_map)

            if method == "all_text":
                parsed_content = self._parse_all_text(page_map)
                return {
                    "metadata": {
                        "filename": metadata.get("filename", ""),
                        "total_pages": total_pages,
                        "parsing_method": method,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "content": parsed_content,
                }
            if method == "by_pages":
                parsed_content = self._parse_by_pages(page_map)
                return {
                    "metadata": {
                        "filename": metadata.get("filename", ""),
                        "total_pages": total_pages,
                        "parsing_method": method,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "content": parsed_content,
                }
            if method == "by_titles":
                parsed_content = self._parse_by_titles(page_map)
                return {
                    "metadata": {
                        "filename": metadata.get("filename", ""),
                        "total_pages": total_pages,
                        "parsing_method": method,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "content": parsed_content,
                }
            if method == "text_and_tables":
                parsed_content = self._parse_text_and_tables(page_map)
                return {
                    "metadata": {
                        "filename": metadata.get("filename", ""),
                        "total_pages": total_pages,
                        "parsing_method": method,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "content": parsed_content,
                }
            if method == "structured_blocks":
                if not file_path:
                    raise ValueError("File path is required for structured parsing.")
                return self._parse_structured_blocks(
                    file_path=file_path,
                    metadata=metadata,
                    total_pages=total_pages,
                )

            raise ValueError(f"Unsupported parsing method: {method}")
        except Exception as e:
            logger.error(f"Error in parse_pdf: {str(e)}")
            raise

    def _parse_all_text(self, page_map: list) -> list:
        return [
            {"type": "Text", "content": page["text"], "page": page["page"]}
            for page in page_map
        ]

    def _parse_by_pages(self, page_map: list) -> list:
        parsed_content = []
        for page in page_map:
            parsed_content.append(
                {"type": "Page", "page": page["page"], "content": page["text"]}
            )
        return parsed_content

    def _parse_by_titles(self, page_map: list) -> list:
        parsed_content = []
        current_title = None
        current_content = []
        current_page = 1

        for page in page_map:
            lines = page["text"].split("\n")
            for line in lines:
                if len(line.strip()) < 60 and line.isupper():
                    if current_title:
                        parsed_content.append(
                            {
                                "type": "section",
                                "title": current_title,
                                "content": "\n".join(current_content),
                                "page": current_page,
                            }
                        )
                    current_title = line.strip()
                    current_content = []
                    current_page = page["page"]
                else:
                    current_content.append(line)

        if current_title:
            parsed_content.append(
                {
                    "type": "section",
                    "title": current_title,
                    "content": "\n".join(current_content),
                    "page": current_page,
                }
            )

        return parsed_content

    def _parse_text_and_tables(self, page_map: list) -> list:
        parsed_content = []
        for page in page_map:
            content = page["text"]
            if "|" in content or "\t" in content:
                parsed_content.append(
                    {"type": "table", "content": content, "page": page["page"]}
                )
            else:
                parsed_content.append(
                    {"type": "text", "content": content, "page": page["page"]}
                )
        return parsed_content

    def _parse_structured_blocks(
        self, file_path: str | Path, metadata: dict, total_pages: int
    ) -> dict:
        file_path = Path(file_path)
        layout_pages = self._extract_layout_pages(file_path)
        repeated_noise = self._detect_repeated_noise(layout_pages)
        table_candidates = self._extract_table_candidates(file_path)
        blocks, diagnostics = self._build_semantic_blocks(
            layout_pages=layout_pages,
            repeated_noise=repeated_noise,
            table_candidates=table_candidates,
        )

        return {
            "metadata": {
                "filename": metadata.get("filename", ""),
                "file_type": file_path.suffix.lstrip(".").lower(),
                "total_pages": total_pages,
                "loading_method": metadata.get("loading_method", ""),
                "parsing_method": "structured_blocks",
                "timestamp": datetime.now().isoformat(),
            },
            "blocks": blocks,
            "diagnostics": diagnostics,
        }

    def _extract_layout_pages(self, file_path: Path) -> list[dict]:
        layout_pages = []
        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc, 1):
                page_dict = page.get_text("dict")
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
                text_blocks = []

                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue

                    lines = []
                    font_sizes = []
                    is_bold = False
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            if span_text:
                                line_text += span_text
                            size = span.get("size")
                            if size:
                                font_sizes.append(float(size))
                            font_name = str(span.get("font", "")).lower()
                            if "bold" in font_name:
                                is_bold = True
                        clean_line = " ".join(line_text.split())
                        if clean_line:
                            lines.append(clean_line)

                    text = "\n".join(lines).strip()
                    if not text:
                        continue

                    bbox = [float(v) for v in block.get("bbox", (0, 0, 0, 0))]
                    text_blocks.append(
                        {
                            "page": page_number,
                            "text": text,
                            "bbox": bbox,
                            "font_size": (
                                round(statistics.median(font_sizes), 2)
                                if font_sizes
                                else 0.0
                            ),
                            "is_bold": is_bold,
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                        }
                    )

                body_font_size = self._estimate_body_font_size(text_blocks)
                columns = self._assign_columns(text_blocks, page_width, page_number)
                layout_pages.append(
                    {
                        "page": page_number,
                        "width": page_width,
                        "height": page_height,
                        "body_font_size": body_font_size,
                        "blocks": columns,
                    }
                )
        return layout_pages

    def _estimate_body_font_size(self, text_blocks: list[dict]) -> float:
        sizes = [
            block["font_size"]
            for block in text_blocks
            if block["font_size"] > 0 and len(block["text"]) > 40
        ]
        if not sizes:
            sizes = [block["font_size"] for block in text_blocks if block["font_size"] > 0]
        return round(statistics.median(sizes), 2) if sizes else 10.0

    def _assign_columns(
        self, text_blocks: list[dict], page_width: float, page_number: int
    ) -> list[dict]:
        if not text_blocks:
            return []

        column_candidates = [
            block["bbox"][0]
            for block in text_blocks
            if block["width"] < page_width * 0.75
        ]
        column_count = 1
        if len(column_candidates) >= 4:
            sorted_x = sorted(column_candidates)
            if sorted_x[-1] - sorted_x[0] > page_width * 0.22:
                column_count = 2

        processed = []
        for block in text_blocks:
            is_full_width = block["width"] >= page_width * 0.72
            if self._is_first_page_centered_front_matter(block, page_width, page_number):
                is_full_width = True
            if page_number == 1 and block["bbox"][1] < 220:
                is_full_width = True
            if column_count == 2 and not is_full_width:
                midpoint = block["bbox"][0] + block["width"] / 2
                column_index = 0 if midpoint < (page_width / 2) else 1
            else:
                column_index = 0

            block_copy = {**block}
            block_copy["is_full_width"] = is_full_width
            block_copy["column_index"] = column_index
            processed.append(block_copy)

        if column_count == 2:
            full_width_blocks = [block for block in processed if block["is_full_width"]]
            column_blocks = [block for block in processed if not block["is_full_width"]]
            ordered = sorted(full_width_blocks, key=lambda item: item["bbox"][1])
            ordered.extend(
                sorted(
                    column_blocks,
                    key=lambda item: (item["column_index"], item["bbox"][1], item["bbox"][0]),
                )
            )
        else:
            ordered = sorted(processed, key=lambda item: (item["bbox"][1], item["bbox"][0]))

        for index, block in enumerate(ordered, 1):
            block["layout_index"] = index
            block["page"] = page_number

        return ordered

    def _is_first_page_centered_front_matter(
        self, block: dict, page_width: float, page_number: int
    ) -> bool:
        if page_number != 1:
            return False
        if block["bbox"][1] > 220:
            return False
        center = block["bbox"][0] + block["width"] / 2
        centered = abs(center - (page_width / 2)) <= page_width * 0.12
        return centered and block["width"] >= page_width * 0.45

    def _detect_repeated_noise(self, layout_pages: list[dict]) -> set[tuple[int, str]]:
        total_pages = len(layout_pages)
        if total_pages < 2:
            return set()

        occurrences: Counter[str] = Counter()
        positions: dict[str, list[tuple[int, float, float]]] = {}

        for page in layout_pages:
            for block in page["blocks"]:
                normalized = self._normalize_repetition_text(block["text"])
                if not normalized:
                    continue
                top_ratio = block["bbox"][1] / page["height"]
                bottom_ratio = block["bbox"][3] / page["height"]
                if top_ratio <= 0.12 or bottom_ratio >= 0.92:
                    occurrences[normalized] += 1
                    positions.setdefault(normalized, []).append(
                        (page["page"], top_ratio, bottom_ratio)
                    )

        repeated_noise = set()
        threshold = max(2, int(total_pages * 0.7))
        for normalized, count in occurrences.items():
            if count < threshold:
                continue
            for page_num, top_ratio, bottom_ratio in positions[normalized]:
                if top_ratio <= 0.12 or bottom_ratio >= 0.92:
                    repeated_noise.add((page_num, normalized))
        return repeated_noise

    def _is_acl_edge_noise(self, text: str, bbox: list[float], page_height: float) -> bool:
        lowered = text.lower()
        near_edge = (bbox[1] / page_height) <= 0.12 or (bbox[3] / page_height) >= 0.92
        if not near_edge:
            return False
        acl_markers = [
            "association for computational linguistics",
            "conference of the european chapter",
            "findings of the association for computational linguistics",
            "proceedings of the",
        ]
        return any(marker in lowered for marker in acl_markers)

    def _normalize_repetition_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        normalized = re.sub(r"\bpages?\s+\d+[–-]\d+\b", "pages", normalized, flags=re.I)
        normalized = re.sub(r"\b\d+\b", "#", normalized)
        return normalized.strip().lower()

    def _extract_table_candidates(self, file_path: Path) -> dict[int, list[dict]]:
        candidates: dict[int, list[dict]] = {}

        try:
            with fitz.open(file_path) as doc:
                for page_number, page in enumerate(doc, 1):
                    if not hasattr(page, "find_tables"):
                        continue
                    try:
                        table_finder = page.find_tables()
                    except Exception as exc:
                        logger.debug("PyMuPDF table detection failed on page %s: %s", page_number, exc)
                        continue

                    tables = getattr(table_finder, "tables", [])
                    page_tables = []
                    for idx, table in enumerate(tables, 1):
                        rows = table.extract() if hasattr(table, "extract") else []
                        cleaned_rows = [
                            [cell.strip() if isinstance(cell, str) else cell for cell in row]
                            for row in rows
                            if row and any(cell not in (None, "") for cell in row)
                        ]
                        bbox = getattr(table, "bbox", None)
                        page_tables.append(
                            {
                                "table_id": f"p{page_number}_t{idx}",
                                "rows": cleaned_rows,
                                "bbox": [float(v) for v in bbox] if bbox else None,
                            }
                        )
                    if page_tables:
                        candidates[page_number] = page_tables
        except Exception as exc:
            logger.warning("PyMuPDF table extraction failed: %s", exc)

        if candidates:
            return candidates

        if pdfplumber is None:
            return candidates

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables() or []
                    page_tables = []
                    for idx, rows in enumerate(tables, 1):
                        cleaned_rows = [
                            [cell.strip() if isinstance(cell, str) else cell for cell in row]
                            for row in rows
                            if row and any(cell not in (None, "") for cell in row)
                        ]
                        if cleaned_rows:
                            page_tables.append(
                                {
                                    "table_id": f"p{page_number}_t{idx}",
                                    "rows": cleaned_rows,
                                    "bbox": None,
                                }
                            )
                    if page_tables:
                        candidates[page_number] = page_tables
        except Exception as exc:
            logger.warning("pdfplumber table extraction failed: %s", exc)

        return candidates

    def _build_semantic_blocks(
        self,
        layout_pages: list[dict],
        repeated_noise: set[tuple[int, str]],
        table_candidates: dict[int, list[dict]],
    ) -> tuple[list[dict], dict]:
        blocks = []
        current_section_path: list[str] = []
        reference_mode = False
        reference_start_page = None
        in_abstract = False
        diagnostics_header_footer = []
        reference_index = 0

        for page in layout_pages:
            page_tables = list(table_candidates.get(page["page"], []))
            block_index = 0
            while block_index < len(page["blocks"]):
                layout_block = page["blocks"][block_index]
                normalized = self._normalize_repetition_text(layout_block["text"])
                if (page["page"], normalized) in repeated_noise or self._is_acl_edge_noise(
                    layout_block["text"], layout_block["bbox"], page["height"]
                ):
                    diagnostics_header_footer.append(
                        {
                            "page": page["page"],
                            "text": layout_block["text"],
                            "bbox": layout_block["bbox"],
                        }
                    )
                    block_index += 1
                    continue

                semantic_type = self._classify_block(
                    block=layout_block,
                    body_font_size=page["body_font_size"],
                    first_page=(page["page"] == 1),
                    reference_mode=reference_mode,
                    page_height=page["height"],
                )

                if semantic_type == "author_metadata":
                    merged_blocks = [layout_block]
                    next_index = block_index + 1
                    while next_index < len(page["blocks"]):
                        next_block = page["blocks"][next_index]
                        next_normalized = self._normalize_repetition_text(next_block["text"])
                        if (page["page"], next_normalized) in repeated_noise or self._is_acl_edge_noise(
                            next_block["text"], next_block["bbox"], page["height"]
                        ):
                            break
                        next_type = self._classify_block(
                            block=next_block,
                            body_font_size=page["body_font_size"],
                            first_page=(page["page"] == 1),
                            reference_mode=reference_mode,
                            page_height=page["height"],
                        )
                        if next_type != "author_metadata":
                            break
                        merged_blocks.append(next_block)
                        next_index += 1

                    merged_block = self._merge_blocks(merged_blocks)
                    blocks.append(
                        self._make_block(
                            layout_block=merged_block,
                            block_type="author_metadata",
                            text=merged_block["text"],
                            section_path=[],
                            extra={"column_index": 0},
                        )
                    )
                    block_index = next_index
                    continue

                if semantic_type == "title":
                    title_text = self._normalize_heading_text(layout_block["text"])
                    level = self._infer_title_level(title_text)

                    if title_text == "Abstract":
                        in_abstract = True
                        block = self._make_block(
                            layout_block=layout_block,
                            block_type="title",
                            text=title_text,
                            section_path=current_section_path.copy(),
                            extra={"level": 0},
                        )
                        blocks.append(block)
                        block_index += 1
                        continue

                    if title_text == "References":
                        reference_mode = True
                        reference_start_page = page["page"]
                        in_abstract = False
                        current_section_path = ["References"]
                        block = self._make_block(
                            layout_block=layout_block,
                            block_type="title",
                            text=title_text,
                            section_path=current_section_path.copy(),
                            extra={"level": 0},
                        )
                        blocks.append(block)
                        block_index += 1
                        continue

                    if self._is_document_title(layout_block, page["body_font_size"], page["page"]):
                        blocks.append(
                            self._make_block(
                                layout_block=layout_block,
                                block_type="title",
                                text=title_text,
                                section_path=[],
                                extra={"level": -1, "title_role": "document_title"},
                            )
                        )
                        block_index += 1
                        continue

                    in_abstract = False
                    current_section_path = self._update_section_path(
                        current_section_path, title_text, level
                    )
                    block = self._make_block(
                        layout_block=layout_block,
                        block_type="title",
                        text=title_text,
                        section_path=current_section_path.copy(),
                        extra={"level": level},
                    )
                    blocks.append(block)
                    block_index += 1
                    continue

                if semantic_type == "caption":
                    caption_text = self._normalize_heading_text(layout_block["text"])
                    caption_kind = "table" if caption_text.lower().startswith("table") else "figure"
                    target_id = page_tables[0]["table_id"] if caption_kind == "table" and page_tables else None
                    blocks.append(
                        self._make_block(
                            layout_block=layout_block,
                            block_type="caption",
                            text=caption_text,
                            section_path=current_section_path.copy(),
                            extra={
                                "caption_kind": caption_kind,
                                "target_id": target_id,
                            },
                        )
                    )
                    block_index += 1
                    continue

                if semantic_type == "footnote":
                    blocks.append(
                        self._make_block(
                            layout_block=layout_block,
                            block_type="footnote",
                            text=layout_block["text"],
                            section_path=current_section_path.copy() if current_section_path else [],
                            extra={"column_index": layout_block["column_index"]},
                        )
                    )
                    block_index += 1
                    continue

                if reference_mode:
                    reference_index += 1
                    blocks.append(
                        self._make_block(
                            layout_block=layout_block,
                            block_type="reference_item",
                            text=layout_block["text"],
                            section_path=current_section_path.copy(),
                            extra={"reference_index": reference_index},
                        )
                    )
                    block_index += 1
                    continue

                if page_tables and self._looks_like_table_region(layout_block, page_tables):
                    table = page_tables.pop(0)
                    blocks.append(
                        self._make_table_block(
                            layout_block=layout_block,
                            table=table,
                            section_path=current_section_path.copy(),
                        )
                    )
                    block_index += 1
                    continue

                block_type = "abstract_body" if in_abstract else "paragraph"
                blocks.append(
                    self._make_block(
                        layout_block=layout_block,
                        block_type=block_type,
                        text=layout_block["text"],
                        section_path=current_section_path.copy(),
                        extra={"column_index": layout_block["column_index"]},
                    )
                )
                block_index += 1

            while page_tables and not reference_mode:
                table = page_tables.pop(0)
                if not self._should_emit_table(table):
                    continue
                blocks.append(
                    {
                        "block_id": table["table_id"],
                        "page": page["page"],
                        "type": "table",
                        "text": self._table_to_text(table["rows"]),
                        "bbox": table["bbox"] or [0.0, 0.0, 0.0, 0.0],
                        "section_path": current_section_path.copy(),
                        "source_method": "pymupdf",
                        "table_id": table["table_id"],
                        "table_caption": None,
                        "rows": table["rows"],
                    }
                )

        diagnostics = {
            "header_footer_candidates": diagnostics_header_footer,
            "table_count": sum(1 for block in blocks if block["type"] == "table"),
            "reference_start_page": reference_start_page,
        }
        return blocks, diagnostics

    def _classify_block(
        self,
        block: dict,
        body_font_size: float,
        first_page: bool,
        reference_mode: bool,
        page_height: float,
    ) -> str:
        text = block["text"].strip()
        if not text:
            return "paragraph"

        if text in {"Abstract", "References"}:
            return "title"

        if reference_mode:
            return "reference_item"

        if self._is_author_metadata(text, block, first_page):
            return "author_metadata"

        if self._is_footnote_block(text, block, body_font_size, page_height):
            return "footnote"

        if self._is_formula_or_algorithm_block(text):
            return "paragraph"

        if self._is_axis_or_legend_label(text):
            return "paragraph"

        if self.CAPTION_RE.match(text) and len(text) <= 240:
            return "caption"

        is_numbered_title = bool(self.TITLE_NUMBER_RE.match(text))
        is_prominent = block["font_size"] >= (body_font_size + 1.5)
        short_enough = len(text) <= 120
        looks_title_like = text[:1].isalnum() and not text.endswith(".")

        if short_enough and (is_numbered_title or (is_prominent and (block["is_bold"] or looks_title_like))):
            return "title"

        if self.REFERENCE_ITEM_RE.match(text):
            return "reference_item"

        return "paragraph"

    def _is_author_metadata(self, text: str, block: dict, first_page: bool) -> bool:
        if not first_page:
            return False
        if block["bbox"][1] > 320:
            return False
        lowered = text.lower()
        if any(
            token in lowered
            for token in ["@", "university", "institute", "department", "school", "laboratories"]
        ):
            return True
        short_lines = len(text.split()) <= 12
        looks_like_name_line = bool(re.search(r"\b[a-z]+\.[a-z]+", lowered))
        has_author_markers = bool(re.search(r"\b\d+(?:,\d+)*\b", text))
        likely_author_line = sum(1 for line in text.splitlines() if line.strip()) <= 3
        return short_lines and likely_author_line and (looks_like_name_line or has_author_markers)

    def _is_footnote_block(
        self, text: str, block: dict, body_font_size: float, page_height: float
    ) -> bool:
        lowered = text.lower()
        near_bottom = (block["bbox"][3] / page_height) >= 0.82
        small_font = block["font_size"] <= max(body_font_size - 1.5, 8.0)
        footnote_markers = any(
            marker in lowered
            for marker in [
                "http://",
                "https://",
                "correspondence:",
                "∗correspondence:",
                "*correspondence:",
            ]
        )
        numbered_source_lines = bool(
            re.search(r"(^|\n)[\d*∗]+\s*https?://", text, flags=re.I)
        )
        return near_bottom and small_font and (footnote_markers or numbered_source_lines)

    def _is_document_title(self, block: dict, body_font_size: float, page_number: int) -> bool:
        if page_number != 1:
            return False
        text = block["text"].strip()
        if len(text) < 20 or "\n" in text:
            return False
        if block["bbox"][1] > 120:
            return False
        if "@" in text or "university" in text.lower():
            return False
        return block["font_size"] >= body_font_size + 3.0

    def _is_formula_or_algorithm_block(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if re.search(r"\(\d+\)\s*$", stripped):
            return True
        if re.search(r"[=∥√βϕθψαλµσ→←]", stripped):
            return True
        if stripped.startswith(("if ", "while ", "return ", "foreach ")):
            return True
        if re.match(r"^\d+\s+(foreach|while|return|end|if)\b", stripped, flags=re.I):
            return True
        if "/*" in stripped or "*/" in stripped or ";" in stripped:
            return True
        return False

    def _normalize_heading_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        if len(lines) == 2 and self.TITLE_NUMBER_RE.match(lines[0] + " "):
            return " ".join(lines)
        return "\n".join(lines)

    def _is_axis_or_legend_label(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines and all(re.fullmatch(r"[\d.]+", line) for line in lines):
            return True
        if re.match(r"^\d+\s+[a-z]", stripped):
            return True
        return False

    def _infer_title_level(self, text: str) -> int:
        if text in {"Abstract", "References"}:
            return 0
        match = self.TITLE_NUMBER_RE.match(text)
        if match:
            return match.group(1).count(".") + 1
        return 1

    def _update_section_path(
        self, current_section_path: list[str], title_text: str, level: int
    ) -> list[str]:
        if level <= 1:
            return [title_text]
        if not current_section_path:
            return [title_text]
        trimmed = current_section_path[: level - 1]
        if len(trimmed) < level - 1:
            trimmed = current_section_path[:]
        return trimmed + [title_text]

    def _looks_like_table_region(self, block: dict, tables: list[dict]) -> bool:
        if not tables:
            return False
        if self._is_formula_or_algorithm_block(block["text"]):
            return False
        text = block["text"]
        if "|" in text or "\t" in text:
            return True
        if len(text.split()) < 6 and any(char.isdigit() for char in text):
            return False
        for table in tables:
            if not self._should_emit_table(table):
                continue
            bbox = table.get("bbox")
            if not bbox:
                continue
            vertical_gap = abs(block["bbox"][1] - bbox[1])
            horizontal_overlap = min(block["bbox"][2], bbox[2]) - max(block["bbox"][0], bbox[0])
            if vertical_gap <= 28 and horizontal_overlap > 0:
                return True
        return False

    def _should_emit_table(self, table: dict) -> bool:
        rows = table.get("rows") or []
        if len(rows) < 2:
            return False
        max_cols = max((len(row) for row in rows if row), default=0)
        if max_cols < 3:
            return False
        populated_cells = sum(
            1 for row in rows for cell in row if cell not in (None, "", " ")
        )
        if populated_cells < 6:
            return False
        return True

    def _make_block(
        self,
        layout_block: dict,
        block_type: str,
        text: str,
        section_path: list[str],
        extra: dict | None = None,
    ) -> dict:
        block = {
            "block_id": f"p{layout_block['page']}_b{layout_block['layout_index']}",
            "page": layout_block["page"],
            "type": block_type,
            "text": text,
            "bbox": layout_block["bbox"],
            "section_path": section_path,
            "source_method": "pymupdf",
        }
        if extra:
            block.update(extra)
        return block

    def _make_table_block(
        self, layout_block: dict, table: dict, section_path: list[str]
    ) -> dict:
        return {
            "block_id": table["table_id"],
            "page": layout_block["page"],
            "type": "table",
            "text": self._table_to_text(table["rows"]) or layout_block["text"],
            "bbox": table["bbox"] or layout_block["bbox"],
            "section_path": section_path,
            "source_method": "pymupdf",
            "table_id": table["table_id"],
            "table_caption": None,
            "rows": table["rows"],
        }

    def _merge_blocks(self, blocks: list[dict]) -> dict:
        merged_text = "\n".join(block["text"].strip() for block in blocks if block["text"].strip())
        x0 = min(block["bbox"][0] for block in blocks)
        y0 = min(block["bbox"][1] for block in blocks)
        x1 = max(block["bbox"][2] for block in blocks)
        y1 = max(block["bbox"][3] for block in blocks)
        font_sizes = [block["font_size"] for block in blocks if block.get("font_size", 0) > 0]
        return {
            "page": blocks[0]["page"],
            "text": merged_text,
            "bbox": [x0, y0, x1, y1],
            "font_size": round(statistics.median(font_sizes), 2) if font_sizes else 0.0,
            "is_bold": any(block.get("is_bold") for block in blocks),
            "width": x1 - x0,
            "height": y1 - y0,
            "column_index": 0,
            "is_full_width": True,
            "layout_index": blocks[0]["layout_index"],
        }

    def _table_to_text(self, rows: list[list]) -> str:
        flat_rows = []
        for row in rows or []:
            values = [str(cell).strip() for cell in row if cell not in (None, "")]
            if values:
                flat_rows.append(" | ".join(values))
        return "\n".join(flat_rows)

from datetime import datetime
import logging
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkingService:
    """Text and structured-block chunking service."""

    def chunk_text(
        self,
        text: str,
        method: str,
        metadata: dict,
        page_map: list | None = None,
        chunk_size: int = 1000,
    ) -> dict:
        try:
            if not page_map:
                raise ValueError("Page map is required for chunking.")

            chunks = []
            total_pages = len(page_map)

            if method == "by_pages":
                for page_data in page_map:
                    chunk_metadata = {
                        "chunk_id": len(chunks) + 1,
                        "page_number": page_data["page"],
                        "page_range": str(page_data["page"]),
                        "word_count": self._estimate_word_count(page_data["text"]),
                    }
                    chunks.append({"content": page_data["text"], "metadata": chunk_metadata})

            elif method == "fixed_size":
                for page_data in page_map:
                    page_chunks = self._fixed_size_chunks(page_data["text"], chunk_size)
                    for chunk in page_chunks:
                        chunk_metadata = {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data["page"],
                            "page_range": str(page_data["page"]),
                            "word_count": self._estimate_word_count(chunk["text"]),
                        }
                        chunks.append({"content": chunk["text"], "metadata": chunk_metadata})

            elif method in ["by_paragraphs", "by_sentences"]:
                splitter_method = (
                    self._paragraph_chunks if method == "by_paragraphs" else self._sentence_chunks
                )
                for page_data in page_map:
                    page_chunks = splitter_method(page_data["text"])
                    for chunk in page_chunks:
                        chunk_metadata = {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_data["page"],
                            "page_range": str(page_data["page"]),
                            "word_count": self._estimate_word_count(chunk["text"]),
                        }
                        chunks.append({"content": chunk["text"], "metadata": chunk_metadata})
            else:
                raise ValueError(f"Unsupported chunking method: {method}")

            return {
                "filename": metadata.get("filename", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "loading_method": metadata.get("loading_method", ""),
                "chunking_method": method,
                "chunk_source": metadata.get("chunk_source", "loaded_text"),
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks,
            }
        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}")
            raise

    def chunk_structured_blocks(
        self,
        blocks: list,
        method: str,
        metadata: dict,
        chunk_size: int = 1200,
    ) -> dict:
        try:
            if not blocks:
                raise ValueError("Structured blocks are required for structured chunking.")

            filtered_blocks = [
                block
                for block in blocks
                if block.get("type") not in {"footnote", "reference_item"}
            ]
            if not filtered_blocks:
                raise ValueError("No usable structured blocks remain after filtering.")

            if method == "by_blocks":
                chunks = self._chunk_by_blocks(filtered_blocks)
            elif method == "by_sections":
                chunks = self._chunk_by_sections(filtered_blocks, chunk_size)
            else:
                raise ValueError(f"Unsupported structured chunking method: {method}")

            total_pages = len({block.get("page") for block in filtered_blocks if block.get("page")})

            return {
                "filename": metadata.get("filename", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "loading_method": metadata.get("loading_method", ""),
                "chunking_method": method,
                "chunk_source": "structured_blocks",
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks,
            }
        except Exception as e:
            logger.error(f"Error in chunk_structured_blocks: {str(e)}")
            raise

    def _chunk_by_blocks(self, blocks: list) -> list[dict]:
        chunks = []
        current_title = None
        document_title = None

        for block in blocks:
            block_type = block.get("type")
            if block_type == "title" and block.get("title_role") == "document_title":
                document_title = block.get("text")
                continue

            if block_type == "title":
                current_title = block.get("text")
                continue

            if block_type not in {"paragraph", "abstract_body", "table", "caption"}:
                continue

            page = block.get("page")
            content_parts = []
            if current_title:
                content_parts.append(current_title)
            if block_type == "caption" and chunks and chunks[-1]["metadata"].get("block_type") == "table":
                chunks[-1]["metadata"]["table_caption"] = block.get("text")
                continue
            content_parts.append(block.get("text", ""))

            chunk_content = "\n\n".join(part for part in content_parts if part).strip()
            chunk_metadata = {
                "chunk_id": len(chunks) + 1,
                "page_number": page,
                "page_range": str(page),
                "word_count": self._estimate_word_count(chunk_content),
                "block_type": block_type,
                "section_title": current_title,
                "section_path": block.get("section_path", []),
                "document_title": document_title,
                "source_method": block.get("source_method", "pymupdf"),
            }
            if block_type == "table":
                chunk_metadata["table_id"] = block.get("table_id")
                chunk_metadata["rows"] = block.get("rows", [])

            chunks.append({"content": chunk_content, "metadata": chunk_metadata})

        return chunks

    def _chunk_by_sections(self, blocks: list, chunk_size: int) -> list[dict]:
        chunks = []
        document_title = None
        sections: dict[tuple, dict] = {}

        for block in blocks:
            block_type = block.get("type")
            if block_type == "title" and block.get("title_role") == "document_title":
                document_title = block.get("text")
                continue

            if block_type not in {"title", "paragraph", "abstract_body", "table", "caption"}:
                continue

            if block_type == "caption":
                continue

            section_path = block.get("section_path") or ["Document"]
            key = tuple(section_path)
            section = sections.setdefault(
                key,
                {
                    "texts": [],
                    "pages": set(),
                    "block_types": set(),
                    "section_path": list(section_path),
                },
            )

            if block_type == "title":
                continue

            section["texts"].append(block.get("text", ""))
            if block.get("page"):
                section["pages"].add(block["page"])
            section["block_types"].add(block_type)
            source_method = block.get("source_method")
            if source_method:
                section.setdefault("source_methods", set()).add(source_method)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " "],
        )

        for section in sections.values():
            combined_text = "\n\n".join(text for text in section["texts"] if text).strip()
            if not combined_text:
                continue
            section_title = section["section_path"][-1] if section["section_path"] else None
            page_numbers = sorted(section["pages"])
            source_methods = sorted(section.get("source_methods", set()))
            page_range = (
                f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0])
            ) if page_numbers else "N/A"

            for piece in splitter.split_text(combined_text):
                piece_content = (
                    f"{section_title}\n\n{piece}".strip()
                    if section_title and not piece.startswith(section_title)
                    else piece
                )
                chunks.append(
                    {
                        "content": piece_content,
                        "metadata": {
                            "chunk_id": len(chunks) + 1,
                            "page_number": page_numbers[0] if page_numbers else None,
                            "page_range": page_range,
                            "word_count": self._estimate_word_count(piece_content),
                            "block_type": "section",
                            "section_title": section_title,
                            "section_path": section["section_path"],
                            "document_title": document_title,
                            "source_method": source_methods[0] if len(source_methods) == 1 else source_methods,
                            "contains_tables": "table" in section["block_types"],
                        },
                    }
                )

        return chunks

    def _fixed_size_chunks(self, text: str, chunk_size: int) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " ", ""],
            length_function=len,
        )
        chunks = []
        for piece in splitter.split_text(text):
            cleaned = piece.strip()
            if cleaned and self._estimate_word_count(cleaned) > 0:
                chunks.append({"text": cleaned})
        return chunks

    def _paragraph_chunks(self, text: str) -> list[dict]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [{"text": para} for para in paragraphs]

    def _sentence_chunks(self, text: str) -> list[dict]:
        sentence_candidates = re.split(r"(?<=[。！？；.!?])\s*", text or "")
        sentences = []
        overflow_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " ", ""],
            length_function=len,
        )

        for candidate in sentence_candidates:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            if len(cleaned) <= 1000:
                if self._estimate_word_count(cleaned) > 0:
                    sentences.append({"text": cleaned})
                continue

            for piece in overflow_splitter.split_text(cleaned):
                piece = piece.strip()
                if piece and self._estimate_word_count(piece) > 0:
                    sentences.append({"text": piece})

        return sentences

    def _estimate_word_count(self, text: str) -> int:
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text or "")
        return len(tokens)

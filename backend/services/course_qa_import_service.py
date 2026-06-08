import hashlib
import json
from datetime import datetime
from pathlib import Path

from utils.paths import LOADED_DOCS_DIR, WORKSPACE_DIR, workspace_relative


DEFAULT_COURSE_QA_SOURCE = WORKSPACE_DIR / "sample_papers_2026" / "course_qa.json"
DEFAULT_LOADED_DOC_NAME = "courseqa_default_loaded.json"


class CourseQaImportService:
    """Import course QA JSON into the existing loaded-document format."""

    def __init__(
        self,
        source_path: Path = DEFAULT_COURSE_QA_SOURCE,
        output_dir: Path = LOADED_DOCS_DIR,
        output_name: str = DEFAULT_LOADED_DOC_NAME,
    ):
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.output_name = output_name

    def ensure_default_loaded_document(self, overwrite: bool = False) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self.output_name
        if output_path.exists() and not overwrite:
            with output_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        document_data = self.build_loaded_document()
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(document_data, handle, ensure_ascii=False, indent=2)
        return document_data

    def build_loaded_document(self) -> dict:
        raw_data = json.loads(self.source_path.read_text(encoding="utf-8"))
        chunks = []
        question_index = 0

        for category, items in raw_data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                question = str(item.get("question", "")).strip()
                if not question:
                    continue
                question_index += 1
                qa_id = int(item.get("id", question_index))
                for answer_rank, answer in enumerate(item.get("answers", []), start=1):
                    answer_text = str(answer.get("answer", "")).strip()
                    if not answer_text:
                        continue
                    answer_id = self._stable_answer_id(category, qa_id, answer_text)
                    content = (
                        f"课程主题：{category}\n"
                        f"问题：{question}\n"
                        f"候选答案：{answer_text}"
                    )
                    chunks.append(
                        {
                            "content": content,
                            "metadata": {
                                "chunk_id": len(chunks) + 1,
                                "page_number": question_index,
                                "page_range": str(question_index),
                                "word_count": len(content.split()),
                                "category": category,
                                "qa_id": qa_id,
                                "question": question,
                                "answer_id": answer_id,
                                "answer_rank": answer_rank,
                                "source_dataset": workspace_relative(self.source_path),
                            },
                        }
                    )

        return {
            "filename": "courseqa.pdf",
            "total_chunks": len(chunks),
            "total_pages": question_index,
            "loading_method": "course_qa_json",
            "source_file": workspace_relative(self.source_path),
            "loading_strategy": None,
            "chunking_strategy": None,
            "chunking_method": "loaded",
            "timestamp": datetime.now().isoformat(),
            "chunks": chunks,
        }

    @staticmethod
    def _stable_answer_id(category: str, qa_id: int, answer_text: str) -> str:
        seed = f"{category}::{qa_id}::{answer_text}"
        return f"ans-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"

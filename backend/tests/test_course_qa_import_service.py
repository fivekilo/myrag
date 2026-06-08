import tempfile
import unittest
from pathlib import Path

from services.course_qa_import_service import CourseQaImportService


class CourseQaImportServiceTests(unittest.TestCase):
    def test_build_loaded_document_matches_existing_loaded_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "course_qa.json"
            source_path.write_text(
                """
                {
                  "课程知识问答": [
                    {
                      "id": 1,
                      "question": "什么是NLP？",
                      "answers": [
                        {"answer_quality": 9, "answer": "自然语言处理研究机器理解和生成语言。"},
                        {"answer_quality": 1, "answer": "它和语言有关。"}
                      ]
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            service = CourseQaImportService(
                source_path=source_path,
                output_dir=temp_path,
                output_name="courseqa_default_loaded.json",
            )

            document = service.build_loaded_document()

            self.assertEqual(document["filename"], "courseqa.pdf")
            self.assertEqual(document["loading_method"], "course_qa_json")
            self.assertEqual(document["total_pages"], 1)
            self.assertEqual(document["total_chunks"], 2)
            self.assertEqual(document["chunking_method"], "loaded")
            self.assertTrue(document["source_file"].endswith("course_qa.json"))

            first_chunk = document["chunks"][0]
            self.assertIn("课程主题：课程知识问答", first_chunk["content"])
            self.assertIn("问题：什么是NLP？", first_chunk["content"])
            self.assertIn("候选答案：自然语言处理研究机器理解和生成语言。", first_chunk["content"])
            self.assertIn("answer_id", first_chunk["metadata"])
            self.assertNotIn("answer_quality", first_chunk["content"])
            self.assertNotIn("answer_quality", first_chunk["metadata"])


if __name__ == "__main__":
    unittest.main()

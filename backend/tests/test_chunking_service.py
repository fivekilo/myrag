import unittest

from services.chunking_service import ChunkingService


class ChunkingServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ChunkingService()
        self.metadata = {
            "filename": "paper.pdf",
            "loading_method": "pymupdf",
            "chunk_source": "structured_blocks",
        }

    def test_chunk_by_blocks_filters_footnotes_and_references(self):
        blocks = [
            {
                "page": 1,
                "type": "title",
                "text": "Paper Title",
                "title_role": "document_title",
                "section_path": [],
            },
            {
                "page": 1,
                "type": "title",
                "text": "1 Introduction",
                "section_path": ["1 Introduction"],
            },
            {
                "page": 1,
                "type": "paragraph",
                "text": "This is the first body paragraph.",
                "section_path": ["1 Introduction"],
                "source_method": "pymupdf",
            },
            {
                "page": 1,
                "type": "footnote",
                "text": "Footnote text that should be ignored.",
                "section_path": ["1 Introduction"],
            },
            {
                "page": 5,
                "type": "reference_item",
                "text": "Smith et al. 2026.",
                "section_path": ["References"],
            },
        ]

        result = self.service.chunk_structured_blocks(blocks, "by_blocks", self.metadata)

        self.assertEqual(result["chunk_source"], "structured_blocks")
        self.assertEqual(result["total_chunks"], 1)
        self.assertIn("1 Introduction", result["chunks"][0]["content"])
        self.assertEqual(result["chunks"][0]["metadata"]["document_title"], "Paper Title")
        self.assertEqual(result["chunks"][0]["metadata"]["block_type"], "paragraph")

    def test_chunk_by_sections_groups_same_section_across_pages(self):
        blocks = [
            {
                "page": 1,
                "type": "title",
                "text": "Paper Title",
                "title_role": "document_title",
                "section_path": [],
            },
            {
                "page": 2,
                "type": "title",
                "text": "2 Method",
                "section_path": ["2 Method"],
            },
            {
                "page": 2,
                "type": "paragraph",
                "text": "First method paragraph.",
                "section_path": ["2 Method"],
                "source_method": "pymupdf",
            },
            {
                "page": 3,
                "type": "paragraph",
                "text": "Second method paragraph on the next page.",
                "section_path": ["2 Method"],
                "source_method": "pymupdf",
            },
            {
                "page": 3,
                "type": "caption",
                "text": "Figure 1: Example figure.",
                "section_path": ["2 Method"],
            },
        ]

        result = self.service.chunk_structured_blocks(
            blocks,
            "by_sections",
            self.metadata,
            chunk_size=500,
        )

        self.assertEqual(result["total_chunks"], 1)
        chunk = result["chunks"][0]
        self.assertEqual(chunk["metadata"]["block_type"], "section")
        self.assertEqual(chunk["metadata"]["section_title"], "2 Method")
        self.assertEqual(chunk["metadata"]["page_range"], "2-3")
        self.assertIn("First method paragraph.", chunk["content"])
        self.assertIn("Second method paragraph on the next page.", chunk["content"])


if __name__ == "__main__":
    unittest.main()

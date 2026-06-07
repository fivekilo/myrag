import unittest

from services.parsing_service import ParsingService


class ParsingServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ParsingService()

    def test_caption_regex_matches_acl_style_caption(self):
        self.assertTrue(self.service.CAPTION_RE.match("Figure 1: Architecture overview"))
        self.assertTrue(self.service.CAPTION_RE.match("Table 2: Results on dev set"))

    def test_numbered_title_level_is_inferred(self):
        self.assertEqual(self.service._infer_title_level("1 Introduction"), 1)
        self.assertEqual(self.service._infer_title_level("3.2 Model"), 2)
        self.assertEqual(self.service._infer_title_level("References"), 0)

    def test_repeated_noise_normalization_collapses_page_ranges(self):
        normalized = self.service._normalize_repetition_text(
            "Proceedings of ACL 2026, pages 4886–4909"
        )
        self.assertIn("pages", normalized)
        self.assertNotIn("4886", normalized)

    def test_author_metadata_on_first_page_is_not_title(self):
        block = {
            "text": "Heidelberg Institute for Theoretical Studies | wei@example.com",
            "bbox": [100.0, 150.0, 500.0, 175.0],
            "font_size": 11.0,
            "is_bold": False,
        }
        semantic_type = self.service._classify_block(
            block=block,
            body_font_size=10.0,
            first_page=True,
            reference_mode=False,
            page_height=842.0,
        )
        self.assertEqual(semantic_type, "author_metadata")

    def test_reference_mode_forces_reference_items(self):
        block = {
            "text": "Smith et al. 2024. Dense Retrieval Advances.",
            "bbox": [90.0, 200.0, 460.0, 220.0],
            "font_size": 9.0,
            "is_bold": False,
        }
        semantic_type = self.service._classify_block(
            block=block,
            body_font_size=10.0,
            first_page=False,
            reference_mode=True,
            page_height=842.0,
        )
        self.assertEqual(semantic_type, "reference_item")

    def test_bottom_small_url_block_is_classified_as_footnote(self):
        block = {
            "text": "1https://sdgs.un.org/goals\n*Correspondence: foo@example.com",
            "bbox": [70.0, 742.0, 280.0, 775.0],
            "font_size": 6.9,
            "is_bold": False,
        }
        semantic_type = self.service._classify_block(
            block=block,
            body_font_size=10.5,
            first_page=True,
            reference_mode=False,
            page_height=842.0,
        )
        self.assertEqual(semantic_type, "footnote")

    def test_first_page_author_blocks_are_merged_into_single_block(self):
        layout_pages = [
            {
                "page": 1,
                "width": 595.0,
                "height": 842.0,
                "body_font_size": 10.0,
                "blocks": [
                    {
                        "page": 1,
                        "text": "Paper Title for Testing",
                        "bbox": [80.0, 60.0, 500.0, 95.0],
                        "font_size": 16.0,
                        "is_bold": True,
                        "width": 420.0,
                        "height": 35.0,
                        "column_index": 0,
                        "layout_index": 1,
                    },
                    {
                        "page": 1,
                        "text": "Ana Ezquerro1,2",
                        "bbox": [120.0, 120.0, 300.0, 140.0],
                        "font_size": 11.0,
                        "is_bold": False,
                        "width": 180.0,
                        "height": 20.0,
                        "column_index": 0,
                        "layout_index": 2,
                    },
                    {
                        "page": 1,
                        "text": "ana.ezquerro\nCarlos Gomez-Rodriguez1",
                        "bbox": [120.0, 142.0, 320.0, 178.0],
                        "font_size": 10.5,
                        "is_bold": False,
                        "width": 200.0,
                        "height": 36.0,
                        "column_index": 0,
                        "layout_index": 3,
                    },
                    {
                        "page": 1,
                        "text": "Abstract",
                        "bbox": [90.0, 230.0, 170.0, 248.0],
                        "font_size": 12.0,
                        "is_bold": True,
                        "width": 80.0,
                        "height": 18.0,
                        "column_index": 0,
                        "layout_index": 4,
                    },
                ],
            }
        ]

        blocks, _ = self.service._build_semantic_blocks(layout_pages, set(), {})
        author_blocks = [block for block in blocks if block["type"] == "author_metadata"]

        self.assertEqual(len(author_blocks), 1)
        self.assertIn("Ana Ezquerro1,2", author_blocks[0]["text"])
        self.assertIn("Carlos Gomez-Rodriguez1", author_blocks[0]["text"])


if __name__ == "__main__":
    unittest.main()

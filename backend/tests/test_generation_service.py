import unittest

from services.generation_service import GenerationService


class GenerationServiceTests(unittest.TestCase):
    def test_aliyun_models_include_qwen37max(self):
        service = GenerationService()

        aliyun_models = service.get_available_models()["aliyun"]

        self.assertIn("qwen3.7-max", aliyun_models)
        self.assertEqual(aliyun_models["qwen3.7-max"], "qwen3.7-max")
        self.assertIn("qwen3.7-max-2026-05-20", aliyun_models)


if __name__ == "__main__":
    unittest.main()

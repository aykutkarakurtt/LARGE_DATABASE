import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from llm_client import LLMClientError
from rag_service import RagService


class FakeSearchEngine:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, top_k=5, threshold=0.0):
        self.calls.append((query, top_k, threshold))
        return self.hits


class FakeLLMClient:
    def __init__(self, error=None):
        self.error = error
        self.prompt = None
        self.system_prompt = None

    def complete(self, prompt, system_prompt=None):
        self.prompt = prompt
        self.system_prompt = system_prompt
        if self.error:
            raise self.error
        return "Cevap [1]."


class RagServiceTests(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {
                "text": "Belgede geçen önemli bilgi.",
                "source": "ornek.pdf",
                "page": 3,
                "chunk_index": 2,
                "score": 0.82,
            }
        ]

    def test_answer_builds_context_and_returns_answer(self):
        search_engine = FakeSearchEngine(self.hits)
        llm_client = FakeLLMClient()
        service = RagService(search_engine, llm_client=llm_client)

        result = service.answer("Bu bilgi nedir?", top_k=3, threshold=0.4)

        self.assertEqual(result["answer"], "Cevap [1].")
        self.assertIsNone(result["warning"])
        self.assertEqual(search_engine.calls, [("Bu bilgi nedir?", 3, 0.4)])
        self.assertIn("ornek.pdf", llm_client.prompt)
        self.assertIn("Sayfa: 3", llm_client.prompt)
        self.assertIn("Türkçe", llm_client.system_prompt)

    def test_answer_without_hits_does_not_call_llm(self):
        search_engine = FakeSearchEngine([])
        llm_client = FakeLLMClient()
        service = RagService(search_engine, llm_client=llm_client)

        result = service.answer("Eksik bilgi")

        self.assertIn("yeterli kaynak bulunamadı", result["answer"])
        self.assertEqual(result["hits"], [])
        self.assertIsNone(llm_client.prompt)

    def test_answer_returns_retrieval_fallback_on_llm_error(self):
        search_engine = FakeSearchEngine(self.hits)
        llm_client = FakeLLMClient(LLMClientError("sunucu yok"))
        service = RagService(search_engine, llm_client=llm_client)

        result = service.answer("Bu bilgi nedir?")

        self.assertIn("cevap üretemedi", result["answer"])
        self.assertEqual(result["warning"], "sunucu yok")
        self.assertEqual(result["hits"], self.hits)

    def test_duplicate_hits_are_removed(self):
        duplicate_hits = self.hits + [dict(self.hits[0])]
        search_engine = FakeSearchEngine(duplicate_hits)
        llm_client = FakeLLMClient()
        service = RagService(search_engine, llm_client=llm_client)

        result = service.answer("Bu bilgi nedir?")

        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(llm_client.prompt.count("Belge: ornek.pdf"), 1)


if __name__ == "__main__":
    unittest.main()

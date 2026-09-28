import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    RAG_CONTEXT_SIZE,
    RAG_MAX_TOKENS,
    RAG_TEMPERATURE,
)


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        timeout=OLLAMA_TIMEOUT,
        temperature=RAG_TEMPERATURE,
        context_size=RAG_CONTEXT_SIZE,
        max_tokens=RAG_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/api"):
            self.base_url = self.base_url[:-4]
        self.model = model
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.context_size = int(context_size)
        self.max_tokens = int(max_tokens)

    def complete(self, prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_size,
                "num_predict": self.max_tokens,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = self._error_detail(error)
            raise LLMClientError(
                f"Ollama isteği başarısız ({error.code}): {detail}"
            ) from error
        except TimeoutError as error:
            raise LLMClientError("Ollama yanıt vermedi; zaman aşımına uğradı.") from error
        except URLError as error:
            raise LLMClientError("Ollama sunucusuna ulaşılamadı.") from error

        try:
            data = json.loads(body)
        except json.JSONDecodeError as error:
            raise LLMClientError("Ollama geçersiz bir yanıt döndürdü.") from error

        if not isinstance(data, dict):
            raise LLMClientError("Ollama yanıtı beklenen biçimde değil.")

        if data.get("error"):
            raise LLMClientError(str(data["error"]))

        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMClientError("Ollama yanıtında mesaj bulunamadı.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("Ollama boş bir cevap döndürdü.")

        return content.strip()

    @staticmethod
    def _error_detail(error):
        try:
            detail = error.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = str(error.reason)
        return detail[:240] or "Bilinmeyen Ollama hatası"

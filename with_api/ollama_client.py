import os
import requests
from typing import Optional

class OllamaClient:
    """
    Thin wrapper around local or cloud Ollama HTTP API.
    Handles both Native (/api/chat) and OpenAI-compatible (/v1/chat/completions) formats.
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", os.getenv("OLLAMA_DEFAULT_MODEL", "llama3"))
        self.headers = {"Content-Type": "application/json"}

        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            # Cloud mode
            self.headers["Authorization"] = f"Bearer {api_key}"
            self._preferred_endpoints = ["/v1/chat/completions"]
        else:
            # Local mode - try Native first, then OpenAI compatible
            self._preferred_endpoints = ["/api/chat", "/v1/chat/completions"]

    def _full_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def summarise(
        self,
        text: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a concise, factual summariser."},
                {"role": "user", "content": f"Summarise the following text:\n\n{text}"},
            ],
            "temperature": temperature,
            "stream": stream,
        }

        last_err = None
        for ep in self._preferred_endpoints:
            url = self._full_url(ep)
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
            except Exception as exc:
                last_err = exc
                continue

            if resp.status_code == 200:
                data = resp.json()

                # 1. Handle Native Ollama Format (key: "message")
                if "message" in data and isinstance(data["message"], dict):
                    return data["message"].get("content", "").strip()

                # 2. Handle OpenAI Format (key: "choices")
                if "choices" in data and isinstance(data["choices"], list):
                    choice = data["choices"][0]
                    if "message" in choice:
                        return choice["message"].get("content", "").strip()

                # 3. Handle /api/generate format (key: "response") - just in case
                if "response" in data:
                    return data.get("response", "").strip()

            # If we get here, the endpoint exists but format was unknown or error code
            last_err = Exception(f"{resp.status_code} {resp.reason} from {url}: {resp.text[:200]}")
            continue

        if last_err:
            raise last_err
        raise Exception("No endpoints available or valid responses received from Ollama.")

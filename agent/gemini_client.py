import os
import requests
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

DEFAULT_GEMINI_BASE = os.getenv("GEMINI_BASE_URL", "https://api.gemini.example/v1")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash"))


def call_gemini(prompt: str, model: str = DEFAULT_GEMINI_MODEL, host: Optional[str] = None, timeout: int = 60) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    base = (host or os.getenv("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE).rstrip("/")
    if "/models/" in base or base.endswith(":predict"):
        url = base
    else:
        url = f"{base}/models/{model}:predict"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"prompt": prompt}

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "candidates" in data and isinstance(data["candidates"], list):
                for c in data["candidates"]:
                    if isinstance(c, dict) and "content" in c:
                        return c.get("content", "").strip()
            if "choices" in data and isinstance(data["choices"], list):
                choice = data["choices"][0]
                if isinstance(choice, dict) and "message" in choice:
                    return choice["message"].get("content", "").strip()
                if isinstance(choice, dict) and "content" in choice:
                    return choice.get("content", "").strip()
            if "output" in data:
                out = data.get("output")
                if isinstance(out, dict) and "text" in out:
                    return out.get("text", "").strip()
                if isinstance(out, list) and out:
                    first = out[0]
                    if isinstance(first, dict) and "content" in first:
                        return first["content"].strip()
            if "text" in data:
                return str(data.get("text", "")).strip()
        return r.text.strip()[:65536]
    except Exception:
        return ""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from agent.gemini_client import call_gemini

# Default configuration
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

PROMPT_SUMMARY = (
    "You are a helpful legal summarization assistant. Summarize the following legal text in 6 concise bullet points focusing on: purpose, key definitions, eligibility, obligations, payments/entitlements, and enforcement. Be factual and cite short evidence snippets when possible.\n\nText:\n{content}"
)

PROMPT_EXTRACT = (
    "Extract the following sections from the provided legal text and return strictly valid JSON with keys: definitions, obligations, responsibilities, eligibility, payments, penalties, record_keeping. If a section is not present, return an empty string for that key. Do not include markdown formatting (like ```json).\n\nText:\n{content}"
)

PROMPT_SUMMARY_OF_SUMMARIES = (
    "You are a helpful legal summarization assistant. Given the following short bullet summaries (one per chunk), produce a final concise summary in 6 bullet points focusing on: purpose, key definitions, eligibility, obligations, payments/entitlements, and enforcement. Be factual and cite short evidence snippets when possible.\n\nSummaries:\n{content}"
)

def call_ollama(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 2000, host: str = DEFAULT_OLLAMA_HOST) -> str:
    """Call an Ollama server's /api/chat endpoint with robust response parsing."""
    url = host.rstrip("/") + "/api/chat"
    
    # Check if we are using an OpenAI-compatible endpoint path instead
    if "/v1/" in host:
        url = host.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # Note: 'max_tokens' is not always respected by native Ollama /api/chat, but good to keep
        "options": {"num_predict": max_tokens} 
    }
    
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # --- ROBUST PARSING LOGIC ---
        
        # 1. Native Ollama Format (key: "message")
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "").strip()

        # 2. OpenAI Format (key: "choices")
        if "choices" in data and isinstance(data["choices"], list):
            choice = data["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                return choice["message"].get("content", "").strip()
            if isinstance(choice, dict) and "text" in choice: # Legacy completion
                return choice.get("text", "").strip()

        # 3. Legacy /api/generate Format (key: "response")
        if "response" in data:
             return data.get("response", "").strip()

    except Exception as e:
        print(f"API Call Failed: {e}")
        return ""

    return ""


def summarize_with_api(text: str, api_key: Optional[str] = None, provider: str = "ollama", model: Optional[str] = None, host: Optional[str] = None) -> str:
    """Unified summarization API."""
    provider = (provider or "ollama").lower()
    model = model or DEFAULT_MODEL
    
    # Prefer Gemini if GEMINI_API_KEY present
    env_gemini = os.getenv("GEMINI_API_KEY")
    use_gemini = bool(env_gemini) or (api_key and api_key.lower().startswith("g"))

    if use_gemini:
        # Send full text to Gemini (no client-side chunking or truncation).
        return call_gemini(PROMPT_SUMMARY.format(content=text), model=(model or os.getenv("GEMINI_MODEL")))

    # Fallback for OpenAI
    if provider == "openai":
        return _call_openai(PROMPT_SUMMARY.format(content=text), api_key, model)
    
    return "Unsupported provider"


def extract_sections_with_api(text: str, api_key: Optional[str] = None, provider: str = "ollama", model: Optional[str] = None, host: Optional[str] = None) -> str:
    """Unified section extraction API."""
    provider = (provider or "ollama").lower()
    model = model or DEFAULT_MODEL
    # Prefer Gemini if present
    env_gemini = os.getenv("GEMINI_API_KEY")
    use_gemini = bool(env_gemini) or (api_key and api_key.lower().startswith("g"))

    if use_gemini:
        prompt = PROMPT_EXTRACT.format(content=text)
        return call_gemini(prompt, model=(model or os.getenv("GEMINI_MODEL")))

    if provider == "openai":
        return _call_openai(PROMPT_EXTRACT.format(content=text), api_key, model)

    if provider == "ollama":
        host = host or DEFAULT_OLLAMA_HOST
        prompt = PROMPT_EXTRACT.format(content=text)
        return call_ollama(prompt, model=model, host=host)

    return "{}"

def _call_openai(prompt: str, api_key: str, model: str) -> str:
    from requests import post
    if not api_key: 
        api_key = os.getenv("OPENAI_API_KEY")
    
    url = "[https://api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions)"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    try:
        r = post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI Call Failed: {e}")
        return "{}"
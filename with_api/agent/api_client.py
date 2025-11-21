import os
import requests
from typing import Optional
from dotenv import load_dotenv

from with_api.agent.gemini_client import call_gemini

load_dotenv()

# Default configuration
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- PROMPT 1: SUMMARY ---

PROMPT_SUMMARY = (
    "You are a legal expert. Summarize the following text in exactly 6 concise bullet points. "
    "Format your response strictly as a list, with each point on a new line starting with '* '.\n\n"
    "Focus specifically on:\n"
    "1. Purpose of the Act\n"
    "2. Key Definitions (specifically mention 'pre-2026 claimant')\n"
    "3. Eligibility Criteria\n"
    "4. Responsibilities of the Secretary of State\n"
    "5. Payment Rates (mention specific amounts like £217.26 and uplift %)\n"
    "6. Enforcement (or lack thereof)\n\n"
    "Text:\n{content}"
)

# --- PROMPT 2: EXTRACTION ---
# Explicitly maps the 'hidden' locations of definitions and responsibilities
PROMPT_EXTRACT = (
    "Analyze the provided legal text and extract content for the following 7 sections. "
    "Return STRICTLY valid JSON with these keys. If a section is empty, return \"\".\n\n"
    "KEYS TO EXTRACT:\n"
    "1. definitions: Look specifically for:\n"
    "   - Definitions inside 'Section 1(6)' (e.g., 'consumer prices index', 'relevant power').\n"
    "   - Definitions inserted into Regulation 2 via Schedule 1 (e.g., 'pre-2026 claimant').\n"
    "   - Terms defined in new Regulations 27A and 40A (e.g., 'Meaning of \"pre-2026 claimant\"').\n"
    "2. obligations: Extract duties of the **claimant** (individual). (Note: Likely empty for this Act).\n"
    "3. responsibilities: Extract duties of the **Secretary of State**. Look for phrases like 'The Secretary of State must exercise a relevant power'.\n"
    "4. eligibility: Extract criteria for who is entitled. Look for 'pre-2026 claimant' entitlement rules in Regulation 27A.\n"
    "5. payments: Extract all financial figures. Specifically look for:\n"
    "   - The 'Step 1, Step 2, Step 3' calculation method.\n"
    "   - The specific amount '£217.26'.\n"
    "   - The 'uplift percentage' table (2.3% to 4.8%).\n"
    "6. penalties: Extract text about offences or sanctions. (If none exist, return empty string).\n"
    "7. record_keeping: Extract reporting rules. Look for amendments to 'Regulation 43 (information requirement)'.\n\n"
    "Do not include Markdown formatting (no ```json). Output ONLY the JSON object.\n\nText:\n{content}"
)

def call_ollama(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 2000, host: str = DEFAULT_OLLAMA_HOST) -> str:
    """Call an Ollama server's /api/chat endpoint with robust response parsing."""
    url = host.rstrip("/") + "/api/chat"
    
    if "/v1/" in host:
        url = host.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
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

        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "").strip()

        if "choices" in data and isinstance(data["choices"], list):
            choice = data["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                return choice["message"].get("content", "").strip()
            if isinstance(choice, dict) and "text" in choice:
                return choice.get("text", "").strip()

        if "response" in data:
             return data.get("response", "").strip()

    except Exception as e:
        print(f"API Call Failed: {e}")
        return ""

    return ""


def summarize_with_api(text: str, api_key: Optional[str] = None, provider: str = "ollama", model: Optional[str] = None, host: Optional[str] = None) -> str:
    """Unified summarization API."""
    provider = (provider or "ollama").lower()
    
    if provider == "gemini":
        # Use model passed in arg, or env var, or default to 2.0-flash
        target_model = model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        return call_gemini(PROMPT_SUMMARY.format(content=text), model=target_model)

    # Fallback for OpenAI
    if provider == "openai":
        return _call_openai(PROMPT_SUMMARY.format(content=text), api_key, model)
    
    # Fallback for Ollama
    if provider == "ollama":
        host = host or DEFAULT_OLLAMA_HOST
        prompt = PROMPT_SUMMARY.format(content=text)
        return call_ollama(prompt, model=(model or DEFAULT_MODEL), host=host)
    
    return "Unsupported provider"


def extract_sections_with_api(text: str, api_key: Optional[str] = None, provider: str = "ollama", model: Optional[str] = None, host: Optional[str] = None) -> str:
    """Unified section extraction API."""
    provider = (provider or "ollama").lower()
    
    # FORCE Gemini 2.0 Flash if provider is Gemini
    if provider == "gemini":
        target_model = model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        return call_gemini(PROMPT_EXTRACT.format(content=text), model=target_model)

    if provider == "openai":
        return _call_openai(PROMPT_EXTRACT.format(content=text), api_key, model)

    if provider == "ollama":
        host = host or DEFAULT_OLLAMA_HOST
        prompt = PROMPT_EXTRACT.format(content=text)
        return call_ollama(prompt, model=(model or DEFAULT_MODEL), host=host)

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
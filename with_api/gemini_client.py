import os
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except Exception:
    genai = None

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if API_KEY and genai:
    try:
        genai.configure(api_key=API_KEY)
    except Exception:
        # Non-fatal: configuration may fail in restricted environments
        pass


def call_gemini(prompt: str, model: str = "gemini-2.0-flash", host: str = None, timeout: int = 60) -> str:
    """
    Calls Google Gemini API using the official SDK when available.
    Returns generated text, or an empty string on failure.
    """
    # Quick availability checks
    if genai is None:
        print("Gemini SDK not installed (google-generativeai). Install with `pip install google-generativeai`.")
        return ""

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Error: Missing GEMINI_API_KEY / GOOGLE_API_KEY in environment (.env).")
        return ""

    try:
        target_model = model or "gemini-2.0-flash"
        # Instantiate the model and generate content
        model_instance = genai.GenerativeModel(target_model)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model_instance.generate_content(prompt, safety_settings=safety_settings)

        # Prefer the high-level text attribute
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()

        # Fallbacks for other response shapes
        if hasattr(response, "candidates") and isinstance(response.candidates, (list, tuple)):
            first = response.candidates[0]
            return getattr(first, "content", str(first)).strip()

        # Last resort: try stringifying the response
        return str(response).strip()
    except Exception as e:
        print(f"Gemini API Call Failed: {e}")
        return ""

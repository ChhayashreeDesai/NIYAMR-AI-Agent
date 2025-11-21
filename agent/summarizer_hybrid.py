"""Hybrid summariser: extractive + Gemini abstractive refinement."""
from typing import List, Dict, Optional
import re
from with_api.gemini_client import call_gemini

# Helper functions (Extractive)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extractive_sentences(text: str, n: int = 12) -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences: return []
    if SKLEARN_OK:
        try:
            vect = TfidfVectorizer(stop_words="english").fit_transform(sentences)
            sim = linear_kernel(vect, vect)
            scores = sim.sum(axis=1)
            ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
            return [sentences[i] for i in ranked_idx[:n]]
        except Exception: pass
    return sentences[:n]


def categorize_by_headings(sentences: List[str], text: str) -> Dict[str, str]:
    # Bucket keywords
    buckets = {
        "Purpose": ["act", "provision", "purpose", "assist"],
        "Definitions": ["means", "interpretation", "meaning"],
        "Eligibility": ["claimant", "eligible", "entitled"],
        "Responsibilities": ["secretary of state", "must", "department"],
        "Payments": ["£", "%", "amount", "rate", "allowance"],
        "Enforcement": ["offence", "fraud", "penalty", "cease"],
    }
    results = {k: [] for k in buckets}
    for s in sentences:
        s_lower = s.lower()
        for cat, kws in buckets.items():
            if any(kw in s_lower for kw in kws):
                results[cat].append(s)
                break
    return {k: " ".join(v) for k, v in results.items() if v}


def abstractive_refine(text_chunk: str, category: str, model: str = "gemini-2.0-flash") -> str:
    if not text_chunk: return ""
    prompt = (
        f"Refine the following extracted text regarding '{category}' into a single, "
        "concise, legal-style bullet point. Preserve specific numbers/dates if present.\n"
        f"Text: {text_chunk}"
    )
    return call_gemini(prompt, model=model)


def hybrid_summary(text: str, extract_n: int = 20, abstractive_model: str = "gemini-2.0-flash") -> List[str]:
    # 1. Extractive Step
    sentences = extractive_sentences(text, n=extract_n)
    
    # 2. Categorize Step
    categories = categorize_by_headings(sentences, text)
    
    bullets = []
    # 3. Abstractive Refinement Step (Gemini)
    for cat, content in categories.items():
        refined = abstractive_refine(content, cat, model=abstractive_model)
        if refined:
            bullets.append(f"**{cat}:** {refined}")
            
    return bullets

"""Hybrid summariser: extractive + light abstractive refinement.

This module implements:
- extractive_sentences(text, method='tfidf'|'textrank') -> list[str]
- categorize_by_headings(sentences, text) -> dict of sections
- deduplicate_and_clean(sentences) -> list[str]
- abstractive_refine(sentences, model_name) -> str (uses transformers if available)

All heavy dependencies are optional; the module falls back to safe behaviour
if libraries are not installed.
"""
from typing import List, Dict, Optional
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import spacy
    SPACY_OK = True
except Exception:
    SPACY_OK = False

try:
    from transformers import pipeline
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False


def split_into_sentences(text: str) -> List[str]:
    if SPACY_OK:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            pass
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extractive_sentences(text: str, n: int = 20, method: str = "tfidf") -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    if method == "textrank":
        try:
            import pytextrank
            if SPACY_OK:
                nlp = spacy.load("en_core_web_sm")
                nlp.add_pipe("textrank")
                doc = nlp(text)
                ranked = [p.text for p in doc._.textrank.summary(limit_phrases=15, limit_sentences=n)]
                return ranked
        except Exception:
            pass

    if SKLEARN_OK:
        try:
            vect = TfidfVectorizer(stop_words="english").fit_transform(sentences)
            sim = linear_kernel(vect, vect)
            scores = sim.sum(axis=1)
            ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
            return [sentences[i] for i in ranked_idx[:n]]
        except Exception:
            pass

    return sentences[:n]


def deduplicate_and_clean(sentences: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in sentences:
        key = re.sub(r"[^\w\s]", "", s.lower()).strip()
        # allow shorter sentences through; only filter extremely short noise
        if not key or key in seen or len(key) < 10:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


def categorize_by_headings(sentences: List[str], text: str) -> Dict[str, str]:
    buckets = {
        "purpose": [], "definitions": [], "eligibility": [],
        "responsibilities": [], "payments": [], "penalties": [], "record_keeping": []
    }

    keywords = {
        "purpose": ["purpose", "object", "aim", "act"],
        "definitions": ["means", "interpretation", "defined"],
        "eligibility": ["eligible", "entitled", "claimant"],
        "responsibilities": ["must", "shall", "secretary", "duty"],
        "payments": ["allowance", "payment", "£", "rate", "uplift"],
        "penalties": ["penalty", "offence", "fraud"],
        "record_keeping": ["record", "evidence", "report"]
    }

    for s in sentences:
        low = s.lower()
        assigned = False
        for k, kws in keywords.items():
            if any(kw in low for kw in kws):
                buckets[k].append(s)
                assigned = True
                break
        if not assigned:
            buckets["purpose"].append(s)

    return {k: "\n".join(v) for k, v in buckets.items()}


def abstractive_refine(sentences: List[str], model_name: Optional[str] = None) -> str:
    if not sentences:
        return ""
    if TRANSFORMERS_OK:
        try:
            model = model_name or "facebook/bart-large-cnn"
            summariser = pipeline("summarization", model=model)
            out = summariser("\n".join(sentences), max_length=150, min_length=40, do_sample=False)
            if out and isinstance(out, list):
                return out[0].get("summary_text", "").strip()
        except Exception:
            pass
    return " ".join(sentences)


def hybrid_summary(text: str, extract_n: int = 20, extract_method: str = "tfidf", abstractive_model: Optional[str] = None) -> List[str]:
    """ROBUST VERSION: Guarantees 6-10 items."""
    sentences = extractive_sentences(text, n=extract_n, method=extract_method)
    candidates = deduplicate_and_clean(sentences)

    # If dedup removed everything, fall back to raw sentence split
    if not candidates:
        candidates = split_into_sentences(text)[:extract_n]

    cats = categorize_by_headings(candidates, text)
    bullets = []
    used = set()

    order = ["purpose", "definitions", "eligibility", "responsibilities", "payments", "penalties", "record_keeping"]
    for key in order:
        bucket_text = cats.get(key, "")
        if bucket_text:
            snippet = bucket_text[:300].rstrip()
            bullets.append(f"**{key.title()}:** {snippet}")
            # Mark approximate used sentences
            for s in candidates:
                if s in bucket_text:
                    used.add(s)

    # If we still have too few bullets, try splitting bucket_texts into individual sentences
    if len(bullets) < 6:
        for key in order:
            bucket_text = cats.get(key, "")
            if not bucket_text:
                continue
            parts = split_into_sentences(bucket_text)
            for p in parts:
                if len(bullets) >= 6:
                    break
                if p and p not in bullets:
                    bullets.append(p)
            if len(bullets) >= 6:
                break

    # FORCE FILLER: top unused candidates to reach at least 6 bullets
    if len(bullets) < 6:
        needed = 6 - len(bullets)
        added = 0
        for s in candidates:
            if s not in used and len(s) > 40:
                bullets.append(s)
                added += 1
                if added >= needed:
                    break

    # If still short, split long bullets into smaller parts (naive split)
    if len(bullets) < 6:
        long_source = " ".join(candidates)
        parts = split_into_sentences(long_source)
        for p in parts:
            if len(bullets) >= 6:
                break
            if len(p) > 50:
                bullets.append(p)

    # Cap to 10
    return bullets[:10]


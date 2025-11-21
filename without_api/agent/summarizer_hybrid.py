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
    # scikit-learn based TF-IDF sentence scoring
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
    # Very simple sentence splitter (fallback)
    if SPACY_OK:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            pass
    # fallback regex split
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extractive_sentences(text: str, n: int = 8, method: str = "tfidf") -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    if method == "textrank":
        # Try pytextrank if available
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

    # Default: TF-IDF sentence scoring
    if SKLEARN_OK:
        try:
            vect = TfidfVectorizer(stop_words="english").fit_transform(sentences)
            sim = linear_kernel(vect, vect)
            # Score sentences by sum of similarities
            scores = sim.sum(axis=1)
            ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
            top = [sentences[i] for i in ranked_idx[:n]]
            return top
        except Exception:
            pass

    # Fallback: return first n non-empty sentences
    return sentences[:n]


def deduplicate_and_clean(sentences: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


def categorize_by_headings(sentences: List[str], text: str) -> Dict[str, str]:
    # Simple keyword-based categorization
    buckets = {
        "purpose": [],
        "definitions": [],
        "eligibility": [],
        "responsibilities": [],
        "payments": [],
        "penalties": [],
        "record_keeping": [],
    }

    keywords = {
        "purpose": ["purpose", "object", "aim", "this Act"],
        "definitions": ["means", "interpretation", "definition", "defined"],
        "eligibility": ["eligible", "entitled", "eligibility", "claimant"],
        "responsibilities": ["must", "shall", "responsible", "Secretary of State", "duty"],
        "payments": ["allowance", "payment", "amount", "rate", "entitlement"],
        "penalties": ["penalty", "liable", "sanction", "forfeit"],
        "record_keeping": ["record", "retain", "evidence", "report", "reporting"],
    }

    for s in sentences:
        low = s.lower()
        assigned = False
        for k, kws in keywords.items():
            for kw in kws:
                if kw in low:
                    buckets[k].append(s)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            # place in purpose by default
            buckets["purpose"].append(s)

    # Join lists to single string for each section
    return {k: "\n".join(v) for k, v in buckets.items()}


def abstractive_refine(sentences: List[str], model_name: Optional[str] = None) -> str:
    # Use transformers summarization pipeline if available
    if not sentences:
        return ""

    text = "\n".join(sentences)
    if TRANSFORMERS_OK:
        try:
            model = model_name or "facebook/bart-large-cnn"
            summariser = pipeline("summarization", model=model)
            # models often have input length limits; we keep a reasonably small chunk
            out = summariser(text, max_length=150, min_length=40, do_sample=False)
            if out and isinstance(out, list):
                return out[0].get("summary_text", "").strip()
        except Exception:
            pass

    # Fallback: return joined extractive sentences as a single paragraph
    return " ".join(sentences)


def hybrid_summary(text: str, extract_n: int = 12, extract_method: str = "tfidf", abstractive_model: Optional[str] = None) -> List[str]:
    """Return a list of 5-10 concise bullet points produced by extractive selection
    followed by an optional abstractive refinement.
    """
    sentences = extractive_sentences(text, n=extract_n, method=extract_method)
    sentences = deduplicate_and_clean(sentences)
    # categorize and pick top sentences per bucket
    cats = categorize_by_headings(sentences, text)
    bullets = []
    for key in ["purpose", "definitions", "eligibility", "responsibilities", "payments", "penalties", "record_keeping"]:
        bucket_text = cats.get(key, "")
        if bucket_text:
            # refine each bucket into a short bullet
            sents = split_into_sentences(bucket_text)
            refined = abstractive_refine(sents[:3], model_name=abstractive_model)
            if refined:
                bullets.append(refined)

    # ensure we return 5-10 bullets (pad with top extractive sentences)
    if len(bullets) < 5:
        more = [s for s in sentences if s not in bullets]
        bullets.extend(more[: (5 - len(bullets)) ])

    return bullets[:10]

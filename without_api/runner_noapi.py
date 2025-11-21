import json
import os
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path so imports like `without_api.agent...` work
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from without_api.agent.extractor import extract_text_from_pdf
from without_api.agent.summarizer import extractive_summary
from without_api.agent.summarizer_hybrid import hybrid_summary
from without_api.agent.sections import extract_sections
from without_api.agent.rule_checks import run_all_checks


def build_summary(text: str, min_bullets: int = 5, max_bullets: int = 6) -> List[str]:
    """Use the hybrid summariser to produce assignment-compliant bullets."""
    bullets = [b.strip() for b in hybrid_summary(text, extract_n=15) if b.strip()]
    bullets = [" ".join(b.split()) for b in bullets]

    if len(bullets) < min_bullets:
        filler_needed = min_bullets - len(bullets)
        fallback = extractive_summary(text, n_sentences=max(min_bullets, filler_needed * 2))
        for sentence in fallback:
            cleaned = " ".join(sentence.split())
            if cleaned and cleaned not in bullets:
                bullets.append(cleaned)
            if len(bullets) >= min_bullets:
                break

    return bullets[:max_bullets]


def run_no_api_local(pdf_path: str, out_dir: str = "outputs") -> dict:
    txt = extract_text_from_pdf(pdf_path)
    Path(out_dir).mkdir(exist_ok=True)
    out_txt = Path(out_dir) / (Path(pdf_path).stem + ".txt")
    out_txt.write_text(txt, encoding="utf-8")

    summary = build_summary(txt)
    sections = extract_sections(txt)
    checks_dict = run_all_checks(sections)
    rule_checks = list(checks_dict.values())

    report = {
        "source": str(pdf_path),
        "summary": summary,
        "sections": sections,
        "rule_checks": rule_checks,
    }
    out_json = Path(out_dir) / (Path(pdf_path).stem + "_report.json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(pdf_path: str):
    report = run_no_api_local(pdf_path)
    print("No-API report written (without_api runner). Summary bullets:\n")
    for s in report.get("summary", []):
        print("- ", s[:300])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to PDF to analyze")
    args = parser.parse_args()
    main(args.pdf)

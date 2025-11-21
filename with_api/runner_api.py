import json
import os
import sys
import re
from pathlib import Path
from typing import List

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.extractor import extract_text_from_pdf
# CHANGED: Force import from the local 'with_api' folder
from with_api.agent.summarizer_hybrid import hybrid_summary 
from with_api.agent import api_client
from with_api.rule_checks_api import run_checks_with_gemini
from with_api.agent.summary_refiner import refine_summary_with_gemini
# Import the regex extractor from your existing file
from without_api.agent.sections import extract_sections as regex_extract


def clean_json_string(json_str: str) -> str:
    if not json_str: return "{}"
    s = str(json_str).strip()
    s = re.sub(r"^`{3}(json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*`{3}$", "", s)
    return s.strip()

# --- SAFETY NET: Force Split Clumped Summaries ---
def force_split_summary(summary_list: List[str]) -> List[str]:
    """If the summarizer returns one giant string, chop it up."""
    if len(summary_list) > 1:
        return summary_list
    
    if not summary_list:
        return []
        
    text = summary_list[0]
    # Split by bold headers (e.g. **Purpose:**)
    parts = re.split(r"(\*\*[A-Z][a-z]+:\*\*)", text)
    
    new_list = []
    current_header = ""
    
    for p in parts:
        if re.match(r"\*\*[A-Z][a-z]+:\*\*", p):
            current_header = p
        elif current_header:
            # We have a header and now the content
            clean_content = p.strip().strip("*").strip()
            if clean_content:
                new_list.append(f"{current_header} {clean_content}")
            current_header = ""
        elif p.strip():
            # No header, just text
            new_list.append(p.strip())
            
    return new_list if new_list else summary_list
# ---------------------------------------------------

def run_with_api_local(input_path: str, api_key: str = None, out_dir: str = "outputs", provider: str = "gemini", model: str = None) -> dict:
    Path(out_dir).mkdir(exist_ok=True)
    input_p = Path(input_path)

    # 1. Text Extraction
    if input_p.suffix.lower() == ".txt":
        print(f"📄 Reading text directly from {input_p.name}...")
        txt = input_p.read_text(encoding="utf-8")
    else:
        print(f"📄 Extracting text from PDF {input_p.name}...")
        txt = extract_text_from_pdf(str(input_p))
        (Path(out_dir) / (input_p.stem + ".txt")).write_text(txt, encoding="utf-8")

    # 2. TASK 2: Summary
    print("🤖 Generating Summary (Task 2) using local hybrid base...")
    summary_base = hybrid_summary(txt, extract_n=15, abstractive_model=None)
    
    # FORCE SPLIT if clumping happened
    summary_base = force_split_summary(summary_base)

    if provider and provider.lower() == "gemini":
        
        print(f"   ✨ Refining presentation with Gemini (Input items: {len(summary_base)})...")
        try:
            # Task 3 Sections needed for context
            sections_raw = api_client.extract_sections_with_api(txt, api_key=api_key, provider=provider, model=model)
            sections_dict = json.loads(clean_json_string(sections_raw))
            
            summary_bullets = refine_summary_with_gemini(summary_base, sections_dict, api_key=api_key, model=model)
        except Exception as e:
            print(f"   ⚠️ Refinement failed ({e}), using base summary.")
            summary_bullets = summary_base
            sections_dict = {}
    else:
        summary_bullets = summary_base
        sections_dict = {}

    # 3. TASK 3: Section Extraction (if not done)
    print("📑 Extracting Sections (Task 3)...")
    sections_dict = {}

    # Step A: Try API Extraction
    if provider and provider.lower() == "gemini":
        try:
            sections_raw = api_client.extract_sections_with_api(txt, api_key=api_key, provider=provider, model=model)
            sections_dict = json.loads(clean_json_string(sections_raw))
            print("   ✅ Parsed sections from API.")
        except Exception as e:
            print(f"   ⚠️ API extraction failed ({e}). Switching to fallback.")
            sections_dict = {}

    # Step B: Regex Safety Net (Backfill)
    # Always run this to check for missing keys or if API failed entirely
    fallback_data = regex_extract(txt)
    
    required_keys = ["definitions", "responsibilities", "eligibility", "payments", "penalties", "record_keeping"]
    
    # Ensure all keys exist
    for key in required_keys:
        # If API result is empty, missing, or too short/garbage, overwrite with Regex
        if not sections_dict.get(key) or len(sections_dict.get(key, "")) < 5:
            if fallback_data.get(key):
                print(f"   ⚙️  Backfilling '{key}' using Regex...")
                sections_dict[key] = fallback_data[key]
            else:
                # Ensure key exists even if empty to prevent Rule Check crashes
                sections_dict[key] = ""
    # 4. TASK 4: Rule Checks
    print("⚖️  Verifying Rules (Task 4)...")
    checks_api_dict = run_checks_with_gemini(txt, sections=sections_dict, model=model)
    rule_checks_list = list(checks_api_dict.values())

    # 5. Final Report
    report = {
        "source": str(input_path),
        "summary_base": summary_base, # Now distinct items
        "summary": summary_bullets,
        "sections": sections_dict,
        "rule_checks": rule_checks_list
    }

    out_json = Path(out_dir) / (Path(input_path).stem + "_api_report.json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"✅ Report saved to: {out_json}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDF or extracted .txt")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-2.0-flash") 
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    run_with_api_local(args.input, api_key=args.api_key, provider=args.provider, model=args.model)
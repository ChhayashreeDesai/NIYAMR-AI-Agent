import json
import re
from typing import List, Dict, Optional
from .gemini_client import call_gemini

def refine_summary_with_gemini(summary_base: List[str], sections: Dict[str, str], api_key: Optional[str] = None, model: Optional[str] = None) -> List[str]:
    """
    Ask Gemini to reorganize/format the `summary_base` into a clean 6-point summary.
    It allows splitting long bullets and organizing input, but forbids inventing facts.
    """
    if not summary_base:
        return []

    model = model or "gemini-2.0-flash"

    # Instruction to guide Gemini as an "Editor"
    instr = {
        "role": "You are a Legal Editor. Convert the draft data into a clean 6-point summary.",
        "input_context": "The input 'summary_base' may contain raw sentences or combined sections.",
        "task": "Create a structured summary with exactly these headers: Purpose, Definitions, Eligibility, Responsibilities, Payments, Enforcement.",
        "rules": [
            "1. Use facts from 'summary_base' AND 'sections_snippets'.",
            "2. If the input is just one long string, SPLIT it into the correct sections.",
            "3. Keep specific numbers (e.g. £217.26) intact.",
            "4. Output strictly a JSON array of strings."
        ],
        "output_format": "JSON array of strings"
    }

    # Prepare the prompt
    prompt = """STRICT INSTRUCTION JSON:
{instruction}

INPUT DRAFT (summary_base):
{bullets}

CONTEXT SNIPPETS:
{sections}

Output ONLY the JSON array of strings.
""".format(
        instruction=json.dumps(instr, indent=2),
        bullets=json.dumps(summary_base, ensure_ascii=False, indent=2),
        sections=json.dumps({k: (v[:800] if v else "") for k, v in sections.items()}, ensure_ascii=False, indent=2),
    )

    # Retry loop for robust parsing
    for attempt in range(2):
        resp = call_gemini(prompt, model=model)
        if not resp:
            continue
        
        # Clean response
        s = resp.strip()
        
        # Remove markdown code fences if present (e.g. ```json ... ```)
        if "```" in s:
            s = re.sub(r"^```[a-zA-Z]*\n?", "", s) # remove start fence
            s = s.replace("```", "").strip()      # remove end fence

        try:
            # 1. Try direct JSON parse if it looks like a list
            if s.startswith("["):
                arr = json.loads(s)
                if isinstance(arr, list) and len(arr) > 0:
                    return [str(x).strip() for x in arr]
            
            # 2. Try regex extraction if there is extra text around the JSON
            m = re.search(r"(\[.*\])", s, flags=re.DOTALL)
            if m:
                candidate = m.group(1)
                arr = json.loads(candidate)
                if isinstance(arr, list) and len(arr) > 0:
                    return [str(x).strip() for x in arr]
                    
        except Exception:
            continue

    # Fallback: Return the original base if API fails
    return summary_base
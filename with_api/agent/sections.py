import re
from typing import Dict, List
from without_api.agent.utils import clean_whitespace


DEFAULT_KEYS = ["definitions", "responsibilities", "eligibility", "payments", "penalties", "record_keeping"]


def extract_sections(text: str) -> Dict[str, str]:
    text = clean_whitespace(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    section_map = {
        "definitions": [r"Interpretation", r"Meaning of", r"interpretation"],
        "eligibility": [r"Standard allowance", r"limited capability", r"entitlement", r"claimant", r"eligible", r"eligib"],
        # RENAME: 'obligations' -> 'responsibilities' to match rule_checks.py
        "responsibilities": [r"Secretary of State must", r"must exercise", r"must not", r"shall", r"require", r"Department for Communities"],
        "payments": [r"rates of", r"amount", r"£", r"allowance", r"rate", r"amounts? of", r"uplift percentage"],
        "penalties": [r"offence", r"fraud", r"recovery", r"penalt", r"sanction", r"contraven", r"ceases to be met"],
        "record_keeping": [r"information requirement", r"evidence", r"record", r"report", r"retain"]
    }

    result = {k: "" for k in DEFAULT_KEYS}
    current_bucket = None  # Now updates for ALL sections

    for line in lines:
        assigned = False
        
        # Check all categories
        for key, patterns in section_map.items():
            if any(re.search(pat, line, re.IGNORECASE) for pat in patterns):
                result[key] += ("\n" + line)
                current_bucket = key 
                assigned = True
                break
        
        # Special case: Currency almost always implies payments
        if not assigned and "£" in line:
            result["payments"] += ("\n" + line)
            current_bucket = "payments"
            assigned = True

        # If line wasn't a header/keyword match, append it to the active bucket
        if not assigned and current_bucket:
            result[current_bucket] += ("\n" + line)

    # Final cleanup
    for k in result:
        result[k] = result[k].strip()

    return result

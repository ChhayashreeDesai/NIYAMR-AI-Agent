import json
import re
from typing import Dict, Any
from with_api.gemini_client import call_gemini

PROMPT_RULE = (
    "Assess the text against this rule: '{rule}'.\n"
    "Return STRICT JSON with keys: 'status' ('pass'/'fail'), "
    "'evidence' (quote from text), 'confidence' (0-100).\n"
    "Text: {context}"
)

RULES = {
    "rule_1_definitions": "Act must define key terms",
    "rule_2_eligibility": "Act must specify eligibility criteria",
    "rule_3_responsibilities": "Act must specify responsibilities of the administering authority",
    "rule_4_enforcement_penalties": "Act must include enforcement or penalties",
    "rule_5_payments": "Act must include payment calculation or entitlement structure",
    "rule_6_record_keeping": "Act must include record-keeping or reporting requirements",
}

def run_checks_with_gemini(text: str, sections: Dict[str, str], model: str = "gemini-2.0-flash") -> Dict[str, Any]:
    results = {}
    for rule_id, rule_desc in RULES.items():
        # Use full text context for Gemini 2.0
        context = text[:50000] 
        
        prompt = PROMPT_RULE.format(rule=rule_desc, context=context)
        resp = call_gemini(prompt, model=model)
        
        # Clean JSON
        clean_resp = resp.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean_resp)
            results[rule_id] = {
                "rule": rule_desc,
                "status": data.get("status", "fail"),
                "evidence": data.get("evidence", "No evidence found"),
                "confidence": data.get("confidence", 0)
            }
        except:
            results[rule_id] = {"rule": rule_desc, "status": "fail", "evidence": "API Error", "confidence": 0}
            
    return results

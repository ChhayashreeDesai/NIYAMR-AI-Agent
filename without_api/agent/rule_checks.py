from typing import Dict, Any


def _make_result(rule: str, status: bool, evidence: str = "", confidence: int = 75) -> Dict[str, Any]:
    return {"rule": rule, "status": "pass" if status else "fail", "evidence": evidence, "confidence": confidence}


def check_definitions(sections: Dict[str, str]) -> Dict[str, Any]:
    text = sections.get("definitions", "")
    lower = text.lower()
    present = False
    evidence = ""
    if "interpretation" in lower or "meaning of" in lower or len(text) > 60:
        present = True
        evidence = text[:400]
    else:
        evidence = "No 'Interpretation' or 'Meaning' sections found."
    confidence = 95 if present else 30
    return _make_result("Act must define key terms", present, evidence, confidence)


def check_record_keeping(sections: Dict[str, str]) -> Dict[str, Any]:
    text = sections.get("record_keeping", "")
    present = False
    evidence = ""
    if text and any(k in text.lower() for k in ["record", "report", "log", "retain", "keeping", "reporting", "evidence", "information"]):
        present = True
        evidence = text[:400]
    else:
        evidence = "No explicit record-keeping or reporting requirements detected."
    return _make_result("Act must include record-keeping or reporting requirements", present, evidence, 85 if present else 30)


def check_penalties(sections: Dict[str, str]) -> Dict[str, Any]:
    text = sections.get("penalties", "")
    present = False
    evidence = ""
    if text and (len(text) > 30 or any(k in text.lower() for k in ["penalt", "enforce", "fine", "sanction", "offence", "contraven"])):
        present = True
        evidence = text[:400]
    else:
        evidence = "No penalties/enforcement section found."
    return _make_result("Act must include enforcement or penalties", present, evidence, 90 if present else 35)


def check_eligibility(sections: Dict[str, str]) -> Dict[str, Any]:
    text = sections.get("eligibility", "")
    lower = text.lower()
    present = False
    evidence = ""
    if any(k in lower for k in ["eligib", "eligible", "claimant", "entitlement"]) or len(text) > 60:
        present = True
        evidence = text[:400]
    else:
        evidence = "No clear eligibility criteria detected."
    return _make_result("Act must specify eligibility criteria", present, evidence, 90 if present else 35)


def check_payments(sections: Dict[str, str]) -> Dict[str, Any]:
    text = sections.get("payments", "")
    lower = text.lower()
    present = False
    evidence = ""
    if "£" in text or any(k in lower for k in ["payment", "entitlement", "amount", "rate", "calculation", "allowance"]):
        present = True
        evidence = text[:400]
    else:
        evidence = "No payment calculation or entitlement structure detected."
    return _make_result("Act must include payment calculation or entitlement structure", present, evidence, 90 if present else 35)


def check_responsibilities(sections: Dict[str, str]) -> Dict[str, Any]:
    text = sections.get("responsibilities", "")
    present = False
    evidence = ""
    if text and (len(text) > 40 or any(k in text.lower() for k in ["respons", "adminis", "authority", "duty", "obligat", "must", "shall", "exercise"])):
        present = True
        evidence = text[:400]
    else:
        evidence = "No clear responsibilities of the administering authority detected."
    return _make_result("Act must specify responsibilities of the administering authority", present, evidence, 88 if present else 30)


def run_all_checks(sections: Dict[str, str]) -> Dict[str, Dict]:
    return {
        "rule_1_definitions": check_definitions(sections),
        "rule_2_eligibility": check_eligibility(sections),
        "rule_3_responsibilities": check_responsibilities(sections),
        "rule_4_enforcement_penalties": check_penalties(sections),
        "rule_5_payments": check_payments(sections),
        "rule_6_record_keeping": check_record_keeping(sections),
    }


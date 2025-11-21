# NIYAMR-AI-Agent: Internship Assignment — Universal Credit Act Processing

Summary
-------
This repository implements two processing pipelines for legal texts (the Universal Credit Act used for testing):

- `without_api/`: deterministic local pipeline (text extraction, extractive summariser, sections extractor, rule checks).
- `with_api/`: API-enabled pipeline that uses a local deterministic base and calls an LLM (Gemini) to refine presentation (hybrid mode).

The project uses a hybrid "Architect & Editor" architecture: deterministic Python extraction and TF‑IDF based extraction provide a reliable canonical base, while Google Gemini is used for presentation and fluency.


Design goals
------------
- Produce a concise, auditable summary (5–6 bullets) and strict JSON section extraction with fixed keys.
- Keep deterministic local outputs as canonical base to avoid hallucination.
- Use Gemini only for presentation/refinement with JSON enforcement and deterministic fallbacks.

Structure
---------
- `without_api/` — local runner and agent modules; deterministic extraction and summariser.
- `with_api/` — API/hybrid runner and agent modules; includes `gemini_client.py`, `summary_refiner.py`, and a local copy of deterministic agents used as the hybrid base.
- `outputs/` — runner outputs: extracted `.txt` and `<stem>_report.json` / `<stem>_api_report.json`.

Video Walkthrough
-----------------
[ ]

Key Features
------------

1. Hybrid Summarization (Architect & Editor)

   - Architect (Local): TF‑IDF extraction and keyword clustering gather candidate sentences and organize them into topical buckets (Purpose, Eligibility, Obligations, Payments, Enforcement, etc.).
   - Editor (Gemini): Polishes the extracted data into fluent bullet points without inventing new facts.

2. Robust Section Extraction (Task 3)

   - Extracts seven canonical sections: `definitions`, `obligations`, `responsibilities`, `eligibility`, `payments`, `penalties`, and `record_keeping`.
   - Safety net: If the LLM returns malformed or missing JSON, a deterministic regex-based extractor backfills missing sections to guarantee a complete report.

3. Automated Rule Compliance (Task 4)

   - Validates the text against six legislative rules (for example, "Act must define key terms" or "Act must include penalties").
   - Each rule check returns `{rule, status, evidence, confidence}`.


Setup & Installation
--------------------

Clone the repository and enter the project directory:

```powershell
git clone <repo-url>
cd "Internship Assignment"
```

Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Required packages (examples): `google-genai` (or your preferred Gemini client), `scikit-learn`, `PyPDF2`, `python-dotenv`, `requests`.

Configuration
-------------

Set your Gemini API key in `with_api/.env` or export it in your environment:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_BASE_URL=https://<your-gemini-endpoint>
```

Usage
-----

Run the hybrid API pipeline (preferred, uses Gemini for refinement):

```powershell
& ".\.venv\Scripts\python.exe" .\with_api\runner_api.py inputs\ukpga_20250022_en.pdf --provider gemini
```

Run the deterministic local pipeline (no network):

```powershell
& ".\.venv\Scripts\python.exe" .\without_api\runner_noapi.py inputs\ukpga_20250022_en.pdf
```

Output
------

The API runner writes `outputs\<stem>_api_report.json`. The JSON report includes at minimum the following keys:

```json
{
  "source": "inputs/ukpga_20250022_en.pdf",
  "summary": [
    "**Purpose:** To alter rates of the standard allowance...",
    "**Eligibility:** Defines 'pre-2026 claimant' regulations..."
  ],
  "sections": {
    "definitions": "Meaning of 'pre-2026 claimant'...",
    "eligibility": "Standard allowance... limited capability...",
    "payments": "£217.26... uplift percentage..."
  },
  "rule_checks": [
    {
      "rule": "Act must define key terms",
      "status": "pass",
      "evidence": "Section 2 defines 'pre-2026 claimant'...",
      "confidence": 100
    },
    {
      "rule": "Act must include enforcement or penalties",
      "status": "fail",
      "evidence": "No specific section regarding penalties was found...",
      "confidence": 100
    }
  ]
}
```

Design Decisions
----------------

- Hybrid approach ensures determinism for facts (numbers, quoted definitions) while using an LLM only for style and organization.
- The safety-net ensures downstream rule checks are never starved of data: deterministic fallbacks run whenever API responses are missing or malformed.

Author
------

Chhayashree Desai


import os
import re
import json
import urllib.request
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

app = FastAPI(title="BrandGuard AI", version="2.1.0")

# Setup Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HAS_GEMINI = False

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        HAS_GEMINI = True
    except Exception as e:
        print(f"Warning: Gemini init failed: {e}")

# In-Memory Dynamic Brand Guidelines (User Editable)
CURRENT_BRAND = {
    "brand_name": "AcmeCloud",
    "website_url": "https://acmecloud.io",
    "tone": "Professional, authoritative, customer-focused, humble yet innovative.",
    "banned_words": ["guaranteed", "100% risk-free", "unbeatable", "cheapest in the world", "bulletproof"],
    "approved_facts": [
        "AcmeCloud delivers 99.95% uptime SLA.",
        "Over 10,000 engineering teams use AcmeCloud across 45 countries.",
        "SOC2 Type II certified and GDPR compliant.",
        "Offers 24/7 priority customer support for Enterprise tier only."
    ],
    "mandatory_disclaimers": [
        "Terms and SLA availability subject to contract terms."
    ]
}

AUDIT_HISTORY = []

# --- Data Schemas ---
class Finding(BaseModel):
    id: str
    issue: str
    severity: str
    category: str
    evidence: str
    highlight_text: str
    suggested_fix: str

class AuditRequest(BaseModel):
    content: str
    use_mock: bool = False

class AuditResponse(BaseModel):
    id: str
    timestamp: str
    brand_audited: str
    original_text: str
    findings: List[Finding]
    readability_score: str
    status: str
    execution_source: str

class ReviewDecisionRequest(BaseModel):
    audit_id: str
    decision: str
    reviewer_notes: Optional[str] = ""

class ExtractRulesRequest(BaseModel):
    brand_or_url: str

class UpdateRulesRequest(BaseModel):
    brand_name: str
    tone: str
    banned_words: List[str]
    approved_facts: List[str]
    mandatory_disclaimers: List[str]

# --- Deterministic Rule Engine ---
def run_deterministic_checks(text: str) -> List[Finding]:
    findings = []
    text_lower = text.lower()
    
    for word in CURRENT_BRAND["banned_words"]:
        if word.lower() in text_lower:
            findings.append(Finding(
                id=f"rule-banned-{len(findings)+1}",
                issue=f"Prohibited claim '{word}' detected.",
                severity="High",
                category="Prohibited Claim",
                evidence=f"{CURRENT_BRAND['brand_name']} Policy: Words like '{word}' violate guidelines.",
                highlight_text=word,
                suggested_fix=f"Replace '{word}' with verifiable metrics."
            ))
            
    for disclaimer in CURRENT_BRAND["mandatory_disclaimers"]:
        if "sla" in text_lower or "uptime" in text_lower or "terms" in text_lower:
            if disclaimer.lower() not in text_lower:
                findings.append(Finding(
                    id=f"rule-disclaimer-{len(findings)+1}",
                    issue="Missing mandatory brand disclaimer.",
                    severity="Medium",
                    category="Missing Disclaimer",
                    evidence=f"Mandatory requirement: Include '{disclaimer}'",
                    highlight_text="SLA" if "sla" in text_lower else "claim",
                    suggested_fix=f"Append footer: '{disclaimer}'"
                ))
                break
            
    return findings

# --- AI Semantic Check via Gemini ---
def run_ai_semantic_check(text: str) -> List[Finding]:
    if not HAS_GEMINI:
        return []
        
    prompt = f"""
You are the BrandGuard AI Compliance Auditor. 
Analyze the submitted marketing content against the brand's approved facts and tone guidelines.

Brand Name: {CURRENT_BRAND['brand_name']}
Approved Tone: {CURRENT_BRAND['tone']}
Approved Facts:
{json.dumps(CURRENT_BRAND['approved_facts'], indent=2)}

Submitted Text to Audit:
\"\"\"{text}\"\"\"

Identify any:
1. Inaccurate or unverified factual claims contradicting the approved facts.
2. Wild exaggerations or unsupported superlatives.
3. Inconsistent brand tone.

Return a STRICT JSON list of objects matching this exact format:
[
  {{
    "id": "ai-1",
    "issue": "Brief description of problem",
    "severity": "High",
    "category": "Factual Inaccuracy",
    "evidence": "Quote specific brand fact or tone guideline violated",
    "highlight_text": "Exact short snippet from submitted text that must be corrected",
    "suggested_fix": "Professional rewrite of that snippet"
  }}
]
If there are no violations, return empty list []. Return ONLY valid JSON.
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        findings = []
        for idx, item in enumerate(data):
            findings.append(Finding(
                id=f"ai-{idx+1}",
                issue=item.get("issue", "Brand compliance anomaly"),
                severity=item.get("severity", "Medium"),
                category=item.get("category", "Tone Inconsistency"),
                evidence=item.get("evidence", "Brand knowledge base reference"),
                highlight_text=item.get("highlight_text", ""),
                suggested_fix=item.get("suggested_fix", "")
            ))
        return findings
    except Exception as e:
        print(f"Gemini evaluation error: {e}")
        return []

# --- API Endpoints ---
@app.get("/api/guidelines")
def get_guidelines():
    return CURRENT_BRAND

@app.post("/api/guidelines")
def update_guidelines(req: UpdateRulesRequest):
    global CURRENT_BRAND
    CURRENT_BRAND.update(req.model_dump())
    return {"status": "success", "guidelines": CURRENT_BRAND}

@app.post("/api/guidelines/auto-extract")
def auto_extract_guidelines(req: ExtractRulesRequest):
    """Auto-extracts brand tone, facts, and banned words from a website URL or brand name"""
    brand_input = req.brand_or_url.strip()
    
    if not brand_input:
        raise HTTPException(status_code=400, detail="Brand name or Website URL required")
        
    if HAS_GEMINI:
        try:
            prompt = f"""
Analyze the website domain or brand name '{brand_input}'.
Act as a Brand Compliance Director. Auto-generate the official Brand Compliance Profile for this company.

Return a STRICT JSON object in this format:
{{
  "brand_name": "Official brand name",
  "website_url": "{brand_input}",
  "tone": "2-3 sentences describing the brand's exact tone of voice and writing style",
  "banned_words": ["3 to 5 prohibited superlatives or risky claims for this industry (e.g. guaranteed, risk-free)"],
  "approved_facts": ["3 to 4 realistic, verified facts/stats about this company's products/services"],
  "mandatory_disclaimers": ["1 standard required legal disclaimer for their marketing copy"]
}}
Return ONLY the JSON. No Markdown formatting, no explanation.
"""
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            extracted = json.loads(response.text)
            
            global CURRENT_BRAND
            CURRENT_BRAND = extracted
            return {"status": "success", "source": "Gemini AI Extraction", "guidelines": CURRENT_BRAND}
        except Exception as e:
            print(f"Extraction error: {e}")

    # Fallback template extraction (if offline or no API key)
    clean_name = brand_input.replace("https://", "").replace("http://", "").split(".")[0].capitalize()
    fallback_rules = {
        "brand_name": clean_name,
        "website_url": brand_input,
        "tone": f"Authoritative, transparent, and compliant with {clean_name} industry standards.",
        "banned_words": ["100% guaranteed", "risk-free", "unbeatable", "cheapest in market"],
        "approved_facts": [
            f"{clean_name} is an enterprise-grade service provider.",
            f"Official features and pricing are verified on {brand_input}.",
            "Customer data is strictly encrypted at rest and in transit."
        ],
        "mandatory_disclaimers": [
            f"Results may vary. Visit {brand_input} for full terms and conditions."
        ]
    }
    CURRENT_BRAND = fallback_rules
    return {"status": "success", "source": "Rule Template Engine", "guidelines": CURRENT_BRAND}

@app.post("/api/audit", response_model=AuditResponse)
def audit_content(req: AuditRequest):
    findings: List[Finding] = []
    engine_used = "Deterministic Engine (Mock Fallback)"
    
    rule_findings = run_deterministic_checks(req.content)
    findings.extend(rule_findings)
    
    if not req.use_mock and HAS_GEMINI:
        ai_findings = run_ai_semantic_check(req.content)
        findings.extend(ai_findings)
        engine_used = f"Hybrid (Rules + Gemini 1.5) for {CURRENT_BRAND['brand_name']}"
    else:
        if req.use_mock or not HAS_GEMINI:
            if "100%" in req.content or "instant" in req.content.lower():
                findings.append(Finding(
                    id=f"mock-fact-{len(findings)+1}",
                    issue="Unverified statistical or absolute claim.",
                    severity="High",
                    category="Factual Inaccuracy",
                    evidence=f"{CURRENT_BRAND['brand_name']} approved facts require verified empirical proof.",
                    highlight_text="100%",
                    suggested_fix="statistically verified performance"
                ))

    audit_entry = AuditResponse(
        id=f"AUD-{int(datetime.now().timestamp())}",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        brand_audited=CURRENT_BRAND["brand_name"],
        original_text=req.content,
        findings=findings,
        readability_score="Compliant with review" if findings else "Ready to Publish",
        status="Review Required" if findings else "Auto-Approved",
        execution_source=engine_used
    )
    
    AUDIT_HISTORY.insert(0, audit_entry.model_dump())
    return audit_entry

@app.get("/api/history")
def get_audit_history():
    return AUDIT_HISTORY

@app.post("/api/decision")
def record_decision(req: ReviewDecisionRequest):
    for item in AUDIT_HISTORY:
        if item["id"] == req.audit_id:
            item["status"] = req.decision
            item["reviewer_notes"] = req.reviewer_notes
            item["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {"status": "success", "updated_record": item}
    raise HTTPException(status_code=404, detail="Audit ID not found")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return {"status": "error", "message": "static/index.html not found"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=True)
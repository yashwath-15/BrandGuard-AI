import os
import json
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BrandGuard AI", version="2.2.0")

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

# In-Memory Dynamic Brand Guidelines
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
    ],
    "sample_flawed_draft": "AcmeCloud guarantees 100% risk-free cloud hosting for every business. We are the cheapest in the world and promise instant zero-downtime forever with our bulletproof architecture. Over 1,000,000 companies switch every month!"
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

# --- Deterministic Rule Engine ---
def run_deterministic_checks(text: str) -> List[Finding]:
    findings = []
    text_lower = text.lower()
    
    for word in CURRENT_BRAND.get("banned_words", []):
        if word.lower() in text_lower:
            findings.append(Finding(
                id=f"rule-banned-{len(findings)+1}",
                issue=f"Prohibited absolute claim '{word}' detected.",
                severity="High",
                category="Prohibited Claim",
                evidence=f"{CURRENT_BRAND['brand_name']} Compliance Policy: Claims like '{word}' violate advertising standards.",
                highlight_text=word,
                suggested_fix=f"Replace '{word}' with verifiable metrics."
            ))
            
    for disclaimer in CURRENT_BRAND.get("mandatory_disclaimers", []):
        if any(term in text_lower for term in ["sla", "uptime", "free", "guarantee", "risk-free", "growth"]):
            if disclaimer.lower() not in text_lower:
                findings.append(Finding(
                    id=f"rule-disclaimer-{len(findings)+1}",
                    issue="Missing mandatory brand disclaimer.",
                    severity="Medium",
                    category="Missing Disclaimer",
                    evidence=f"Mandatory requirement: Include '{disclaimer}'",
                    highlight_text="guarantee" if "guarantee" in text_lower else "service",
                    suggested_fix=f"Append footer disclaimer: '{disclaimer}'"
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
{json.dumps(CURRENT_BRAND.get('approved_facts', []), indent=2)}

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

@app.post("/api/guidelines/auto-extract")
def auto_extract_guidelines(req: ExtractRulesRequest):
    brand_input = req.brand_or_url.strip()
    if not brand_input:
        raise HTTPException(status_code=400, detail="Brand name or Website URL required")
        
    clean_name = brand_input.replace("https://", "").replace("http://", "").split(".")[0].capitalize()

    # Brand-specific curated fallbacks
    fallbacks = {
        "Instagram": {
            "brand_name": "Instagram",
            "website_url": "https://instagram.com",
            "tone": "Vibrant, community-first, visual, and inspiring while adhering to Meta community guidelines.",
            "banned_words": ["100% viral", "guaranteed followers", "algorithm hack", "unbeatable growth"],
            "approved_facts": [
                "Instagram has over 2 billion monthly active users worldwide.",
                "Content distribution is governed by personalized engagement and community safety algorithms.",
                "Meta verified badges require verified government identification."
            ],
            "mandatory_disclaimers": ["Reach and engagement vary based on audience and content. Subject to Meta Terms."],
            "sample_flawed_draft": "Instagram guarantees 100% viral reach and instant followers for every new creator account! Use our secret algorithm hack to gain unbeatable growth with zero effort."
        },
        "Stripe": {
            "brand_name": "Stripe",
            "website_url": "https://stripe.com",
            "tone": "Developer-centric, ultra-precise, understated, and financially compliant.",
            "banned_words": ["100% free transactions", "instant wealth", "zero fee forever", "risk-free payouts"],
            "approved_facts": [
                "Stripe maintains 99.999% historical uptime for its core processing APIs.",
                "Supports payments in 135+ currencies across 46+ countries.",
                "Certified PCI Service Provider Level 1."
            ],
            "mandatory_disclaimers": ["Standard card processing fees apply per the Stripe Services Agreement."],
            "sample_flawed_draft": "Stripe guarantees 100% free transactions for every startup in the world! Sign up today for instant zero fee forever and unbeatable payment speed."
        },
        "Spotify": {
            "brand_name": "Spotify",
            "website_url": "https://spotify.com",
            "tone": "Playful, music-driven, culturally connected, and respectful of artist royalties.",
            "banned_words": ["unlimited free royalties", "guaranteed billboard hit", "100% free music forever"],
            "approved_facts": [
                "Spotify offers over 100 million tracks and 5 million podcasts.",
                "Over 600 million active listeners across 180+ global markets.",
                "Royalty pools are distributed proportionally based on total stream shares."
            ],
            "mandatory_disclaimers": ["Subscription and catalog availability subject to regional licensing terms."],
            "sample_flawed_draft": "Spotify guarantees a billboard hit for every artist who uploads music this week! Stream unlimited free music forever with our unbeatable audio quality."
        }
    }

    # Use AI extraction if key exists
    if HAS_GEMINI:
        try:
            prompt = f"""
Analyze the website domain or brand name '{brand_input}'.
Act as a Brand Compliance Director. Auto-generate the official Brand Compliance Profile for this company AND create a brand-specific sample draft with intentional violations.

Return a STRICT JSON object in this format:
{{
  "brand_name": "Official brand name",
  "website_url": "{brand_input}",
  "tone": "2-3 sentences describing the brand's exact tone of voice and writing style",
  "banned_words": ["3 to 5 prohibited superlatives or risky claims for this industry (e.g. guaranteed, risk-free)"],
  "approved_facts": ["3 to 4 realistic, verified facts/stats about this company's products/services"],
  "mandatory_disclaimers": ["1 standard required legal disclaimer for their marketing copy"],
  "sample_flawed_draft": "A realistic 2-3 sentence marketing copy for {brand_input} that intentionally uses 2 of the banned words or false stats so our audit tool can detect them."
}}
Return ONLY valid JSON.
"""
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            extracted = json.loads(response.text)
            
            global CURRENT_BRAND
            CURRENT_BRAND = extracted
            return {"status": "success", "source": "Gemini AI Extraction", "guidelines": CURRENT_BRAND}
        except Exception as e:
            print(f"Gemini auto-extract fallback triggered: {e}")

    # Fallback to predefined or generated rules
    global CURRENT_BRAND
    if clean_name in fallbacks:
        CURRENT_BRAND = fallbacks[clean_name]
    else:
        CURRENT_BRAND = {
            "brand_name": clean_name,
            "website_url": brand_input,
            "tone": f"Authoritative, transparent, and compliant with {clean_name} industry standards.",
            "banned_words": ["100% guaranteed", "risk-free", "unbeatable", "cheapest in the world"],
            "approved_facts": [
                f"{clean_name} is an enterprise-grade service provider.",
                f"Official product features and pricing are verified on {brand_input}.",
                "Customer data is strictly encrypted at rest and in transit."
            ],
            "mandatory_disclaimers": [f"Results may vary. Visit {brand_input} for full terms and conditions."],
            "sample_flawed_draft": f"{clean_name} guarantees 100% risk-free performance for every client. We are the cheapest in the world and promise instant zero-downtime forever with our unbeatable service!"
        }

    return {"status": "success", "source": "Curated Compliance Template", "guidelines": CURRENT_BRAND}

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
            if any(term in req.content.lower() for term in ["100%", "instant", "unbeatable"]):
                findings.append(Finding(
                    id=f"mock-fact-{len(findings)+1}",
                    issue=f"Unverified performance claim for {CURRENT_BRAND['brand_name']}.",
                    severity="High",
                    category="Factual Inaccuracy",
                    evidence=f"{CURRENT_BRAND['brand_name']} approved facts require verified empirical proof.",
                    highlight_text="100%" if "100%" in req.content else "instant",
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

# --- EMBEDDED FRONTEND APPLICATION ---
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BrandGuard AI — Content Audit & Governance</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    body { background-color: #080E1E; color: #E0E1DD; font-family: system-ui, -apple-system, sans-serif; }
    .card { background: #121B33; border: 1px solid #243354; }
    .badge-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-low { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect } = React;

    function App() {
      const [content, setContent] = useState("");
      const [useMock, setUseMock] = useState(false);
      const [loading, setLoading] = useState(false);
      const [extractLoading, setExtractLoading] = useState(false);
      const [websiteInput, setWebsiteInput] = useState("Instagram");
      const [auditResult, setAuditResult] = useState(null);
      const [history, setHistory] = useState([]);
      const [reviewerNotes, setReviewerNotes] = useState("");
      const [activeTab, setActiveTab] = useState("editor");
      const [guidelines, setGuidelines] = useState(null);

      useEffect(() => {
        fetchInitialData();
      }, []);

      const fetchInitialData = async () => {
        try {
          const res = await fetch("/api/guidelines");
          const data = await res.json();
          setGuidelines(data);
          if (data.sample_flawed_draft) {
            setContent(data.sample_flawed_draft);
          }
        } catch(e) { console.error(e); }
        
        try {
          const hRes = await fetch("/api/history");
          const hData = await hRes.json();
          setHistory(hData);
        } catch(e) { console.error(e); }
      };

      const handleAutoExtract = async (target) => {
        const query = target || websiteInput;
        if (!query.trim()) return;
        setExtractLoading(true);
        try {
          const res = await fetch("/api/guidelines/auto-extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ brand_or_url: query })
          });
          const data = await res.json();
          setGuidelines(data.guidelines);
          
          // Automatically update the editor with this brand's custom flawed draft
          if (data.guidelines.sample_flawed_draft) {
            setContent(data.guidelines.sample_flawed_draft);
          }
          
          // Clear previous audit result so user sees fresh state
          setAuditResult(null);
          setActiveTab("editor");
        } catch (e) {
          alert("Auto-extract failed: " + e.message);
        } finally {
          setExtractLoading(false);
        }
      };

      const handleRunAudit = async () => {
        setLoading(true);
        try {
          const res = await fetch("/api/audit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content, use_mock: useMock })
          });
          const data = await res.json();
          setAuditResult(data);
          
          const hRes = await fetch("/api/history");
          const hData = await hRes.json();
          setHistory(hData);
        } catch (e) {
          alert("Audit failed: " + e.message);
        } finally {
          setLoading(false);
        }
      };

      const handleApplyFix = (finding) => {
        if (!finding.highlight_text || !finding.suggested_fix) return;
        const newText = content.replace(finding.highlight_text, finding.suggested_fix);
        setContent(newText);
        setAuditResult(prev => ({
          ...prev,
          findings: prev.findings.filter(f => f.id !== finding.id)
        }));
      };

      const handleDecision = async (decision) => {
        if (!auditResult) return;
        try {
          await fetch("/api/decision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              audit_id: auditResult.id,
              decision,
              reviewer_notes: reviewerNotes
            })
          });
          setAuditResult(prev => ({ ...prev, status: decision }));
          const hRes = await fetch("/api/history");
          const hData = await hRes.json();
          setHistory(hData);
          alert(`Document successfully marked as ${decision}!`);
        } catch (e) {
          alert("Decision save failed: " + e.message);
        }
      };

      return (
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-[#243354] px-6 py-4 flex items-center justify-between bg-[#080E1E]">
            <div className="flex items-center space-x-3">
              <div className="h-9 w-9 bg-emerald-500 rounded-lg flex items-center justify-center font-black text-slate-950 text-xl">B</div>
              <div>
                <div className="flex items-center space-x-2">
                  <h1 className="text-xl font-bold tracking-tight text-white">BrandGuard AI</h1>
                  {guidelines && (
                    <span className="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 px-2.5 py-0.5 rounded-full font-semibold">
                      Active Brand: {guidelines.brand_name}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400">Pre-Publish Content Governance & Multi-Brand Compliance</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <label className="flex items-center space-x-2 text-xs bg-[#121B33] px-3 py-1.5 rounded-full border border-slate-700 cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={useMock} 
                  onChange={(e) => setUseMock(e.target.checked)} 
                  className="rounded text-emerald-500" 
                />
                <span className={useMock ? "text-amber-400 font-medium" : "text-slate-300"}>
                  {useMock ? "Safe Mock Mode" : "Live Gemini AI"}
                </span>
              </label>

              <button 
                onClick={() => {
                  if (guidelines && guidelines.sample_flawed_draft) {
                    setContent(guidelines.sample_flawed_draft);
                  }
                }}
                className="text-xs bg-slate-800 hover:bg-slate-700 text-emerald-400 border border-emerald-500/30 px-3 py-1.5 rounded"
              >
                Reset Brand Demo Text
              </button>
            </div>
          </header>

          <div className="border-b border-[#243354] px-6 flex space-x-6 bg-[#121B33]/50 text-sm">
            <button 
              onClick={() => setActiveTab("editor")} 
              className={`py-3 border-b-2 font-medium ${activeTab === 'editor' ? 'border-emerald-400 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            >
              Audit Studio & Editor
            </button>
            <button 
              onClick={() => setActiveTab("guidelines")} 
              className={`py-3 border-b-2 font-medium ${activeTab === 'guidelines' ? 'border-emerald-400 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            >
              Brand Knowledge Base & Auto-Extract 🌐
            </button>
            <button 
              onClick={() => setActiveTab("history")} 
              className={`py-3 border-b-2 font-medium ${activeTab === 'history' ? 'border-emerald-400 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            >
              Audit Trail & Approval Logs ({history.length})
            </button>
          </div>

          <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
            {activeTab === "editor" && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-6 flex flex-col space-y-4">
                  <div className="card p-5 rounded-xl shadow-lg flex-1 flex flex-col">
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                        CONTENT EDITOR (AUDITING AGAINST: <font color="#34D399">{guidelines ? guidelines.brand_name.toUpperCase() : "ACMECLOUD"}</font>)
                      </span>
                      <span className="text-xs text-slate-400">{content.length} chars</span>
                    </div>

                    <textarea
                      className="w-full flex-1 bg-[#080E1E] border border-slate-700 rounded-lg p-4 text-sm text-slate-100 focus:outline-none focus:border-emerald-500 font-mono min-h-[300px]"
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      placeholder="Type or paste marketing copy here..."
                    />

                    <div className="mt-4 flex items-center justify-between">
                      <button
                        onClick={handleRunAudit}
                        disabled={loading || !content.trim()}
                        className="bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-slate-950 font-bold px-6 py-2.5 rounded-lg transition"
                      >
                        {loading ? "Auditing Content..." : "Run BrandGuard Audit"}
                      </button>

                      {auditResult && (
                        <span className="text-xs text-slate-400">
                          Engine: <strong className="text-emerald-400">{auditResult.execution_source}</strong>
                        </span>
                      )}
                    </div>
                  </div>

                  {auditResult && (
                    <div className="card p-4 rounded-xl border border-slate-700">
                      <h3 className="text-sm font-bold text-slate-200 mb-2">Reviewer Human-in-the-Loop Sign-off</h3>
                      <input 
                        type="text" 
                        value={reviewerNotes} 
                        onChange={(e) => setReviewerNotes(e.target.value)} 
                        placeholder="Reviewer sign-off remarks..."
                        className="w-full bg-[#080E1E] border border-slate-700 text-xs text-slate-200 rounded p-2 mb-3"
                      />
                      <div className="flex space-x-3">
                        <button 
                          onClick={() => handleDecision("Approved")}
                          className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 rounded text-xs transition"
                        >
                          Approve for Publication
                        </button>
                        <button 
                          onClick={() => handleDecision("Changes Requested")}
                          className="flex-1 bg-rose-600 hover:bg-rose-700 text-white font-semibold py-2 rounded text-xs transition"
                        >
                          Request Changes
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="lg:col-span-6 flex flex-col space-y-4">
                  <div className="card p-5 rounded-xl shadow-lg flex-1">
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                        Findings & Evidence
                      </h2>
                      {auditResult && (
                        <span className={`text-xs px-2.5 py-1 rounded-full font-bold ${
                          auditResult.findings.length === 0 ? "badge-low text-emerald-400 border-emerald-500" : "badge-high"
                        }`}>
                          {auditResult.findings.length} Issue(s)
                        </span>
                      )}
                    </div>

                    {!auditResult && !loading && (
                      <div className="h-64 flex flex-col items-center justify-center text-slate-500 text-center p-6">
                        <p className="text-sm font-semibold text-slate-300">Ready to audit.</p>
                        <p className="text-xs text-slate-500 mt-1">Click "Run BrandGuard Audit" or switch active brands in the Knowledge Base tab.</p>
                      </div>
                    )}

                    {loading && (
                      <div className="h-64 flex flex-col items-center justify-center text-slate-400 space-y-3">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-400"></div>
                        <p className="text-xs">Checking rules and querying Gemini...</p>
                      </div>
                    )}

                    {auditResult && (
                      <div className="space-y-4 max-h-[550px] overflow-y-auto pr-1">
                        {auditResult.findings.length === 0 ? (
                          <div className="p-4 rounded-lg bg-emerald-950/30 border border-emerald-500/40 text-emerald-300 text-sm">
                            Zero violations detected! Content complies with all {auditResult.brand_audited} guidelines.
                          </div>
                        ) : (
                          auditResult.findings.map((item) => (
                            <div key={item.id} className="p-4 rounded-lg bg-[#080E1E] border border-slate-800 space-y-2">
                              <span className={`text-xs px-2 py-0.5 rounded font-bold uppercase tracking-wider ${
                                item.severity === 'High' ? 'badge-high' : item.severity === 'Medium' ? 'badge-medium' : 'badge-low'
                              }`}>
                                {item.severity} - {item.category}
                              </span>

                              <p className="text-sm font-semibold text-slate-200">{item.issue}</p>

                              <div className="text-xs text-slate-400 bg-slate-900/80 p-2.5 rounded border-l-2 border-amber-500">
                                <strong className="text-slate-300">Evidence Link: </strong> {item.evidence}
                              </div>

                              {item.highlight_text && (
                                <div className="text-xs text-rose-300 bg-rose-950/20 p-2 rounded">
                                  <strong>Flagged Snippet: </strong> "{item.highlight_text}"
                                </div>
                              )}

                              {item.suggested_fix && (
                                <div className="mt-2 pt-2 border-t border-slate-800 flex items-center justify-between">
                                  <div className="text-xs text-emerald-300">
                                    <strong>Suggested Fix: </strong> "{item.suggested_fix}"
                                  </div>
                                  <button
                                    onClick={() => handleApplyFix(item)}
                                    className="text-xs bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/50 px-2.5 py-1 rounded transition"
                                  >
                                    Apply Rewrite
                                  </button>
                                </div>
                              )}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === "guidelines" && (
              <div className="space-y-6">
                <div className="card p-6 rounded-xl border border-emerald-500/40 bg-gradient-to-r from-[#121B33] to-[#0A1128]">
                  <h3 className="text-lg font-bold text-white mb-1 flex items-center space-x-2">
                    <span>🌐 Auto-Fill Brand Rules & Sample Post from Website</span>
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Enter any company domain or brand name. BrandGuard AI will extract brand rules AND generate a custom marketing draft for that exact brand!
                  </p>

                  <div className="flex flex-col sm:flex-row gap-3">
                    <input
                      type="text"
                      value={websiteInput}
                      onChange={(e) => setWebsiteInput(e.target.value)}
                      placeholder="e.g. Instagram, Stripe, Spotify, Tesla, Nike"
                      className="flex-1 bg-[#080E1E] border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500 font-mono"
                    />
                    <button
                      onClick={() => handleAutoExtract()}
                      disabled={extractLoading}
                      className="bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-slate-950 font-bold px-6 py-2.5 rounded-lg text-sm transition"
                    >
                      {extractLoading ? "Extracting Rules & Draft..." : "Auto-Fill Brand Rules ✨"}
                    </button>
                  </div>

                  <div className="mt-3 flex items-center space-x-2 text-xs text-slate-400">
                    <span>Try quick brand presets:</span>
                    <button onClick={() => { setWebsiteInput("Instagram"); handleAutoExtract("Instagram"); }} className="underline hover:text-emerald-400">Instagram</button>
                    <span>•</span>
                    <button onClick={() => { setWebsiteInput("Stripe"); handleAutoExtract("Stripe"); }} className="underline hover:text-emerald-400">Stripe</button>
                    <span>•</span>
                    <button onClick={() => { setWebsiteInput("Spotify"); handleAutoExtract("Spotify"); }} className="underline hover:text-emerald-400">Spotify</button>
                  </div>
                </div>

                {guidelines && (
                  <div className="card p-6 rounded-xl space-y-6">
                    <div>
                      <h3 className="text-base font-bold text-white">Active Compliance Profile: {guidelines.brand_name}</h3>
                      <p className="text-xs text-slate-400">Tone: {guidelines.tone}</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                      <div className="bg-[#080E1E] p-4 rounded-lg border border-slate-800">
                        <h4 className="font-semibold text-emerald-400 mb-2">Approved Facts & Truth Cards</h4>
                        <ul className="list-disc list-inside space-y-1.5 text-slate-300 text-xs">
                          {guidelines.approved_facts.map((fact, idx) => (
                            <li key={idx}>{fact}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="bg-[#080E1E] p-4 rounded-lg border border-slate-800">
                        <h4 className="font-semibold text-rose-400 mb-2">Prohibited / Blacklisted Claims</h4>
                        <div className="flex flex-wrap gap-2 mb-4">
                          {guidelines.banned_words.map((word, idx) => (
                            <span key={idx} className="badge-high text-xs px-2.5 py-1 rounded">
                              {word}
                            </span>
                          ))}
                        </div>

                        <h4 className="font-semibold text-amber-400 mb-1">Mandatory Disclaimers</h4>
                        <p className="text-xs text-slate-300 italic">{guidelines.mandatory_disclaimers[0]}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === "history" && (
              <div className="card p-6 rounded-xl space-y-4">
                <h3 className="text-lg font-bold text-white mb-1">Traceable Audit History & Governance Trail</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-[#080E1E] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-700">
                      <tr>
                        <th className="p-3">Audit ID</th>
                        <th className="p-3">Brand</th>
                        <th className="p-3">Timestamp</th>
                        <th className="p-3">Issues</th>
                        <th className="p-3">Engine</th>
                        <th className="p-3">Status</th>
                        <th className="p-3">Notes</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {history.length === 0 ? (
                        <tr><td colSpan="7" className="p-4 text-center text-slate-500">No records found.</td></tr>
                      ) : (
                        history.map((h) => (
                          <tr key={h.id} className="hover:bg-slate-800/40">
                            <td className="p-3 font-mono font-bold text-emerald-400">{h.id}</td>
                            <td className="p-3 font-semibold text-white">{h.brand_audited || "AcmeCloud"}</td>
                            <td className="p-3">{h.timestamp}</td>
                            <td className="p-3">
                              <span className={`px-2 py-0.5 rounded ${h.findings.length > 0 ? 'badge-high' : 'badge-low'}`}>
                                {h.findings.length}
                              </span>
                            </td>
                            <td className="p-3">{h.execution_source}</td>
                            <td className="p-3">
                              <span className={`font-semibold ${h.status === 'Approved' ? 'text-emerald-400' : 'text-amber-400'}`}>
                                {h.status}
                              </span>
                            </td>
                            <td className="p-3 text-slate-400">{h.reviewer_notes || "—"}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </main>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=FRONTEND_HTML)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=True)
# app.py
# FieldCloser MVP – GPT-4o-mini integration with stub fallback
# - Auto-loads Business Context from /data
# - Absolute template/static paths
# - Detailed logging
# - Pydantic v1/v2 compatible serialization
# - /render-html accepts raw dicts and coerces to ModelResponse

import os, json, logging, traceback, functools, re, datetime, asyncio
from typing import List, Optional, Literal, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from io import BytesIO
from pathlib import Path
from pydantic import BaseModel, ValidationError
from math import isnan
from jinja2 import Environment, FileSystemLoader, select_autoescape  # add StrictUndefined if you want hard failures
from playwright.sync_api import sync_playwright

# OpenAI client (optional; if missing we fall back to stub)
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# -------------------- Logging --------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("fieldcloser")
#logger.setLevel(logging.DEBUG)
for noisy in ["httpx", "httpcore", "openai", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# -------------------- Paths ----------------------
TENANT = "demo" # Specify provider for customized catalogue
DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(DIR, "tenants")
TEMPLATES_DIR = os.path.join(BASE_DIR, TENANT, "templates")
STATIC_DIR = os.path.join(BASE_DIR, TENANT, "static")
STATIC_BASE = Path(STATIC_DIR).resolve().as_uri()
DATA_DIR = os.path.join(BASE_DIR, TENANT, "data") 
ANCHOR_HIGH_ORDER = [
    "Premium Plus", "Premium", "Standard Plus", "Standard", "Economy", "Band-Aid"
]

# -------------------- Jinja2 --------------------
env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
    # undefined=StrictUndefined,  # uncomment to raise on missing template vars
)

# -------------------- FastAPI --------------------
app = FastAPI(title="FieldCloser MVP", version="0.3.0", debug=True)

# Serve /static so CSS links like /static/style.css work
try:
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except Exception:
    pass

# -------------------- Business Context loaders --------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@functools.lru_cache(maxsize=1)
def load_business_context():
    pricing    = _load_json(os.path.join(DATA_DIR, "pricing.json"))
    warranty   = _load_json(os.path.join(DATA_DIR, "warranty.json"))
    membership = _load_json(os.path.join(DATA_DIR, "membership.json"))
    catalog    = _load_json(os.path.join(DATA_DIR, "catalog.json"))
    #config     = _load_json(os.path.join(BASE_DIR, "config.json"))
    return {"pricing": pricing, "warranty": warranty, "membership": membership, "catalog": catalog}

def _labor_rate(pricing: Dict[str, Any], postal_code: Optional[str]) -> Optional[float]:
    rate = _first_key(pricing.get("defaults", {}), ["labor_rate_per_hour_cad", "labor_rate_per_hour_usd", "labor_rate_per_hour"], None)
    # Postal override: try full, FSA (first 3), or uppercase key
    if postal_code and "regional_overrides" in pricing:
        ro = pricing["regional_overrides"]
        pc_full = postal_code.strip().upper()
        pc_fsa  = pc_full[:3]
        cand = ro.get(pc_full) or ro.get(pc_fsa)
        if isinstance(cand, dict):
            rate = _first_key(cand, ["labor_rate_per_hour_cad", "labor_rate_per_hour_usd", "labor_rate_per_hour"], rate)
        elif isinstance(cand, (int, float, str)):
            # allow direct number override
            try:
                rate = float(cand)
            except Exception:
                pass
    try:
        return float(rate) if rate is not None else None
    except Exception:
        return None

def _summarize_repairs(pricing: Dict[str, Any], currency: str, labor_rate: Optional[float], max_items: int = 3) -> List[str]:
    out = []
    repairs = pricing.get("repairs", {})
    if not isinstance(repairs, dict):
        return out
    # Pick first few stable keys (sorted by name for determinism)
    for name in sorted(repairs.keys())[:max_items]:
        item = repairs.get(name, {})
        parts = _first_key(item, ["parts_range_cad", "parts_range_usd", "parts_range"], None)
        labor_hrs = _first_key(item, ["labor_hours_range", "labor_range_hours", "labour_hours_range"], None)
        segs = []
        if parts and isinstance(parts, (list, tuple)) and len(parts) == 2:
            segs.append(f"{_fmt_currency_amount(parts[0], currency)}–{_fmt_currency_amount(parts[1], currency)} parts")
        if labor_hrs and isinstance(labor_hrs, (list, tuple)) and len(labor_hrs) == 2:
            if labor_rate:
                segs.append(f"{labor_hrs[0]}–{labor_hrs[1]} hrs labor ({_fmt_currency_amount(labor_rate, currency)}/hr)")
            else:
                segs.append(f"{labor_hrs[0]}–{labor_hrs[1]} hrs labor")
        if segs:
            # Human label from key
            label = name.replace("_", " ").replace("-", " ").title()
            out.append(f"- {label}: " + ", ".join(segs))
    return out

def _summarize_replacements(pricing: Dict[str, Any], currency: str, max_items: int = 3) -> List[str]:
    out = []
    repl = pricing.get("replacements", {})
    if not isinstance(repl, dict):
        return out
    for name in sorted(repl.keys())[:max_items]:
        item = repl.get(name, {})
        inst = _first_key(item, ["installed_range_cad", "installed_range_usd", "installed_price_range", "installed_range"], None)
        if inst and isinstance(inst, (list, tuple)) and len(inst) == 2:
            label = name.replace("_", " ").replace("-", " ").title()
            out.append(f"- {label}: {_fmt_currency_amount(inst[0], currency)}–{_fmt_currency_amount(inst[1], currency)} install")
    return out

def _summarize_refrigerant(pricing: Dict[str, Any], currency: str) -> List[str]:
    out = []
    defaults = pricing.get("defaults", {})
    for gas_key in ["refrigerant_r410a_per_lb_cad", "refrigerant_r410a_per_lb_usd", "refrigerant_r410a_per_lb",
                    "refrigerant_r22_per_lb_cad",  "refrigerant_r22_per_lb_usd",  "refrigerant_r22_per_lb"]:
        if gas_key in defaults:
            gas = "R-410A" if "410a" in gas_key.lower() else "R-22"
            out.append(f"- Refrigerant {gas}: {_fmt_currency_amount(defaults[gas_key], currency)}/lb")
    return out

def _summarize_membership(membership: Dict[str, Any], currency: str) -> str:
    """
    Accepts arbitrary plan names:
    {
      "Cross Advantage Program": {"price_cad":"$18.95","billing_cycle":"monthly","terms":"..."},
      "VIP": "Priority service, 10% off repairs"
    }
    """
    if not isinstance(membership, dict) or not membership:
        return "N/A"
    lines = []
    for plan_name, val in membership.items():
        if isinstance(val, dict):
            price = _first_key(val, ["price_cad", "price_usd", "price"], "")
            cycle = val.get("billing_cycle", "")
            terms = val.get("terms", "")
            suffix = ""
            if price and cycle:
                suffix = f" — {_fmt_currency_amount(price, currency)}/{cycle}"
            elif price:
                suffix = f" — {_fmt_currency_amount(price, currency)}"
            label = f"{plan_name}{suffix}"
            if terms:
                label += f" ({terms})"
            lines.append(f"- {label}")
        else:
            # string description
            lines.append(f"- {plan_name}: {val}")
    return "\n".join(lines) if lines else "N/A"

'''def _summarize_warranty(warranty: Dict[str, Any]) -> str:
    """
    Accepts nested dicts; we emit a compact summary:
    - Repair default: parts …, labor …
    - Standard replacement (furnace): …
    - Premium replacement (air conditioner): …
    """
    if not isinstance(warranty, dict) or not warranty:
        return "N/A"
    lines = []
    # repair default
    rep = warranty.get("repair_default")
    if isinstance(rep, dict):
        parts = rep.get("parts", "")
        labor = rep.get("labor", "")
        notes = rep.get("notes", "")
        seg = ", ".join([s for s in [parts, labor, notes] if s])
        if seg:
            lines.append(f"- Repair default: {seg}")
    # replacements (standard / premium tiers)
    for tier_key in ["standard_replacement", "premium_replacement"]:
        tier = warranty.get(tier_key)
        if isinstance(tier, dict):
            for equip, terms in tier.items():
                if isinstance(terms, dict):
                    seg = ", ".join([terms.get("parts",""), terms.get("labor",""), terms.get("heat_exchanger",""), terms.get("compressor",""), terms.get("components",""), terms.get("notes","")])
                    seg = ", ".join([s for s in [s.strip() for s in seg.split(",")] if s])
                    label = equip.replace("_"," ").replace("-"," ").title()
                    pretty_tier = tier_key.replace("_"," ").title()
                    if seg:
                        lines.append(f"- {pretty_tier} ({label}): {seg}")
    return "\n".join(lines) if lines else "N/A"
'''

def _summarize_catalog(cat: Dict[str, Any]) -> str:
    if not isinstance(cat, dict) or not cat:
        return "N/A"
    parts = []
    for bucket_key in ["repairs","iaq_addons","replacements","maintenance_services","water_heating","other_services"]:
        items = cat.get(bucket_key, [])
        if isinstance(items, list) and items:
            title = bucket_key.replace("_"," ").title()
            parts.append(f"- {title}: {', '.join(items)}")
    return "\n".join(parts) if parts else "N/A"

def format_context_strings(ctx: dict, postal_code: Optional[str]) -> Tuple[str, str, str, str, str]:
    """
    Build concise, grounded context strings from arbitrary client data:
    - pricing_text: bullets summarizing a few repairs/replacements + refrigerant + labor rate
    - warranty_text: compact summary across tiers/equipment
    - membership_text: list 'Plan — price/cycle (terms)'
    - catalog_text: category lines with comma-joined items
    """
    pricing = ctx.get("pricing", {}) or {}
    warranty = ctx.get("warranty", {}) or {}
    membership = ctx.get("membership", {}) or {}
    catalog = ctx.get("catalog", {}) or {}
    config = ctx.get("config", {}) or {}
    currency = _currency_from_pricing(pricing)
    labor = _labor_rate(pricing, postal_code)

    lines: List[str] = []

    # Labor rate headline (if available)
    if labor is not None:
        lines.append(f"- Labor rate: {_fmt_currency_amount(labor, currency)}/hr")

    # Refrigerant lines (if present)
    lines.extend(_summarize_refrigerant(pricing, currency))

    # A few representative repairs
    lines.extend(_summarize_repairs(pricing, currency, labor, max_items=3))

    # A couple of replacement ranges
    lines.extend(_summarize_replacements(pricing, currency, max_items=2))

    # Build pricing_text from LLM generated repair options
    keys_line = ""
    rep_keys = sorted((pricing.get("repairs") or {}).keys())
    if rep_keys:
        keys_line = "Repairs keys available: " + ", ".join(rep_keys)
    pricing_text = "\n".join([p for p in [keys_line] + lines if p]) if (keys_line or lines) else "N/A"
    
    membership_text = _summarize_membership(membership, currency)
    catalog_text = _summarize_catalog(catalog)

    return pricing_text, warranty, membership_text, catalog_text, config

# -------------------- Pydantic models --------------------
TierLiteral = Literal["Premium Plus", "Premium", "Standard Plus", "Standard", "Economy", "Band-Aid"]

class OptionItem(BaseModel):
    tier: TierLiteral
    name: str
    scope_bullets: List[str]
    price_range_cad: str
    warranty: List[str]
    membership_offer: str | dict
    bonuses: List[str]
    notes: str

class CustomerOptionSheet(BaseModel):
    customer_summary_markdown: str
    options: List[OptionItem]
    disclaimers: List[str]

class ObjectionItem(BaseModel):
    objection: str
    response: str

class TechSalesScript(BaseModel):
    opening: str
    anchor_low: str
    middle_move: str
    top_anchor: str
    objection_handling: List[ObjectionItem]
    close_prompt: str
    suggested_order_to_present: List[TierLiteral]
    escalate: bool = False

class ModelResponse(BaseModel):
    customer_option_sheet: CustomerOptionSheet
    tech_sales_script: TechSalesScript

class GeneratePayload(BaseModel):
    free_text_diagnosis: str
    postal_code: Optional[str] = None
    system_age: Optional[str] = None
    brand_model: Optional[str] = None
    plan_status: Optional[str | dict] = None
    urgency: Optional[Literal["low","normal","high","emergency"]] = "normal"
    constraints: Optional[str] = None
    business_override: Optional[str] = ""  # optional tiny adjustments

# -------------------- Prompts --------------------
SYSTEM_PROMPT = """
You are FieldCloser, an HVAC repair proposal and sales script generator for technicians on-site.

PRIMARY GOAL
- Convert a technician’s diagnosis into a 6-tier option sheet (Premium Plus → Premium → Standard Plus → Standard → Economy → Band-Aid) and a persuasive, field-ready sales script.

CONTEXT-GROUNDED RULES
- Use ONLY facts that appear inside the delimited CONTEXT blocks: <<PRICING>>, <<WARRANTY>>, <<MEMBERSHIP>>, <<CATALOG>>.
- Do NOT invent prices, SKUs, warranties, or plan terms. Use only data from PRICING, WARRANTY, MEMBERSHIP, CATALOG, and DERIVED_TIER_TOTALS in CONTEXT.
- If <<DERIVED_TIER_TOTALS>> is present, use those ranges verbatim for each tier's `price_range_cad`. Do not invent or alter ranges. If a tier is omitted in <<DERIVED_TIER_TOTALS>>, you may compute from PRICING parts+labor; otherwise set "N/A".
- The copy must follow <<TIER_PLAN>> (mode, key, adds, membership, warranty).
- Warranties: quote terms directly from <<WARRANTY>>; do not invent durations.
- Memberships: only offer plans that exist in <<MEMBERSHIP>>. Include them in Standard Plus, Premium, and Premium Plus.
- Catalog: scope bullets and bonuses must be items that exist in <<CATALOG>> (or are logically implied by the diagnosis, like “replace failed capacitor”).
 - The tier names and scope bullets MUST reflect the plan in <<TIER_PLAN>>:
   - If plan.mode == "replacement", the option name and bullets MUST clearly say "replace" and reference the `primary_key`.
   - If plan.mode == "repair", the option MUST focus on the repair keyed by `primary_key` and avoid full-system replacement language.
   - IAQ add-ons must only appear in “Plus” tiers (Standard Plus, Premium Plus). Do not include IAQ in Standard or Premium.
   - When IAQ items are included by the plan, list them under **bonuses** (do not put IAQ items in `scope_bullets`).
   - When warranties are included by the plan, list them under **Warranty** (do not put warranty terms in `scope_bullets`).     
   - When memberships are included by the plan, list them under **Membership** (do not put membership terms in `scope_bullets`).     
- Absolutely NO new keys or schema deviations.

TIER / ORDER / QUANTITY
- Always produce exactly 6 tiers: ["Premium Plus","Premium","Standard Plus","Standard","Economy","Band-Aid"] unless disallowed by the rules below.
- If a full system replacement is required (e.g., heat exchanger cracked), OMIT "Economy" and "Band-Aid" (fewer tiers are allowed).
- If the problem is minor (e.g., capacitor/ignitor), OMIT full-replacement tiers.
- In `tech_sales_script.suggested_order_to_present`, list tiers from HIGH → LOW price.

SAFETY & POLICY
- Be compliant and safety-first; if life-safety is implicated (CO risk, gas leak, cracked heat exchanger), set `tech_sales_script.escalate = true` and include a clear instruction.
- If data conflicts across blocks, prefer WARRANTY over PRICING for warranty terms, and prefer MEMBERSHIP for plan specifics.

TONE
- Customer-facing copy: clear, benefits-forward, plain language.
- Tech script: conversational bullets using SPIN/Challenger/Consultative/Hormozi-style ROI framing.

SALES SCRIPT STRUCTURE
- Opening: summarize issues from FREE_TEXT_DIAGNOSIS, ask 1–2 consultative/SPIN questions, address system age if relevant, highlight implications of inaction.
- Top Anchor: present Premium Plus first, educate like an expert, ask customer to guess cost, reassure it’s less than expected.
- Middle Move: position Standard/Standard Plus as affordable ways to fully solve the issue.
- Anchor Low: Economy/Band-Aid fix today but limited long-term reliability.
- Objection Handling: ≥3 objection/response pairs; focus on ROI math (savings, avoided breakdowns, warranties) and soft takeaways.
- Close Prompt: end with “Which option are you comfortable moving forward with today?”

STRICT OUTPUT
- Return ONLY one JSON object that exactly matches the provided schema. No extra text.

GROUNDING AUDIT (for validation)
- For each option, if you used any PRICING fields, include short ‘notes’ citing the field names (e.g., “Used repairs.condenser_fan_motor.parts_range_cad and defaults.labor_rate_per_hour_cad”).
"""

JSON_SCHEMA_HINT = {
  "customer_option_sheet": {
    "customer_summary_markdown": "string 120–200 words",
    "options": [{
      "tier": "Premium Plus | Premium | Standard Plus | Standard | Economy | Band-Aid",
      "name": "string",
      "scope_bullets": ["string"],
      "price_range_cad": "string like \"$1,100–$1,400\" or \"N/A\"",
      "warranty": ["string"],
      "membership_offer": "string",
      "bonuses": ["string"],
      "notes": "string (risks/assumptions/limitations; list differences from higher tier; ‘Missing Data’ if applicable)"
    }],
    "disclaimers": ["string"]
  },
  "tech_sales_script": {
    "opening": "string (150–250 words, consultative/SPIN style with empathy + implication)",
    "top_anchor": "string (150–250 words, Premium Plus first, Challenger style teaching/expert framing)",
    "middle_move": "string (150–250 words, Standard options positioned as most affordable full solution, note what is lost from Premium Tiers)",
    "anchor_low": "string (150–250 words, contrast Economy/Band-Aid, highlight risks of limited protection)",
    "objection_handling": [{"objection": "string", "response": "string (ROI math, benefits, soft takeaways)"}],
    "close_prompt": "string (assumptive nudge: 'Which of these options are you comfortable moving forward with today?')",
    "escalate": False
  }
}

USER_PROMPT_TEMPLATE = """
FREE_TEXT_DIAGNOSIS:
{diagnosis}

CUSTOMER_CONTEXT:
- postal_code: {postal_code}
- system_age: {system_age}
- brand_model_if_known: {brand_model}
- service_plan: {plan_status}
- urgency: {urgency}
- constraints: {constraints}

<<REPAIR_REPLACE_POLICY>>
equipment_type: {equipment_type}
system_age_years: {system_age_years} ({age_reason})
critical_hits: {critical_hits}
recommend_replacement: {recommend_replacement}
omit_low_tiers: {omit_low_tiers}
<</REPAIR_REPLACE_POLICY>>

<<TIER_PLAN>>
For each tier, follow this EXACT plan when writing names/scope bullets and determining repair vs. replacement:
{derived_tier_plan_text}
<</TIER_PLAN>>

<<DERIVED_TIER_TOTALS>>
Use these price ranges for each tier's `price_range_cad` exactly as shown (do not deviate):
{derived_tier_totals_text}
<</DERIVED_TIER_TOTALS>>

<<WARRANTY>>
{warranty_text}
<</WARRANTY>>

<<MEMBERSHIP>>
{membership_text}
<</MEMBERSHIP>>

<<CATALOG>>
{catalog_text}
<</CATALOG>>

RESPONSE REQUIREMENTS:
- Exactly follow this JSON schema (field names and types must match):

{schema_json}

ADDITIONAL VALIDATION:
- If any field required to compute price is missing, set "price_range_cad": "N/A" and add a 'Missing Data' note specifying which field(s) are missing.
- Use EXACT tier labels (case/spacing): "Premium Plus","Premium","Standard Plus","Standard","Economy","Band-Aid".
- `tech_sales_script.suggested_order_to_present` must be high→low tiers that you actually produced.
- Address system age in summary if `young_system`==false.
- Do not include markdown in JSON fields except `customer_summary_markdown`.

"""

# -------------------- Helpers --------------------
def _to_dict(model: BaseModel) -> dict:
    """Support Pydantic v1 (dict) and v2 (model_dump)."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

def render_option_sheet_html(data: ModelResponse, static_base: str = "") -> str:
    logger.debug("Templates dir: %s; exists=%s", TEMPLATES_DIR, os.path.exists(TEMPLATES_DIR))
    tmpl = env.get_template("option_sheet.html")
    return tmpl.render(obj=_to_dict(data), static_base=static_base)

def _order_for_presentation(returned_order: list[str]) -> list[str]:
    # Keep only tiers the model actually produced, in our desired order
    wanted = [t for t in ANCHOR_HIGH_ORDER if t in returned_order]
    # Append any unexpected tiers to the end (defensive)
    extras = [t for t in returned_order if t not in wanted]
    return wanted + extras

def _first_key(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def _as_money(x) -> Optional[str]:
    if x is None:
        return None
    # Accept "120", 120, 120.0, "$120", "CAD 120"
    s = str(x).strip()
    s = re.sub(r"^[^\d\-]+", "", s)  # strip leading currency text/symbols
    try:
        val = float(s)
        # format with thousands separator, 0 decimals if int-ish
        if abs(val - round(val)) < 1e-9:
            return f"{int(round(val)):,}"
        return f"{val:,.2f}"
    except Exception:
        # keep original if non-numeric like "CAD 149/mo"
        return str(x)

def _fmt_currency_amount(amount, currency_code: Optional[str]) -> str:
    amt = _as_money(amount)
    if not amt:
        return "N/A"
    # Simple display: "CAD 1,234" or "$1,234" if USD
    if not currency_code or currency_code.upper() == "USD":
        # Use $ for USD
        return f"${amt}"
    return f"{currency_code.upper()} {amt}"

def _currency_from_pricing(pricing: Dict[str, Any]) -> str:
    code = _first_key(pricing.get("defaults", {}), ["currency", "currency_code"], None)
    if code:
        return str(code).upper()
    # Heuristics: look for *_cad keys
    if any("cad" in k.lower() for k in pricing.get("defaults", {}).keys()):
        return "CAD"
    return "USD"

def should_recommend_replacement(equip: str, age: Optional[int], critical_hits: list[str], cfg: dict) -> bool:
     pol = cfg.get("replacement_policy", {})
     if critical_hits: return True
     if age is None: return False
     thresh = pol.get("age_replace_top_tiers_at_or_over", {}).get(equip, None)
     return thresh is not None and age >= thresh

def is_replacement_option(opt: dict) -> bool:
    text = (opt.get("name","") + " " + " ".join(opt.get("scope_bullets",[]))).lower()
    return any(w in text for w in ["replace", "replacement", "new system", "install"])

def infer_age_years(free_text: Optional[str], explicit_age: Optional[str]) -> tuple[Optional[int], str]:
    """Return (age_years, reason). Parses explicit field first, then free text (e.g., '12 years old' or 'installed in 2012')."""
    if explicit_age:
        m = re.search(r"(\d{1,2})", str(explicit_age))
        if m:
            try:
                return int(m.group(1)), "from_field"
            except Exception:
                pass
    if free_text:
        m = re.search(r"(\d{1,2})\s*[- ]?(?:year|yr)s?\s*old", free_text, re.I)
        if m:
            return int(m.group(1)), "from_text"
        m = re.search(r"installed\sin\s(\d{4})", free_text, re.I)
        if m:
            year = int(m.group(1))
            now = datetime.date.today().year
            return max(0, now - year), "from_text"
    return None, "unknown"

def infer_equipment_type(free_text: Optional[str]) -> str: # brand_model: Optional[str]
    blob = f"{free_text or ''}".lower() # {brand_model or ''}
    if any(k in blob for k in ["heat pump", "heatpump", "hp"]):
        return "heat_pump"
    if any(k in blob for k in ["mini-split", "ductless"]):
        return "ductless"
    if any(k in blob for k in ["furnace", "heat exchanger", "inducer", "ignitor", "igniter"]):
        return "furnace"
    if any(k in blob for k in ["ac ", " a/c", "air conditioner", "condenser", "compressor", "evaporator"]):
        return "air_conditioner"
    if any(k in blob for k in ["tankless", "water heater", "hwt", "hot water"]):
        return "water_heater"
    return "unknown"

def detect_critical_hits(text: Optional[str], cfg: dict) -> list[str]:
    """Match tenant-defined critical failure phrases (case-insensitive)."""
    src = (text or "").lower()
    crit = [c for c in cfg.get("replacement_policy", {}).get("critical_failures", []) if c.lower() in src]
    return crit

def should_recommend_replacement(equip: str, age: Optional[int], critical_hits: list[str], cfg: dict) -> bool:
    pol = cfg.get("replacement_policy", {})
    if critical_hits:
        return True
    if age is None:
        return False
    thresh = pol.get("age_replace_top_tiers_at_or_over", {}).get(equip, None)
    return thresh is not None and age >= thresh

def enforce_options(model_json: dict, policy: dict) -> dict:
    opts = model_json["customer_option_sheet"]["options"]
    if policy["recommend_replacement"]:
        # Force top tiers to replacement
        for t in ["Premium Plus","Premium","Standard Plus"]:
            for o in opts:
                if o["tier"] == t and not is_replacement_option(o):
                    o["notes"] = (o.get("notes","") + " | Adjusted: top tier must be a full replacement per policy").strip()
        if policy.get("omit_low_tiers"):
            opts = [o for o in opts if o["tier"] not in ["Economy","Band-Aid"]]
            model_json["tech_sales_script"]["suggested_order_to_present"] = [
                x for x in model_json["tech_sales_script"]["suggested_order_to_present"]
                if x not in ["Economy","Band-Aid"]
            ]
    else:
        # Remove replacements for small fixes
        for o in opts:
            if is_replacement_option(o):
                o["notes"] = (o.get("notes","") + " | Adjusted: minor issue, recommend repair path").strip()
    model_json["customer_option_sheet"]["options"] = opts
    return model_json

# --- Compute price ranges ---
def infer_repair_keys_llm(diagnosis: str, available_keys: List[str]) -> List[str]:
    """
    Ask the model to select up to 3 keys from available_keys that best match diagnosis.
    Returns a subset of available_keys in relevance order.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not diagnosis or not available_keys:
        return []
    prompt = (
        "You map a technician diagnosis to known repair keys.\n"
        "Return ONLY a JSON object: {\"keys\": [<up to 3 keys from AVAILABLE_KEYS>]}\n"
        "If nothing matches, return {\"keys\": []}.\n\n"
        f"DIAGNOSIS:\n{diagnosis}\n\n"
        f"AVAILABLE_KEYS:\n{', '.join(sorted(set(available_keys)))}\n"
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("FC_MATCH_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        keys = data.get("keys", [])
        # keep only valid keys, preserve order, dedupe
        seen, out = set(), []
        for k in keys:
            if k in available_keys and k not in seen:
                out.append(k); seen.add(k)
        return out[:3]
    except Exception:
        return []

def infer_replacement_keys_llm(diagnosis: str, available_keys: List[str]) -> List[str]:
    """Same idea, but for replacements; lets the model pick 1–2 likely replacement SKUs/levels."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not diagnosis or not available_keys:
        return []
    prompt = (
        "Select up to 2 replacement keys that best address the diagnosis. "
        "Return ONLY JSON: {\"keys\": [<subset of AVAILABLE_KEYS>]}.\n\n"
        f"DIAGNOSIS:\n{diagnosis}\n\n"
        f"AVAILABLE_KEYS:\n{', '.join(sorted(set(available_keys)))}\n"
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("FC_MATCH_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            response_format={"type":"json_object"},
            messages=[{"role":"user","content":prompt}]
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        keys = data.get("keys", [])
        out, seen = [], set()
        for k in keys:
            if k in available_keys and k not in seen:
                out.append(k); seen.add(k)
        return out[:2]
    except Exception:
        return []

def _to_float_money(val) -> Optional[float]:
    if val is None: return None
    s = str(val).strip()
    m = re.search(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d+)?", s)
    if not m: return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None
    
def _pick_replacement_candidates(pricing: dict, k: int = 2) -> list[str]:
    """Pick up to k replacement keys spanning 'standard' and 'high efficiency' if present."""
    repl = (pricing.get("replacements") or {})
    keys = list(repl.keys())
    if not keys: return []
    hi = [k for k in keys if re.search(r"(high|95|96|seer\s*1[6-9]|cold[- ]?climate|variable)", k, re.I)]
    lo = [k for k in keys if k not in hi]
    out = []
    if lo: out.append(lo[0])
    if hi: out.append(hi[0])
    if not out: out.append(keys[0])
    return out[:k]

def _range_from_item(item: dict, low_key: str, hi_key: str = None) -> Optional[Tuple[float, float]]:
    """Accepts [low,high] in item[low_key] or item[hi_key]."""
    v = item.get(low_key)
    if not v and hi_key:
        v = item.get(hi_key)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo = _to_float_money(v[0]); hi = _to_float_money(v[1])
        if lo is not None and hi is not None:
            return float(lo), float(hi)
    return None

def _compute_repair_total_range(pricing: dict, key: str, labor_rate: Optional[float]) -> Optional[Tuple[float,float]]:
    """parts_range (+ labor_hours * labor_rate) if available; else None."""
    item = (pricing.get("repairs") or {}).get(key, {})
    parts = _range_from_item(item, "parts_range_cad") or _range_from_item(item, "parts_range_usd") or _range_from_item(item, "parts_range")
    hrs   = _range_from_item(item, "labor_hours_range") or _range_from_item(item, "labor_range_hours") or _range_from_item(item, "labour_hours_range")
    if not parts and not (hrs and labor_rate is not None):
        return None
    lo = parts[0] if parts else 0.0
    hi = parts[1] if parts else 0.0
    if hrs and labor_rate is not None:
        lo += hrs[0] * labor_rate
        hi += hrs[1] * labor_rate
    return (lo, hi)

def _compute_replacement_total_range(pricing: dict, key: str) -> Optional[Tuple[float,float]]:
    item = (pricing.get("replacements") or {}).get(key, {})
    rr = _range_from_item(item, "installed_range_cad") or _range_from_item(item, "installed_range_usd") or _range_from_item(item, "installed_range")
    return rr

def _pick_iaq_addons(pricing: dict, max_n: int = 2) -> list[Tuple[str, Tuple[float,float]]]:
    """Return up to max_n IAQ items with ranges (median-ish priced)."""
    out = []
    iaq = (pricing.get("iaq_addons") or {})
    for k, item in iaq.items():
        pr = _range_from_item(item, "parts_range_cad") or _range_from_item(item, "parts_range_usd") or _range_from_item(item, "parts_range")
        hrs = _range_from_item(item, "labor_hours_range") or _range_from_item(item, "labor_range_hours") or _range_from_item(item, "labour_hours_range")
        out.append((k, (pr, hrs)))
    # prefer ones with both parts+labor data
    out.sort(key=lambda x: (x[1][0] is not None and x[1][1] is not None), reverse=True)
    pick = []
    for k, (pr, hrs) in out:
        if pr:
            if hrs:
                pick.append((k, (pr[0], pr[1], hrs[0], hrs[1])))
            else:
                pick.append((k, (pr[0], pr[1], 0.0, 0.0)))
        if len(pick) >= max_n: break
    return pick

def _annualize_membership(membership: dict, plan_key: str) -> Optional[float]:
    """Return approx annual cost for plan_key if price + monthly cycle present."""
    plan = (membership or {}).get(plan_key or "", {})
    p = _to_float_money(plan.get("price_cad") or plan.get("price_usd") or plan.get("price"))
    cycle = str(plan.get("billing_cycle", "")).lower()
    if p is None: return None
    if "month" in cycle: return p * 12.0
    if "year" in cycle:  return p
    return p  # default assume already annual

def _fmt_range_cad(r: Tuple[float,float]) -> str:
    lo, hi = r
    return f"CAD {int(round(lo)):,}–{int(round(hi)):,}"

def _parse_tier_totals_text(txt: str) -> dict[str, str]:
    """
    Parse lines like 'Premium Plus: CAD 8,200–10,400' into {'Premium Plus': 'CAD 8,200–10,400', ...}
    """
    out = {}
    if not txt:
        return out
    for line in txt.splitlines():
        if ": " in line:
            tier, rng = line.split(": ", 1)
            out[tier.strip()] = rng.strip()
    return out

def _is_replacement_copy(opt: dict) -> bool:
    text = (opt.get("name","") + " " + " ".join(opt.get("scope_bullets",[]))).lower()
    return any(w in text for w in ["replace","replacement","new system","install new"])

def _ensure_copy_matches_plan(options: list[dict], plan: dict):
    for o in options:
        tier = o.get("tier")
        p = plan.get(tier, {})
        if not p: continue
        want_repl = (p.get("mode") == "replacement")
        has_repl = _is_replacement_copy(o)
        # If mismatch, patch name minimally and prepend a clarifying bullet
        if want_repl and not has_repl:
            o["name"] = (o.get("name") or "Premium Option") + " — Replace System"
            o.setdefault("scope_bullets", []).insert(0, f"Replace system per {p.get('primary_key')}")
            #logger.info("Fix copy: tier '%s' should be replacement but reads like repair → patching text", tier)
        if (not want_repl) and has_repl:
            o["name"] = (o.get("name") or "Repair Option") + " — Repair Only"
            o.setdefault("scope_bullets", []).insert(0, f"Repair per {p.get('primary_key')}, no full replacement")
            #logger.info("Fix copy: tier '%s' should be repair but reads like replacement → patching text", tier)
        # --- IAQ → ensure they appear as BONUSES (not bullets)
        adds = p.get("adds") or []
        if adds:
            o.setdefault("bonuses", [])
            # 1) Move any IAQ-like bullets into bonuses
            kept_bullets = []
            for b in o.get("scope_bullets", []) or []:
                if any(add.lower().replace("_"," ") in b.lower() or "iaq" in b.lower() for add in adds):
                    # push a clean, canonical label into bonuses
                    # e.g., "UV Light" from "uv_light"
                    for add in adds:
                        if add.lower().replace("_"," ") in b.lower() or "iaq" in b.lower():
                            label = add.replace("_"," ").title()
                            if all(label.lower() != x.lower() for x in o["bonuses"]):
                                o["bonuses"].append(label)
                    # do NOT keep this bullet
                else:
                    kept_bullets.append(b)
            o["scope_bullets"] = kept_bullets
            # 2) Ensure every planned IAQ is present in bonuses
            for add in adds:
                label = add.replace("_"," ").title()
                if all(label.lower() != x.lower() for x in o["bonuses"]):
                    o["bonuses"].append(label)

        # Membership mention lives in bullets
        if p.get("include_membership"):
            if all("membership" not in b.lower() for b in o.get("scope_bullets", [])):
                o.setdefault("scope_bullets", []).append("Include membership benefits in first year")
        if p.get("warranty_years"):
            yrs = p["warranty_years"]
            if all("warranty" not in b.lower() for b in o.get("scope_bullets", [])):
                o.setdefault("scope_bullets", []).append(f"Extended labor warranty: {yrs} years")

def _classify_replacement_key(key: str) -> str:
    if not key: return "standard"
    if re.search(r"(high|95|96|seer\s*1[6-9]|cold[- ]?climate|variable|two[- ]?stage|modulating)", key, re.I):
        return "high"
    return "standard"

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {"ok": True, "templates_dir": TEMPLATES_DIR, "data_dir": DATA_DIR}

@app.post("/admin/reload-context")
def reload_context():
    load_business_context.cache_clear()
    return {"ok": True}

@app.post("/generate", response_model=ModelResponse)
def generate(payload: GeneratePayload):
    # Load business context (structured)
    try:
        ctx = load_business_context()
    except Exception as e:
        logger.error("Failed loading business context: %s", e)
        raise HTTPException(status_code=500, detail="Business context files missing or invalid under /data")

    # Turn it into the text blocks the prompt expects
    pricing_text, warranty_text, membership_text, catalog_text, config_text = format_context_strings(ctx, payload.postal_code or None)
    override_block = (f"\n\nOVERRIDES:\n{payload.business_override.strip()}"
                      if payload.business_override and payload.business_override.strip() else "")
    cfg = ctx.get("config", {})
    equip_type = infer_equipment_type(payload.free_text_diagnosis)  # your existing or simple heuristic
    age, age_reason = infer_age_years(payload.free_text_diagnosis, payload.system_age)
    crit_list = [kw for kw in cfg.get("replacement_policy", {}).get("critical_failures", []) 
                if kw.lower() in (payload.free_text_diagnosis or "").lower()]
    rec_replace = should_recommend_replacement(equip_type, age, crit_list, cfg)
    omit_low = bool(cfg.get("replacement_policy", {}).get("omit_low_tiers_if_replacement_required", True))
    DEFAULT_AGE_CUTOFFS = {"furnace": 12, "air_conditioner": 10, "heat_pump": 10}
    age_thresholds = (cfg.get("replacement_policy", {}) or {}).get("age_replace_top_tiers_at_or_over", {}) or {}
    age_cutoff = age_thresholds.get(equip_type, DEFAULT_AGE_CUTOFFS.get(equip_type))
    young_system = (age is not None and age < age_cutoff)
    #logger.info("Age policy → age=%s equip=%s cutoff=%s young_system=%s", age, equip_type, age_cutoff, young_system)
    
    # Summarize in policy
    policy = {
        "equipment_type": equip_type,
        "system_age_years": age,
        "age_reason": age_reason,
        "critical_hits": crit_list,
        "recommend_replacement": rec_replace,
        "omit_low_tiers": omit_low,
    }

    # --- LLM-assisted mapping and liberal range derivation ---
    labor_rate = _labor_rate(ctx.get("pricing", {}) or {}, payload.postal_code)
    pricing_d = ctx.get("pricing", {}) or {}
    membership_d = ctx.get("membership", {}) or {}
    warranty_d = ctx.get("warranty", {}) or {}
    repairs_keys = list((pricing_d.get("repairs") or {}).keys())
    repl_keys   = list((pricing_d.get("replacements") or {}).keys())
    #logger.debug("Labor rate resolved: %s (postal=%s)", labor_rate, payload.postal_code)

    # Ask LLM which repair/replacement keys match best
    picked_repairs = infer_repair_keys_llm(payload.free_text_diagnosis or "", repairs_keys)
    repair_key = picked_repairs[0] if picked_repairs else None
    repair_viable = False
    if repair_key:
        _rng = _compute_repair_total_range(pricing_d, repair_key, labor_rate)
        repair_viable = bool(_rng)
   # logger.info("Repairs → picked=%s key=%s viable=%s", picked_repairs, repair_key, repair_viable)    
    
    picked_repls   = infer_replacement_keys_llm(payload.free_text_diagnosis or "", repl_keys)
    if not picked_repls:
        picked_repls = _pick_replacement_candidates(pricing_d, k=2)
    
    # Partition into std/high so we can map consistently to Premium / Premium Plus
    repl_std_key = next((k for k in picked_repls if _classify_replacement_key(k)=="standard"), None)
    repl_high_key= next((k for k in picked_repls if _classify_replacement_key(k)=="high"), None)
    if not repl_std_key and repl_keys:
        repl_std_key = next((k for k in repl_keys if _classify_replacement_key(k)=="standard"), repl_keys[0])
    if not repl_high_key:
        repl_high_key = repl_std_key
   # logger.info("Replacements → std=%s high=%s (picked=%s)", repl_std_key, repl_high_key, picked_repls)

    # IAQ add-ons candidates (up to 2)
    iaq_picks = _pick_iaq_addons(pricing_d, max_n=2)  # list[(key,(parts_lo,parts_hi,hrs_lo,hrs_hi))]
    iaq_keys = [k for k, _spec in iaq_picks]
    adds_std  = []
    adds_plus = iaq_keys[:2] if len(iaq_keys) >= 2 else iaq_keys[:1]
   # logger.info("IAQ adds → adds_std=%s adds_plus=%s (all=%s)", adds_std, adds_plus, iaq_keys)

    def _sum_iaq(rng: Tuple[float,float]) -> Tuple[float,float]:
        lo, hi = rng
        for key,(plo,phi,hlo,hhi) in iaq_picks:
            lo += (plo + hlo * (labor_rate or 0.0))
            hi += (phi + hhi * (labor_rate or 0.0))
        logger.debug("  IAQ add %s: parts=(%s,%s) hrs=(%s,%s) → new range=(%s,%s)",
                key, plo, phi, hlo, hhi, lo, hi)    
        return (lo, hi)

    # Membership add (annualized) if selected or per tier rule
    def _membership_add(plan_key: Optional[str]) -> float:
        val = _annualize_membership(membership_d, plan_key) if plan_key else None
        add = float(val) if val is not None else 0.0
       # logger.info("Membership add (annualized) for plan '%s': %s", plan_key, add)
        return add    

    # Warranty add (optional): look for priced extended labor if tenant provides e.g. warranty.prices.extended_labor_years_10
    def _warranty_add(years: int = 10) -> float:
        prices = (warranty_d.get("prices") or {})
        # accept various keys; return 0 if none
        candidates = [f"extended_labor_{years}_years_cad", f"extended_labor_{years}_years", "extended_labor_cad", "extended_labor"]
        for k in candidates:
            v = _to_float_money(prices.get(k))
            if v is not None:
                #logger.info("Warranty add detected for %s: %s", k, v)
                return float(v)        
        return 0.0
    
    include_membership = bool(payload.plan_status and str(payload.plan_status).lower() != "none")
    extended_warranty_yrs = int((cfg.get("premium_warranty_years") or 10))
    #logger.info("Adds → include_membership=%s extended_warranty_yrs=%s", include_membership, extended_warranty_yrs)

    # Compose tier totals as ranges
    derived_tier_totals = {}
    derived_tier_plan = {}

    if repair_viable:
        # Band-Aid — minimal repair
        rng = _compute_repair_total_range(pricing_d, repair_key, labor_rate)
        derived_tier_totals["Band-Aid"] = (rng[0], rng[0])  # anchor low end
        derived_tier_plan["Band-Aid"] = {
            "mode": "repair", "primary_key": repair_key,
            "adds": [], "include_membership": False, "warranty_years": None
        }
       # logger.info("TIER %s → plan=%s range=%s", "Band-Aid", derived_tier_plan["Band-Aid"], derived_tier_totals["Band-Aid"])
        
        # Economy — full repair range
        derived_tier_totals["Economy"] = rng
        derived_tier_plan["Economy"] = {
            "mode": "repair", "primary_key": repair_key,
            "adds": [], "include_membership": False, "warranty_years": None
        }
        #logger.info("TIER %s → plan=%s range=%s", "Economy", derived_tier_plan["Economy"], derived_tier_totals["Economy"])
        
    # STANDARD
    if young_system and repair_viable:
        std_rng = _compute_repair_total_range(pricing_d, repair_key, labor_rate)
    else:
        std_rng = _compute_replacement_total_range(pricing_d, repl_std_key)
    
    # STANDARD: base only (no add-ons)
    derived_tier_totals["Standard"] = std_rng
    derived_tier_plan["Standard"] = {
        "mode": ("repair" if young_system and repair_viable else "replacement"),
        "primary_key": (repair_key if young_system and repair_viable else repl_std_key),
        "adds": [], "include_membership": False, "warranty_years": None
    }
    #logger.info("TIER %s → plan=%s range=%s", "Standard", derived_tier_plan["Standard"], derived_tier_totals["Standard"])

    
    # STANDARD PLUS — Standard base + IAQ(s) + membership 
    def _add_membership(rng):
        if not rng: return None
        add = _membership_add(payload.plan_status)
        return (rng[0]+add, rng[1]+add)
    
    def _add_iaqs(rng, picks):
        if not rng: return None
        lo, hi = rng
        for key,(plo,phi,hlo,hhi) in picks:
            lo += (plo + hlo * (labor_rate or 0.0))
            hi += (phi + hhi * (labor_rate or 0.0))
        return (lo, hi)
    # Standard already included the first IAQ (if available). Add the rest here.
    base_for_plus = derived_tier_totals["Standard"]  # no-IAQ base
    stdp_rng = _add_iaqs(base_for_plus, iaq_picks) or base_for_plus
    stdp_rng = _add_membership(stdp_rng)
    derived_tier_totals["Standard Plus"] = stdp_rng
    derived_tier_plan["Standard Plus"] = {
        "mode": derived_tier_plan["Standard"]["mode"],
        "primary_key": derived_tier_plan["Standard"]["primary_key"],
        "adds": adds_plus, "include_membership": include_membership, "warranty_years": None
    }
    #logger.info("TIER %s → plan=%s range=%s", "Standard Plus", derived_tier_plan["Standard Plus"], derived_tier_totals["Standard Plus"])


    # PREMIUM — standard-eff replacement + membership + (priced warranty if provided)
    prem = _compute_replacement_total_range(pricing_d, repl_std_key)
    prem = _add_membership(prem)
    prem = (prem[0] + _warranty_add(10), prem[1] + _warranty_add(10)) if prem else None
    derived_tier_totals["Premium"] = prem
    derived_tier_plan["Premium"] = {
        "mode": "replacement", "primary_key": repl_std_key,
        "adds": [], "include_membership": include_membership, "warranty_years": extended_warranty_yrs
    }
   # logger.info("TIER %s → plan=%s range=%s", "Premium", derived_tier_plan["Premium"], derived_tier_totals["Premium"])

    
    # PREMIUM PLUS — high-eff replacement + IAQ + membership + warranty
    pp = _compute_replacement_total_range(pricing_d, repl_high_key)
    if pp and iaq_picks:
        lo, hi = pp
        for key,(plo,phi,hlo,hhi) in iaq_picks:
            lo += (plo + hlo * (labor_rate or 0.0))
            hi += (phi + hhi * (labor_rate or 0.0))
        pp = (lo, hi)
    pp = _add_membership(pp)
    pp = (pp[0] + _warranty_add(10), pp[1] + _warranty_add(10)) if pp else None
    derived_tier_totals["Premium Plus"] = pp
    derived_tier_plan["Premium Plus"] = {
        "mode": "replacement", "primary_key": repl_high_key,
        "adds": adds_plus, "include_membership": include_membership, "warranty_years": extended_warranty_yrs
    }
    #logger.info("TIER %s → plan=%s range=%s", "Premium Plus", derived_tier_plan["Premium Plus"], derived_tier_totals["Premium Plus"])

    # Stringify for prompt
    def _fmt_tier_map(td: dict) -> str:
        lines = []
        for tier in ["Premium Plus","Premium","Standard Plus","Standard","Economy","Band-Aid"]:
            if tier in td:
                lines.append(f"{tier}: {_fmt_range_cad(td[tier])}")
        return "\n".join(lines) if lines else "none"
    
    derived_tier_totals_text = _fmt_tier_map(derived_tier_totals)
    #logger.info("Derived tier totals text:\n%s", derived_tier_totals_text)
    derived_tier_plan_text = json.dumps(derived_tier_plan, ensure_ascii=False, indent=2)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        diagnosis=(payload.free_text_diagnosis or "").strip(),
        postal_code=payload.postal_code or "N/A",
        system_age=payload.system_age or "N/A",
        brand_model=payload.brand_model or "N/A",
        plan_status=payload.plan_status or "N/A",
        urgency=payload.urgency or "normal",
        constraints=payload.constraints or "N/A",
        pricing_text=(pricing_text + override_block) if pricing_text else "N/A",
        warranty_text=warranty_text or "N/A",
        membership_text=membership_text or "N/A",
        catalog_text=catalog_text or "N/A",
        derived_tier_totals_text=derived_tier_totals_text,
        derived_tier_plan_text=derived_tier_plan_text,    
        equipment_type=equip_type or "unknown",
        system_age_years=age if age is not None else "unknown",
        age_reason=age_reason,
        critical_hits=", ".join(crit_list) or "none",
        recommend_replacement=str(rec_replace).lower(),
        omit_low_tiers=str(omit_low).lower(),
        few_shot="",  # optional few-shot exemplars
        schema_json=json.dumps(JSON_SCHEMA_HINT, ensure_ascii=False, indent=2)
    )

    api_key = os.getenv("OPENAI_API_KEY")
    #api_key = False

    if not api_key or not _HAS_OPENAI:
        logger.debug("Stub mode: returning demo payload (no OPENAI_API_KEY or no openai client)")
        demo = {
          "customer_option_sheet": {
            "customer_summary_markdown": "Your AC is older and has a seized condenser fan plus low refrigerant. We’ve mapped options from a quick fix to full replacement so you can choose what fits your comfort and budget.",
            "options": [
                {
                "tier": "Premium Plus",
                "name": "Top-Tier System + IAQ + Platinum",
                "scope_bullets": ["High-efficiency condenser", "IAQ package", "Platinum membership"],
                "price_range_cad": "$9,500–$11,000",
                "warranty": ["10-year parts; lifetime labor"],
                "membership_offer": "Platinum Plan included",
                "bonuses": ["Duct sealing evaluation", "Smart Thermostat"],
                "notes": "Max comfort and peace of mind."
              },
              {
                "tier": "Premium",
                "name": "High-Efficiency + Smart Thermostat",
                "scope_bullets": ["16+ SEER system", "Smart thermostat"],
                "price_range_cad": "$6,800–$8,200",
                "warranty": ["10-year parts; 3-year labor"],
                "membership_offer": "Gold Plan included (1 year)",
                "bonuses": ["Air purifier"],
                "notes": "Comfort & efficiency upgrade."
              },
              {
                "tier": "Standard Plus",
                "name": "Repair + Membership",
                "scope_bullets": ["Replace motor", "Leak remediation (minor)", "Enroll in Silver Plan"],
                "price_range_cad": "$1,100–$1,400",
                "warranty": ["1-year labor; part per OEM"],
                "membership_offer": "Silver Plan included (1 year)",
                "bonuses": ["Smart thermostat"],
                "notes": "Stabilizes reliability for at least 1 season."
              },
              {
                "tier": "Standard",
                "name": "Motor + Leak Inspection",
                "scope_bullets": ["Replace motor", "Electronic leak check"],
                "price_range_cad": "$650–$850",
                "warranty": ["6-month labor; part per OEM"],
                "membership_offer": "Add Silver Plan $14/mo",
                "bonuses": ["Condensate safety switch"],
                "notes": "If major leak found, replacement may be more cost-effective."
              },
               {
                "tier": "Economy",
                "name": "Motor + Leak Inspection + Top-up",
                "scope_bullets": ["Replace motor", "Electronic leak check", "Add 2 lbs R-410A"],
                "price_range_cad": "$550–$650",
                "warranty": ["6-month labor; part per OEM"],
                "membership_offer": "Add Silver Plan $14/mo",
                "bonuses": ["Condensate safety switch"],
                "notes": "If major leak found, replacement may be more cost-effective."
              },
              {
                "tier": "Band-Aid",
                "name": "Replace Fan Motor + Top-Up",
                "scope_bullets": ["Replace condenser fan motor", "Add 2 lbs R-410A"],
                "price_range_cad": "$350–$500",
                "warranty": ["90-day part"],
                "membership_offer": "Add Silver Plan $14/mo",
                "bonuses": ["Surge protector"],
                "notes": "Does not address likely leak; short-term reliability only."
              }
            ],
            "disclaimers": ["Pricing pending site confirmation; Location rates apply; parts extra if needed."]
          },
          "tech_sales_script": {
            "opening": "I’ve mapped your options from quick fix to full solution so you can pick what fits comfort and budget.",
            "anchor_low": "The first two options get you cooling today but don’t address age or leaks—likely short-term fixes.",
            "middle_move": "Enhanced repair stabilizes the system and includes a plan—good value if you want one more season.",
            "top_anchor": "Replacement options deliver lower bills, stronger warranties, and peace of mind; premium adds air quality benefits.",
            "objection_handling": [
              {"objection": "Price is high", "response": "We can stage repairs or apply financing; premium reduces future risk and bills."},
              {"objection": "We’ll wait", "response": "We can hold a slot and lock today’s promo; waiting risks another breakdown in peak heat."}
            ],
            "close_prompt": "Which option feels right for your home today? Most choose Enhanced or Premium for reliability and comfort.",
            "suggested_order_to_present": ["Premium Plus","Premium","Standard Plus", "Standard", "Economy","Band-Aid"],
            "escalate": False
          }
        }
        return ModelResponse(**demo)

    # Live call with OpenAI
    try:
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.4,
            messages=messages,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        data = enforce_options(data, {   # enforce replace vs repair policy deterministically
            "recommend_replacement": rec_replace,
            "omit_low_tiers": omit_low
        })
        if "tech_sales_script" in data:
            so = data["tech_sales_script"].get("suggested_order_to_present", [])
            data["tech_sales_script"]["suggested_order_to_present"] = _order_for_presentation(so)
        
        # Enforce our computed tier prices and plan (keeps copy aligned with math)
        opts = data.get("customer_option_sheet", {}).get("options", []) or []
        try:
            tier_price_map = _parse_tier_totals_text(derived_tier_totals_text)
            if tier_price_map and opts:
                for o in opts:
                    t = o.get("tier")
                    if t in tier_price_map:
                        logger.debug("Overriding model price for tier '%s': %s → %s", t, o["price_range_cad"], tier_price_map[t])
                        o["price_range_cad"] = tier_price_map[t]
            if opts:
                _ensure_copy_matches_plan(opts, derived_tier_plan)  # mutates in place
        except Exception:
            logger.exception("post-process enforcement failed:")
        finally:
            if "customer_option_sheet" in data:
                data["customer_option_sheet"]["options"] = opts

        return ModelResponse(**data)
            
    except Exception as e:
        logger.error("LLM call failed:\n%s", traceback.format_exc())
        # You could fallback to stub here if you prefer resiliency in prod:
        # return ModelResponse(**demo)
        raise HTTPException(status_code=500, detail=f"LLM call failed: {type(e).__name__}: {e}")

@app.get("/metadata/memberships")
def list_membership_plans():
    # Return normalized membership plan options for the tenant.
    try:
        ctx = load_business_context()
        raw = ctx.get("membership", {}) or {}
        plans = []

        # Always add a "none" option
        plans.append({"key": "none", "label": "none"})

        for name, val in raw.items():
            if isinstance(val, dict):
                label = val.get("label")
                price = val.get("price_cad") or val.get("price") or ""
                cycle = val.get("billing_cycle", "")
                terms = val.get("terms", "")
                plans.append({
                    "key": name,
                    "label": label,
                    "price": price,
                    "billing_cycle": cycle,
                    "terms": terms
                 })
            else:
                # val is a string
                label = name
                plans.append({"key": name, "label": label, "price": "", "billing_cycle": "", "terms": str(val)})

        return {"plans": plans}
    except Exception as e:
        # Safe fallback: only "none"
        return {"plans": [{"key": "none", "label": "none"}]}

@app.post("/render-html")
def render_html_endpoint(payload: dict):
    try:
        if not os.path.exists(os.path.join(TEMPLATES_DIR, "option_sheet.html")):
            raise RuntimeError(f"Template not found at {TEMPLATES_DIR}/option_sheet.html")
        model = ModelResponse(**payload)  # validate & coerce
        #static_base = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
        html = render_option_sheet_html(model)
        return {"html": html}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=f"Bad payload: {ve}")
    except Exception as e:
        logger.error("render-html failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Render failed: {type(e).__name__}: {e}")
    
@app.post("/render_pdf")
def render_pdf(payload: dict):
    model = ModelResponse(**payload)
    html = render_option_sheet_html(model)
    html = html.replace('href="/static/', f"{STATIC_DIR}/style.css")

    # Use headless browser to print to PDF
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.emulate_media(media="print")
        page.set_content(html=html, wait_until="load")
        page.add_style_tag(path=f"{STATIC_DIR}/style.css")

        pdf_bytes = page.pdf(
            format="Letter",
            margin={"top": "0.25in", "right": "0.5in", "bottom": "0.25in", "left": "0.5in"},
            print_background=True
        )
        browser.close()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="TESToption-sheet.pdf"'}
    )
    
    

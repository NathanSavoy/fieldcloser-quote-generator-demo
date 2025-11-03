import os, json, requests, hashlib
import streamlit as st

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

if "result_html" not in st.session_state:
    st.session_state.result_html = None
if "result_payload" not in st.session_state:
    st.session_state.result_payload = None
if "result_tss" not in st.session_state:
    st.session_state.result_tss = None
if "result_pdf" not in st.session_state:
    st.session_state.result_pdf = None  # optional cache

# ----- Generate and cache outputs ------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_pdf(payload: dict) -> bytes:
    # payload hash makes cache stable across reruns
    _ = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    r = requests.post(f"{backend_url}/render_pdf", json=payload, timeout=30)
    r.raise_for_status()
    return r.content

# ----- Cache tenant info ------
@st.cache_data(ttl=300)
def fetch_membership_plans():
    try:
        r = requests.get(f"{DEFAULT_BACKEND_URL}/metadata/memberships", timeout=10)
        r.raise_for_status()
        data = r.json()
        plans = data.get("plans", [])
        if not plans:
            return [("none", "none")]
        # Return list of (value, label)
        return [(p["key"], p.get("label", p["key"])) for p in plans]
    except Exception:
        return [("none","none")]


# -------- Streamlit UI ---------
st.set_page_config(page_title="FieldCloser MVP", layout="centered")

with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input("FastAPI backend URL", value=DEFAULT_BACKEND_URL)

# ------- Fetch membership plans for form --------
plan_pairs = fetch_membership_plans()
plan_values = [v for v, _ in plan_pairs]
plan_labels = [l for _, l in plan_pairs]
default_index = plan_values.index("none") if "none" in plan_values else 0

# --- INPUT FORM (only for data entry & submit) ---  
st.title("FieldCloser – HVAC Option Sheet & Sales Script")
with st.form("generate_form", clear_on_submit=False):
    free_text_diagnosis = st.text_area("Describe findings", height=150, value="AC system: 12 years old, condenser fan motor seized. System low on refrigerant. Frequent breakdowns last 2 summers.")
    postal_code = st.text_input("Postal Code", value="N2J 3C3")
    system_age = st.text_input("System age", value="Unknown")
    brand_model = st.text_input("Brand/Model (optional)", value="Unknown")
    selected_plan = st.selectbox("Service Plan", options=plan_labels, index=default_index)
    urgency = st.selectbox("Urgency", ["low","normal","high","emergency"], index=1)
    constraints = st.text_input("Constraints (budget/financing)", value="none")
    st.subheader("Optional Override (promos/adjustments)")
    business_override = st.text_area("Only use when you need to override defaults (optional).", height=80, value="")    
    submitted = st.form_submit_button("Generate Option Sheet & Script", type="primary", use_container_width=True)

if submitted:
    payload = {
        "free_text_diagnosis": free_text_diagnosis,
        "postal_code": postal_code,
        "system_age": system_age,
        "brand_model": brand_model,
        "plan_status": selected_plan,
        "urgency": urgency,
        "constraints": constraints,
        "business_override": business_override
    }
    try:
        g = requests.post(f"{backend_url}/generate", json=payload, timeout=60)
        g.raise_for_status()
        data = g.json()
        tss = data.get("tech_sales_script", {})

        r = requests.post(f"{backend_url}/render-html", json=data, timeout=30)
        r.raise_for_status()
        html = r.json().get("html","")
        html = html.replace('href="/static/', f'href="{backend_url}/static/')
        html = html.replace('src="/static/',  f'src="{backend_url}/static/')

        st.session_state.result_payload = data
        st.session_state.result_tss = tss
        st.session_state.result_html = html
        st.session_state.result_pdf = fetch_pdf(data)
        st.success("Generated successfully.")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.result_html:
    if st.session_state.result_pdf:
        st.download_button(
            "Download option sheet (PDF)",
            key="download_btn",
            data=st.session_state.result_pdf,
            file_name="option_sheet.pdf",
            mime="application.pdf",
            use_container_width=False
        )
    else:
        st.error(f"PDF error: {r.status_code} – {r.text[:300]}")  
    
    tab1, tab2 = st.tabs(["Customer Option Sheet", "Tech Script"])
    with tab1:
        html = st.session_state.result_html
        st.components.v1.html(html, height=900, scrolling=True)
    with tab2:
        tss = st.session_state.result_tss
        st.write("**Opening:**", tss.get("opening",""))
        st.write("**Premium Overview:**", tss.get("top_anchor",""))
        st.write("**Mid-Range Option:**", tss.get("middle_move",""))
        st.write("**Basic Repair:**", tss.get("anchor_low",""))
        st.write("**Objection Handling:**")
        for obj in tss.get("objection_handling", []):
            st.write(f"- **{obj.get('objection','')}** — {obj.get('response','')}")
        st.write("**Closing:**", tss.get("close_prompt",""))
        st.write("**Escalate:**", tss.get("escalate", False))

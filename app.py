
import json
import pathlib
import streamlit as st

DEFAULT_RULES_PATHS = [
    pathlib.Path("eligibility_rules.json"),
    pathlib.Path("/mnt/data/eligibility_bundle/eligibility_rules.json"),
]

def load_rules():
    for p in DEFAULT_RULES_PATHS:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f), str(p)
    st.error("eligibility_rules.json not found. Place it next to app.py or in /mnt/data/eligibility_bundle/")
    st.stop()

rules, rules_path = load_rules()

st.set_page_config(page_title="Eligibility Checker", page_icon="✅", layout="centered")
st.title("Eligibility Checker (No-Interview Workflow)")
st.caption(f"Rules loaded from: {rules_path} — version {rules.get('version')} (updated {rules.get('updated')})")

st.markdown("""
This demo interface implements an automated eligibility flow without interviews.
It evaluates **Inclusion**, **Exclusion**, and **Safety** rules, and (optionally) a 14-day **Run-in** adherence check.
""")

with st.sidebar:
    st.header("How to use")
    st.write("1) Fill out the *Pre-screen* and *Baseline* tabs.")
    st.write("2) Click **Evaluate eligibility**.")
    st.write("3) (Optional) Enter run-in adherence and re-evaluate.")
    st.write("All thresholds can be edited in eligibility_rules.json.")

tab1, tab2, tab3 = st.tabs(["Pre-screen", "Baseline & Safety", "Run-in (optional)"])

with tab1:
    st.subheader("Pre-screen")
    age = st.number_input("Age", min_value=0, max_value=120, value=19, step=1)
    enrollment_status = st.selectbox("Current enrollment at UPR-RP?", options=["active", "inactive"], index=0)
    email = st.text_input("Institutional email (@upr.edu / @uprrp.edu)", value="jserra@upr.edu")
    consent = st.selectbox("I have read and accept the informed consent.", options=["yes", "no"], index=0)
    failed_attention_checks = st.number_input("Failed attention checks", min_value=0, max_value=10, value=0, step=1)
    response_time_seconds = st.number_input("Average response time (seconds)", min_value=0, value=90, step=5)
    min_time_seconds = st.number_input("Configured minimum reasonable time per page (seconds)", min_value=0, value=60, step=5)

with tab2:
    st.subheader("Baseline instruments (self-report)")
    phq9_total = st.slider("PHQ-9 total", 0, 27, 8)
    phq9_item9 = st.slider("PHQ-9 item 9 (suicidality)", 0, 3, 0)
    gad7_total = st.slider("GAD-7 total", 0, 21, 9)
    ders_e_percentile = st.slider("DERS-E percentile", 0, 100, 60)

    st.markdown("**CCAPS-62**")
    ccaps_t_max = st.slider("Max T-score across relevant subscales", 30, 100, 65)

    st.markdown("**C-SSRS (self-report)**")
    cssrs_level = st.selectbox(
        "Highest risk level indicated",
        options=["none", "ideation", "ideation_with_plan", "recent_attempt"],
        index=0
    )

    st.divider()
    st.subheader("Recent clinical context (self-report)")
    med_change_days = st.number_input("Days since last psychotropic medication change", min_value=0, value=120, step=5)
    psych_hosp_days = st.number_input("Days since last psychiatric hospitalization", min_value=0, value=365, step=5)

with tab3:
    st.subheader("Run-in adherence (14 days)")
    ema_completion_rate = st.slider("EMA completion rate", 0.0, 1.0, 0.75, 0.01)
    app_usage_days = st.slider("App usage days", 0, 14, 12)

def email_domain_ok(e):
    e = e.strip().lower()
    return e.endswith("upr.edu") or e.endswith("uprrp.edu")

def compute_flags():
    inc_age = (age >= 18)
    inc_enroll = (enrollment_status == "active")
    inc_email = email_domain_ok(email)

    exc_phq9 = (phq9_total >= rules["scales"]["phq9"]["severe_cut"]) or (phq9_item9 > 0)
    exc_gad7 = (gad7_total >= rules["scales"]["gad7"]["severe_cut"])
    exc_ccaps = (ccaps_t_max >= rules["scales"]["ccaps"]["t_score_max_exclusion"])
    exc_cssrs = (cssrs_level in ["ideation_with_plan", "recent_attempt"])
    exc_med = (med_change_days < 30)
    exc_hosp = (psych_hosp_days < 60)
    exc_quality = (failed_attention_checks >= 2) or (response_time_seconds < min_time_seconds)

    safety = exc_cssrs

    inclusion_all = inc_age and inc_enroll and inc_email and (consent == "yes")
    exclusion_any = exc_phq9 or exc_gad7 or exc_ccaps or exc_cssrs or exc_med or exc_hosp or exc_quality

    runin_ok = (ema_completion_rate >= rules["run_in"]["adherence_threshold"]) and (app_usage_days >= 10)

    flags = {
        "inclusion_all": inclusion_all,
        "exclusion_any": exclusion_any,
        "safety": safety,
        "runin_ok": runin_ok,
        "details": {
            "inc_age": inc_age,
            "inc_enroll": inc_enroll,
            "inc_email": inc_email,
            "exc_phq9": exc_phq9,
            "exc_gad7": exc_gad7,
            "exc_ccaps": exc_ccaps,
            "exc_cssrs": exc_cssrs,
            "exc_med": exc_med,
            "exc_hosp": exc_hosp,
            "exc_quality": exc_quality,
        }
    }
    return flags

def decide(flags):
    if flags["safety"]:
        return "NotEligible_Safety", rules["messages"]["safety"]
    if flags["inclusion_all"] and not flags["exclusion_any"]:
        if flags["runin_ok"]:
            return "Eligible", rules["messages"]["eligible"]
        else:
            return "ProvisionalEligibility", "Provisional eligibility granted. Complete the 14-day run-in to finalize."
    else:
        return "NotEligible", rules["messages"]["not_eligible"]

st.divider()
if st.button("Evaluate eligibility", type="primary"):
    flags = compute_flags()
    status, message = decide(flags)

    st.subheader("Decision")
    st.markdown(f"**Status:** `{status}`")
    st.info(message)

    if flags["safety"]:
        st.error("Safety trigger active (C-SSRS). In PR, Línea PAS: 1-800-981-0023 (24/7). If in immediate danger, call 9-1-1.")

    with st.expander("Show rule evaluation details"):
        st.write(flags["details"])

    if status == "ProvisionalEligibility":
        st.markdown("""
        **Next steps:** Track EMA prompts and app usage during the 14-day run-in.
        This app will mark you as **Eligible** once `EMA ≥ 70%` and `usage days ≥ 10`.
        """)

st.caption("© Eligibility demo • thresholds editable in eligibility_rules.json")

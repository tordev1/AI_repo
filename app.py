"""
Student Success Copilot — Streamlit Application
Hybrid AI system combining search-based planning, rule-based expert system,
and machine learning for student risk assessment and study scheduling.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from ai.rule_engine import assess_student, explain_risk
from ai.search_planner import create_schedule, format_schedule, DAYS, Subject
from ai.ml_model import load_model, predict_risk, get_feature_importance, MODEL_PATH
from ai.fuzzy_logic import FUZZY_AVAILABLE, get_simulator, fuzzy_risk_assessment

# ── Page Config ───────────────────────────────────────────────

st.set_page_config(page_title="Student Success Copilot", page_icon="🎓", layout="wide")
st.title("🎓 Student Success Copilot")
st.caption("A hybrid AI system using Search, Rule-Based Reasoning, and Machine Learning")

# ── Session State Init ────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "student_data" not in st.session_state:
    st.session_state.student_data = {}
if "step" not in st.session_state:
    st.session_state.step = "input"  # input -> subjects -> results

# ── Load ML Model ─────────────────────────────────────────────

@st.cache_resource
def get_ml_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("ML model not found. Run `python ai/data_generator.py` then `python ai/ml_model.py` first.")
        return None, None
    return load_model()


# ── Helper: Chat Message ──────────────────────────────────────

def add_bot_message(text):
    st.session_state.messages.append({"role": "assistant", "content": text})


def add_user_message(text):
    st.session_state.messages.append({"role": "user", "content": text})


# ── Helper Functions ──────────────────────────────────────────

def _display_schedule(schedule):
    """Render a schedule as a streamlit table."""
    rows = []
    for day in DAYS:
        slots = schedule[day]
        if slots:
            merged = {}
            for subj, hours in slots:
                merged[subj] = merged.get(subj, 0) + hours
            cell = ", ".join(f"{s} ({h:.0f}h)" for s, h in merged.items())
        else:
            cell = "Free"
        rows.append({"Day": day, "Schedule": cell})
    st.table(pd.DataFrame(rows))


def _generate_summary(sd, ml_pred, rule_result):
    """Generate a natural language summary combining all AI components."""
    risk = rule_result["risk_level"]
    risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    if ml_pred not in ("unknown",) and ml_pred != risk:
        risk_text = (
            f"The ML model predicts **{ml_pred}** risk while the rule engine assesses **{risk}** risk. "
            f"Taking both into account, "
        )
        risk_order = {"low": 0, "medium": 1, "high": 2}
        combined = max(ml_pred, risk, key=lambda x: risk_order.get(x, 1))
        risk_text += f"your overall risk level is {risk_colors.get(combined, '')} **{combined.upper()}**."
    else:
        combined = risk
        risk_text = f"Your overall risk level is {risk_colors.get(risk, '')} **{risk.upper()}**."

    lines = [f"### Summary\n\n{risk_text}\n"]

    if combined == "high":
        lines.append(
            "**Action needed:** You are at high risk of falling behind. "
            "Focus on your nearest deadline first and consider speaking to your tutor."
        )
    elif combined == "medium":
        lines.append(
            "**Heads up:** You have some risk factors to watch. "
            "Stick to your study plan and try to manage stress proactively."
        )
    else:
        lines.append(
            "**Looking good!** You are on track. Keep up your current study routine "
            "and maintain a healthy balance."
        )

    if rule_result["recommendations"]:
        lines.append("\n**Key recommendations:**")
        for rec in rule_result["recommendations"]:
            lines.append(f"- {rec}")

    return "\n".join(lines)


# ── Display Chat History ──────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Main Interface ────────────────────────────────────────────

if st.session_state.step == "input":
    st.subheader("Step 1: Tell me about yourself")
    st.markdown("Fill in your details below. The system will assess your risk and create a study plan.")

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["male", "female", "non-binary"])
            workload = st.slider("Number of active modules/subjects", 1, 6, 3)
            study_hours = st.slider("Study hours per week", 2, 40, 12)

        with col2:
            confidence = st.slider("Confidence level (1-10)", 1.0, 10.0, 6.0, 0.5)
            stress = st.slider("Stress level (1-10)", 1.0, 10.0, 5.0, 0.5)
            days_to_deadline = st.slider("Days to nearest deadline", 1, 60, 14)

        submitted = st.form_submit_button("Analyse My Situation", type="primary")

        if submitted:
            # Validate — ask follow-up if needed
            missing = []
            if study_hours < 2:
                missing.append("study hours seem very low")
            if workload < 1:
                missing.append("workload cannot be zero")

            if missing:
                msg = "I need a bit more information:\n" + "\n".join(f"- {m}" for m in missing)
                add_bot_message(msg)
            else:
                st.session_state.student_data = {
                    "gender": gender,
                    "workload": workload,
                    "study_hours": study_hours,
                    "confidence": confidence,
                    "stress": stress,
                    "days_to_deadline": days_to_deadline,
                }
                add_user_message(
                    f"Gender: {gender}, Workload: {workload} modules, "
                    f"Study: {study_hours}h/week, Confidence: {confidence}, "
                    f"Stress: {stress}, Deadline: {days_to_deadline} days"
                )
                st.session_state.step = "subjects"
                st.rerun()

elif st.session_state.step == "subjects":
    st.subheader("Step 2: Your Subjects")
    st.markdown("Add the subjects you are studying this week so I can create a schedule.")

    num_subjects = st.number_input("How many subjects?", min_value=1, max_value=6,
                                   value=min(st.session_state.student_data.get("workload", 3), 6))

    subjects_data = []
    with st.form("subjects_form"):
        for i in range(int(num_subjects)):
            st.markdown(f"**Subject {i+1}**")
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name", value=f"Subject {i+1}", key=f"subj_name_{i}")
            hours = c2.number_input("Hours needed this week", 1, 20, 3, key=f"subj_hours_{i}")
            diff = c3.slider("Difficulty (1-10)", 1, 10, 5, key=f"subj_diff_{i}")
            subjects_data.append({
                "name": name,
                "hours_needed": hours,
                "days_to_deadline": st.session_state.student_data["days_to_deadline"],
                "difficulty": diff,
            })

        submitted = st.form_submit_button("Generate Study Plan", type="primary")
        if submitted:
            st.session_state.subjects_data = subjects_data
            st.session_state.step = "results"
            st.rerun()

    if st.button("← Back to Step 1"):
        st.session_state.step = "input"
        st.rerun()

elif st.session_state.step == "results":
    sd = st.session_state.student_data
    subjects_data = st.session_state.get("subjects_data", [])

    # ── ML Prediction ─────────────────────────────────────
    st.subheader("🤖 ML Risk Prediction (Random Forest)")

    risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}

    clf, le = get_ml_model()
    if clf is not None:
        ml_pred, ml_probs = predict_risk(
            clf, le, sd["gender"], sd["workload"], sd["study_hours"],
            sd["confidence"], sd["stress"], sd["days_to_deadline"]
        )

        st.markdown(f"**Predicted Risk Level:** {risk_colors.get(ml_pred, '')} **{ml_pred.upper()}**")

        prob_df = pd.DataFrame([ml_probs])
        st.bar_chart(prob_df.T, horizontal=True)

        with st.expander("Feature Importance"):
            fi = get_feature_importance(clf)
            fi_df = pd.DataFrame(fi, columns=["Feature", "Importance"])
            st.bar_chart(fi_df.set_index("Feature"))
    else:
        ml_pred = "unknown"

    # ── Rule-Based Expert System ──────────────────────────
    st.subheader("📋 Rule-Based Expert System")

    rule_result = assess_student(sd)

    st.markdown(f"**Rule-Based Risk Level:** {risk_colors.get(rule_result['risk_level'], '')} **{rule_result['risk_level'].upper()}**")

    st.markdown("**Recommendations:**")
    for rec in rule_result["recommendations"]:
        st.markdown(f"- {rec}")

    with st.expander("Fired Rules (Forward Chaining)"):
        for rid, desc in rule_result["fired_rules"]:
            st.markdown(f"- **[{rid}]** {desc}")

    with st.expander("Explanation (Backward Chaining)"):
        explanations = explain_risk(rule_result["all_facts"], rule_result["risk_level"])
        if explanations:
            for exp in explanations:
                st.markdown(f"- {exp}")
        else:
            st.markdown("No specific backward chaining trace for this risk level. "
                        "The risk was determined by the combination of individual factor assessments.")
            # Show derived facts instead
            derived = {k: v for k, v in rule_result["all_facts"].items() if k not in sd}
            for k, v in derived.items():
                st.markdown(f"- **{k}**: {v}")

    # ── Fuzzy Logic (Optional) ────────────────────────────
    if FUZZY_AVAILABLE:
        st.subheader("🔮 Fuzzy Logic Assessment")
        sim = get_simulator()
        fuzzy_result = fuzzy_risk_assessment(sd["stress"], sd["confidence"], sim)
        if fuzzy_result:
            st.markdown(
                f"**Fuzzy Risk Score:** {fuzzy_result['risk_score']}/10 "
                f"→ **{fuzzy_result['risk_category'].upper()}**"
            )
            st.markdown(
                f"_Based on fuzzy membership of stress={sd['stress']} "
                f"and confidence={sd['confidence']}_"
            )
        else:
            st.info("Fuzzy inference could not compute a result for these inputs.")
    else:
        st.info("Fuzzy logic module not available (install scikit-fuzzy for this feature).")

    # ── Search-Based Planner ──────────────────────────────
    st.subheader("📅 Search-Based Study Planner")

    if subjects_data:
        max_daily = max(2, sd["study_hours"] // 5)  # rough daily budget

        with st.spinner("Running A* and Greedy search..."):
            plan_result = create_schedule(subjects_data, available_hours_per_day=max_daily)

        tab1, tab2 = st.tabs(["A* Search", "Greedy Best-First Search"])

        with tab1:
            st.markdown("**A* Search** uses f(n) = g(n) + h(n) — considers both path cost and heuristic.")
            if plan_result["astar"]["schedule"]:
                _display_schedule(plan_result["astar"]["schedule"])
            else:
                st.warning("A* could not find a schedule within the search limit. Try increasing daily hours.")
            st.metric("Nodes Explored", plan_result["astar"]["stats"]["nodes_explored"])

        with tab2:
            st.markdown("**Greedy Best-First** uses f(n) = h(n) only — faster but may not find optimal schedule.")
            if plan_result["greedy"]["schedule"]:
                _display_schedule(plan_result["greedy"]["schedule"])
            else:
                st.warning("Greedy could not find a schedule within the search limit.")
            st.metric("Nodes Explored", plan_result["greedy"]["stats"]["nodes_explored"])

        with st.expander("Algorithm Comparison"):
            comp_data = {
                "Metric": ["Nodes Explored", "Considers Path Cost", "Optimal"],
                "A*": [
                    plan_result["astar"]["stats"]["nodes_explored"],
                    "Yes (g(n) + h(n))",
                    "Yes (with admissible heuristic)",
                ],
                "Greedy Best-First": [
                    plan_result["greedy"]["stats"]["nodes_explored"],
                    "No (h(n) only)",
                    "No (can get stuck in local optima)",
                ],
            }
            st.table(pd.DataFrame(comp_data))
    else:
        st.info("No subjects were provided for scheduling.")

    # ── Combined Summary ──────────────────────────────────
    st.subheader("📊 Combined Assessment Summary")

    # Generate natural language advice
    summary = _generate_summary(sd, ml_pred, rule_result)
    st.markdown(summary)
    add_bot_message(summary)

    # Reset button
    if st.button("🔄 Start New Assessment"):
        st.session_state.step = "input"
        st.session_state.student_data = {}
        st.session_state.messages = []
        st.rerun()

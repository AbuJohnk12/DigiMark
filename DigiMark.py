# -*- coding: utf-8 -*-
"""
SME Strategy Recommender — Streamlit app (OLD OpenAI SDK compatible, 0.28.x)

- No "Micro" in Business Size
- Blank inputs on load (no preview row until valid)
- "Aspirational" trust questions but values are written to original columns
- Optional AI explanation (uses openai==0.28.1 if OPENAI_API_KEY is set)
"""

import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import os

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# -------------------- Env / Keys --------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# -------------------- Page & Styles ----------------
st.set_page_config(page_title="SME Strategy Recommender", page_icon="📊", layout="centered")
st.title("📊 SME Digital Marketing Strategy Recommender")

st.markdown(
    """
    <style>
      .tooltip { position:relative; display:inline-flex; align-items:center; gap:4px; }
      .tooltip sup { background:#444; color:#fff; border-radius:50%; padding:0 6px; }
      .tooltip .tooltiptext {
        visibility:hidden; width:220px; background:#555; color:#fff; text-align:center;
        border-radius:6px; padding:8px; position:absolute; z-index:1; bottom:130%; left:50%;
        transform:translateX(-50%); opacity:0; transition:opacity .2s;
      }
      .tooltip:hover .tooltiptext { visibility:visible; opacity:1; }
    </style>
    """,
    unsafe_allow_html=True,
)

def tip(text: str) -> str:
    return f'<span class="tooltip"><sup>?</sup><span class="tooltiptext">{text}</span></span>'

# -------------------- File helpers -----------------
def pick(path: str) -> str:
    for p in (path, os.path.join(os.getcwd(), path)):
        if os.path.exists(p):
            return p
    return path

@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        model = joblib.load(pick("sme_strategy_model.pkl"))
        label_enc = joblib.load(pick("label_encoder.pkl"))
        with open(pick("feature_columns.json"), "r", encoding="utf-8") as f:
            feature_cols = [c.strip() for c in json.load(f)]
        if hasattr(model, "feature_names_in_"):
            feature_cols = model.feature_names_in_.tolist()
        return model, label_enc, feature_cols, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

model, label_enc, feature_cols, load_err = load_artifacts()
if load_err:
    st.error("Failed to load artifacts:")
    st.error(load_err)
    st.stop()

# -------------------- Feature utils ----------------
def options_from_prefix(prefix: str) -> list[str]:
    return sorted({c[len(prefix):] for c in feature_cols if c.startswith(prefix)})

industry_opts  = options_from_prefix("Industry_")
budget_opts    = options_from_prefix("Budget_")
follower_opts  = options_from_prefix("Followers_")

# Remove "Micro"; keep numeric mapping your model expects
size_map_name_to_val = {"Small": 2, "Medium": 3, "Large": 4}
size_names = list(size_map_name_to_val.keys())

maturity_col = next((c for c in feature_cols if "Digital Marketing Maturity" in c), None)
if not maturity_col:
    st.error("Couldn’t find the Digital Marketing Maturity column in feature_columns.json.")
    st.stop()

def map_budget(v: float) -> str:
    if v < 500:        return "<500"
    elif v <= 1000:    return "500-1000"
    elif v <= 5000:    return "1000-5000"
    else:              return ">5000"

def map_followers(v: float) -> str:
    if v < 500:        return "<500"
    elif v <= 1000:    return "500-1000"
    elif v <= 5000:    return "1000-5000"
    else:              return ">5000"

def _norm(s: str) -> str:
    return str(s or "").lower().strip().replace(" ", "").replace("-", "").replace("_", "")

def set_one_hot(prefix: str, selected: str, all_opts: list[str], row: dict) -> str:
    """Set row[f'{prefix}{mapped}']=1 and return the mapped option."""
    if not all_opts:
        return None
    sv = _norm(selected)

    # exact
    for opt in all_opts:
        if _norm(opt) == sv:
            col = f"{prefix}{opt}"
            if col in row: row[col] = 1
            return opt
    # partial
    for opt in all_opts:
        no = _norm(opt)
        if no in sv or sv in no:
            col = f"{prefix}{opt}"
            if col in row: row[col] = 1
            return opt
    # fallback
    fallback = all_opts[0]
    col = f"{prefix}{fallback}"
    if col in row: row[col] = 1
    return fallback

# ---------- Aspirational trust questions (UI label -> original column) ----------
trust_q_map = {
    "I want customers to perceive our brand as honest and genuine":
        "My customers perceive our brand as honest and genuine",
    "I want our marketing messages to feel consistent and genuine":
        "Our marketing messages feel consistent and genuine",
    "I want customers to believe we deliver real value":
        "Customers believe we care about delivering real value",
    "I want customers to consider our brand reliable based on our communications":
        "Customers consider our brand reliable based on our communications",
    "I want our marketing to build trust in our brand":
        "Our marketing has built trust in our brand",
    "I want customers to feel confident recommending our brand":
        "Customers feel confident recommending our brand",
    "I want our marketing to generate steady customer responses":
        "Our marketing generates steady customer responses",
    "I want our promotions to receive meaningful clicks and comments":
        "We receive meaningful comments and clicks on our promotion",
    "I want our campaigns to result in repeat interactions or interest":
        "Our campaigns result in repeat interactions or interest",
    "I want the returns to justify the marketing budget spent":
        "I feel the returns justify the marketing budget spent",
    "I want our campaigns to drive cost-effective customer attention":
        "Our campaigns drive cost-effective customer attention",
    "I want marketing to contribute positively to sales or customer growth":
        "Marketing contributes positively to sales or customer growth",
    "I am willing to switch from my current method if data shows a better approach":
        "If data showed a better marketing approach, I would switch from my current method",
}

# -------------------- UI ------------------------
st.subheader("Enter your business details")
c1, c2 = st.columns(2)

# ---- Business Size with placeholder ----
size_placeholder = "— Select —"
with c1:
    size_choice = st.selectbox("Business Size", [size_placeholder] + size_names, key="size_name")
    st.markdown(
        f'Digital Marketing Maturity {tip("How effectively your business uses digital tools (1=basic, 5=advanced)")}',
        unsafe_allow_html=True,
    )
    maturity = st.slider("Rate from 1 to 5", 1, 5, 3, key="maturity")

# ---- Blank numeric inputs (strings) ----
with c2:
    budget_str    = st.text_input("Monthly Marketing Budget (€)", value="", placeholder="e.g., 1500", key="budget_num")
    followers_str = st.text_input("Total Social Followers",       value="", placeholder="e.g., 1200", key="followers_num")

# ---- Industry with placeholder + Others ----
industry_display = ["— Select —"] + industry_opts + ["Others"]
industry_choice  = st.selectbox("Industry", industry_display, key="industry_choice")
custom_industry  = st.text_input("If Others, specify", key="industry_other").strip() if industry_choice == "Others" else ""
industry_ui      = (custom_industry if (industry_choice == "Others" and custom_industry)
                    else ("" if industry_choice == "— Select —" else industry_choice))

# ---- Aspirational trust questions (write to original column names) ----
with st.expander("🔍 Additional Trust Questions (Optional)", expanded=False):
    trust_responses = {}
    for ui_label, col_name in trust_q_map.items():
        if col_name in feature_cols:
            val = st.slider(ui_label, 1, 5, 3, key=f"trust_{_norm(col_name)}")
            trust_responses[col_name] = val

def reset_form():
    for k in ["size_name", "budget_num", "followers_num", "industry_choice", "industry_other"]:
        if k in st.session_state:
            del st.session_state[k]
    # trust sliders reset implicitly when expander is reopened

st.button("🔄 Reset Form", on_click=reset_form)

# -------------------- Validation & parsing -------
errors = []
valid  = True

# size
if size_choice == size_placeholder:
    valid = False
    errors.append("Please choose a Business Size.")

# budget
budget_val: float | None = None
if budget_str.strip() == "":
    valid = False
    errors.append("Please enter Monthly Marketing Budget.")
else:
    try:
        budget_val = float(budget_str)
        if budget_val < 0: raise ValueError()
    except Exception:
        valid = False
        errors.append("Budget must be a non-negative number.")

# followers
followers_val: float | None = None
if followers_str.strip() == "":
    valid = False
    errors.append("Please enter Total Social Followers.")
else:
    try:
        followers_val = float(followers_str)
        if followers_val < 0: raise ValueError()
    except Exception:
        valid = False
        errors.append("Followers must be a non-negative number.")

# industry
if not industry_ui:
    valid = False
    errors.append("Please choose (or type) an Industry.")

# show warnings only if user started typing/selecting
if not valid and any([budget_str, followers_str, industry_choice != "— Select —", size_choice != size_placeholder]):
    for e in errors:
        st.warning(e)

# -------------------- Build row & preview (only when valid) ----
X_new = None
if valid:
    budget_cat    = map_budget(budget_val)
    followers_cat = map_followers(followers_val)

    row = {c: 0 for c in feature_cols}
    if "Business Size" in row:
        row["Business Size"] = size_map_name_to_val[size_choice]
    row[maturity_col] = int(maturity)

    # trust answers (by original column name)
    for colname, v in trust_responses.items():
        row[colname] = v

    industry_mapped  = set_one_hot("Industry_",  industry_ui,  industry_opts,  row)
    budget_mapped    = set_one_hot("Budget_",    budget_cat,   budget_opts,    row)
    followers_mapped = set_one_hot("Followers_", followers_cat, follower_opts,  row)

    st.caption(f"Mapped → Industry: **{industry_mapped}**, Budget: **{budget_mapped}**, Followers: **{followers_mapped}**")

    X_new = pd.DataFrame([row], columns=feature_cols)

    st.subheader("Input Data for Prediction")
    st.dataframe(X_new, use_container_width=True)

# -------------------- Save current run ----------
def append_submission(pred_label: str, conf_pct: float | None,
                      budget_val: float, followers_val: float,
                      budget_cat: str, followers_cat: str,
                      size_choice: str, industry_ui: str, industry_mapped: str):
    try:
        record = {c: 0 for c in feature_cols if c not in ["Business Size", maturity_col, "Primary Marketing Strategy"]}
        record["Business Size"] = size_map_name_to_val[size_choice]
        record[maturity_col] = int(maturity)
        record["Primary Marketing Strategy"] = pred_label

        for q in trust_q_map.values():
            if q in feature_cols and q in row:
                record[q] = row[q]

        if industry_mapped:
            c = f"Industry_{industry_mapped}"
            if c in record: record[c] = 1
        if budget_mapped:
            c = f"Budget_{budget_mapped}"
            if c in record: record[c] = 1
        if followers_mapped:
            c = f"Followers_{followers_mapped}"
            if c in record: record[c] = 1

        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record["Digital Marketing Maturity"] = int(maturity)
        record["Industry (input)"] = industry_ui
        record["Industry (mapped)"] = industry_mapped
        record["Budget"] = float(budget_val)
        record["Followers"] = float(followers_val)
        record["Predicted Strategy"] = pred_label
        record["Confidence"] = conf_pct
        record["Budget Category"] = budget_cat
        record["Followers Category"] = followers_cat

        df_new = pd.DataFrame([record])
        for c in feature_cols:
            if c not in df_new.columns: df_new[c] = 0
        df_new = df_new[feature_cols + [
            "Digital Marketing Maturity", "Industry (input)", "Industry (mapped)",
            "Budget", "Followers", "Predicted Strategy", "Confidence",
            "Budget Category", "Followers Category", "timestamp"
        ]]

        path = "processed_dataset.csv"
        if os.path.exists(path):
            existing = pd.read_csv(path)
            out = pd.concat([existing, df_new], ignore_index=True)
        else:
            out = df_new
        out.to_csv(path, index=False)
    except Exception as e:
        st.error(f"Failed to save: {e}")
        st.error(traceback.format_exc())

# -------------------- AI Suggestion (OLD SDK) -----
def ai_explain(strategy: str, proba_dict: dict | None):
    """Optional: use old OpenAI SDK (0.28.x). Skips gracefully if key/pkg missing."""
    if not OPENAI_KEY:
        st.info("ℹ️ AI suggestions disabled (no OPENAI_API_KEY set).")
        return
    try:
        import openai
        openai.api_key = OPENAI_KEY
    except Exception as e:
        st.info(f"ℹ️ AI suggestions disabled ({e}).")
        return

    profile = {
        "predicted_strategy": strategy,
        "class_probabilities": proba_dict,
        "industry_input": industry_ui,
        "industry_mapped": industry_mapped,
        "business_size": size_choice,
        "digital_maturity": int(maturity),
        "budget_eur": float(budget_val),
        "followers": int(followers_val),
        "readiness_avg": float(np.mean(list(trust_responses.values()))) if trust_responses else None,
    }

    sys_msg = (
        "You are a practical marketing analyst. Explain concisely why the predicted strategy fits now, "
        "then provide 3 specific next steps for a 4–6 week pilot. Do not change the predicted label."
    )
    user_msg = "Business profile JSON:\n" + json.dumps(profile, indent=2)

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user",   "content": user_msg}],
            temperature=0.3,
            max_tokens=400,
        )
        text = resp.choices[0].message["content"].strip()
        with st.expander("🧠 Suggestion & Why (AI-assisted)", expanded=True):
            st.markdown(text)
    except Exception as e:
        st.info(f"(AI suggestion failed: {e})")

# -------------------- Predict -------------------
if st.button("🔮 Get Recommendation"):
    if not valid:
        st.error("Please complete all required fields before getting a recommendation.")
    else:
        try:
            pred_idx = model.predict(X_new)[0]
            strategy = label_enc.inverse_transform([pred_idx])[0]

            proba_dict = None
            conf = None
            if hasattr(model, "predict_proba"):
                proba_vec = model.predict_proba(X_new)[0]
                class_names = label_enc.inverse_transform(model.classes_)
                proba_dict = {str(c): float(p) for c, p in zip(class_names, proba_vec)}
                conf = float(proba_vec[pred_idx]) * 100.0

            st.success(f"✅ Recommended Strategy: **{strategy}**")
            if conf is not None:
                st.write(f"Confidence: **{conf:.1f}%**")

            append_submission(strategy, conf,
                              budget_val, followers_val,
                              budget_cat, followers_cat,
                              size_choice, industry_ui, industry_mapped)

            ai_explain(strategy, proba_dict)
            st.session_state.strategy = strategy
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.error(traceback.format_exc())

# -------------------- Feedback ------------------
def save_feedback(strategy: str, fb: str):
    try:
        r = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy,
            "feedback": fb,
            "business_size": size_choice,
            "industry_input": industry_ui,
            "industry_mapped": industry_mapped if valid else None,
        }
        fp = "feedback_data.csv"
        pd.DataFrame([r]).to_csv(fp, index=False, mode="a", header=not os.path.exists(fp))
    except Exception as e:
        st.error(f"Could not save feedback: {e}")

if "strategy" in st.session_state:
    st.write("Was this recommendation helpful?")
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("👍 Helpful", use_container_width=True, type="primary"):
            save_feedback(st.session_state["strategy"], "Like")
            st.success("Thanks for the feedback! ❤️")
    with cc2:
        if st.button("👎 Needs work", use_container_width=True):
            save_feedback(st.session_state["strategy"], "Dislike")
            st.info("Thanks! We’ll improve it.")

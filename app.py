
"""
app.py

Sticky Note Generation Dashboard
Run:
    streamlit run app.py
"""

import re
import streamlit as st
import pandas as pd
import numpy as np

from data.sample_meetings import MEETINGS
from features import extract_features, to_matrix, FEATURE_COLUMNS
from classifier import (
    train_final_model,
    leave_one_meeting_out_eval,
    feature_importances,
)
from generate_notes import generate_sticky_notes
from evaluate import evaluate_notes


# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Sticky Note Generator",
    page_icon="🗒️",
    layout="wide",
)

st.title("🗒️ Sticky Note Generation from Casual Meeting Conversations")
st.caption(
    "Extract important conversational turns, classify important content and generate concise sticky notes."
)

# ----------------------------------------------------
# CSS (ONLY CSS here)
# ----------------------------------------------------

st.markdown(
    """
<style>

/* Sticky note card */

.sticky-card{
    padding:18px;
    border-radius:12px;
    margin-bottom:18px;
    min-height:140px;
    box-shadow:3px 3px 8px rgba(0,0,0,.15);
}

.sticky-title{
    font-size:17px;
    font-weight:bold;
}

.sticky-meta{
    font-size:11px;
    color:#555;
    margin:6px 0 12px 0;
}

.sticky-body{
    font-size:14px;
    line-height:1.55;
}

</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Input")

mode = st.sidebar.radio(
    "Choose input",
    ["Sample meeting", "Paste your own transcript"],
)

if mode == "Sample meeting":

    meeting_name = st.sidebar.selectbox(
        "Sample meeting",
        list(MEETINGS.keys()),
    )

    turns = MEETINGS[meeting_name]["turns"]
    reference_notes = MEETINGS[meeting_name]["reference_notes"]

else:

    st.sidebar.write("Format each line as:")

    st.sidebar.code("Speaker: text")

    raw = st.sidebar.text_area(
        "Transcript",
        value="Aditi: Let's get started.\nRavi: I finished the report and will send it tomorrow.",
        height=220,
    )

    turns = []

    for i, line in enumerate(raw.splitlines()):

        line = line.strip()

        if not line:
            continue

        m = re.match(r"^([^:]+):\s*(.+)$", line)

        if m:
            turns.append((i * 5, m.group(1), m.group(2)))
        else:
            turns.append((i * 5, "Speaker", line))

    reference_notes = []

threshold = st.sidebar.slider(
    "Importance probability threshold",
    0.10,
    0.90,
    0.42,
    0.02,
)

top_k = st.sidebar.number_input(
    "Force top-K notes (0 = threshold)",
    min_value=0,
    max_value=50,
    value=0,
)

# ----------------------------------------------------
# Validation
# ----------------------------------------------------

if len(turns) < 2:
    st.warning("Please provide at least two conversational turns.")
    st.stop()

# ----------------------------------------------------
# Cache expensive operations
# ----------------------------------------------------


@st.cache_resource
def load_model():
    return train_final_model()


@st.cache_data
def run_cv():
    return leave_one_meeting_out_eval()


@st.cache_data
def get_importance():
    return feature_importances()


# ----------------------------------------------------
# Run pipeline
# ----------------------------------------------------

with st.spinner("Generating sticky notes..."):

    feats = extract_features(turns)
    X = to_matrix(feats)

    clf = load_model()

    notes, probs = generate_sticky_notes(
        turns,
        feats,
        clf,
        X,
        prob_threshold=threshold,
        top_k=(top_k if top_k > 0 else None),
    )

# ----------------------------------------------------
# Summary
# ----------------------------------------------------

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Turns", len(turns))
c2.metric("Notes", len(notes))
c3.metric("Threshold", threshold)

if len(probs):
    c4.metric("Highest Confidence", f"{max(probs):.3f}")

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📌 Sticky Notes",
        "🔎 Feature Analysis",
        "📊 Model Evaluation",
        "📄 Transcript",
    ]
)

# ====================================================
# TAB 1
# ====================================================

with tab1:

    st.subheader(f"Generated Sticky Notes ({len(notes)})")

    if not notes:

        st.info("No important turns found.")

    else:

        colors = [
            "#FFF6A5",
            "#B9F6CA",
            "#FFCCBC",
            "#B3E5FC",
            "#F8BBD0",
            "#D1C4E9",
        ]

        cols = st.columns(3)

        for i, note in enumerate(notes):

            color = colors[i % len(colors)]

            with cols[i % 3]:

                st.markdown(
                    f"""
<div class="sticky-card" style="background:{color};">

<div class="sticky-title">
📌 {note["speaker"]}
</div>

<div class="sticky-meta">
Turns {note["turn_range"]} • Confidence {note["confidence"]}
</div>

<div class="sticky-body">
{note["note"]}
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

    # Evaluation

    if reference_notes and notes:

        st.divider()

        st.subheader("Evaluation vs Human Reference")

        note_texts = [n["note"] for n in notes]

        per_note, agg = evaluate_notes(
            note_texts,
            reference_notes,
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("ROUGE-1", agg["rouge1"])
        m2.metric("ROUGE-2", agg["rouge2"])
        m3.metric("ROUGE-L", agg["rougeL"])
        m4.metric("BLEU", agg["bleu"])
        m5.metric("METEOR", agg["meteor"])

        with st.expander("Human Reference Notes"):

            for r in reference_notes:
                st.write("•", r)

        with st.expander("Per-note Scores"):
            st.dataframe(pd.DataFrame(per_note), use_container_width=True)

# ====================================================
# TAB 2
# ====================================================

with tab2:

    st.subheader("Extracted Features")

    df = pd.DataFrame(feats)

    df["importance_prob"] = np.round(probs, 3)

    st.dataframe(
        df[
            [
                "idx",
                "speaker",
                "text",
                "importance_prob",
            ]
            + FEATURE_COLUMNS
        ],
        use_container_width=True,
    )

    st.divider()

    st.subheader("Feature Importances")

    imp, _ = get_importance()

    imp_df = pd.DataFrame(
        imp,
        columns=["Feature", "Importance"],
    )

    st.dataframe(imp_df, use_container_width=True)

    st.bar_chart(imp_df.set_index("Feature"))

# ====================================================
# TAB 3
# ====================================================

with tab3:

    st.subheader("Classifier Performance")

    cv, overall = run_cv()

    st.dataframe(pd.DataFrame(cv), use_container_width=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Precision", overall["precision"])
    c2.metric("Recall", overall["recall"])
    c3.metric("F1", overall["f1"])
    c4.metric("Accuracy", overall["accuracy"])

# ====================================================
# TAB 4
# ====================================================

with tab4:

    st.subheader("Full Transcript")

    for ts, spk, txt in turns:

        st.write(f"**[{ts:>4}s] {spk}:** {txt}")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.divider()

st.caption(
    "Sticky Note Generation Project • YAKE • TF-IDF • NER • Random Forest • ROUGE • BLEU • METEOR"
)
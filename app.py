"""
app.py

Sticky Note Generation Dashboard

Features:
1. Sample meeting input
2. Paste transcript input
3. Upload audio
4. Whisper audio transcription
5. Audio transcription evaluation
   - WER
   - CER
   - BLEU
   - ROUGE-1
   - ROUGE-2
   - ROUGE-L
   - METEOR
6. Important-content classification
7. Sticky note generation
8. Feature analysis
9. Model evaluation

Run:
    python -m streamlit run app.py
"""

# ============================================================
# IMPORTS
# ============================================================

import re
import html
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from data.sample_meetings import MEETINGS

from features import (
    extract_features,
    to_matrix,
    FEATURE_COLUMNS,
)

from classifier import (
    train_final_model,
    leave_one_meeting_out_eval,
    feature_importances,
)

from generate_notes import generate_sticky_notes

from evaluate import evaluate_notes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sticky Note Generator",
    page_icon="🗒️",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🗒️ Sticky Note Generation from Casual Meeting Conversations"
)

st.caption(
    "Extract important conversational turns, classify important "
    "content and generate concise sticky notes."
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   Sticky Note
========================================================= */

.sticky-card {
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
    min-height: 140px;
    box-shadow: 3px 3px 8px rgba(0,0,0,.15);
}

.sticky-title {
    font-size: 17px;
    font-weight: bold;
}

.sticky-meta {
    font-size: 11px;
    color: #555;
    margin: 6px 0 12px 0;
}

.sticky-body {
    font-size: 14px;
    line-height: 1.55;
}


/* =========================================================
   Audio Evaluation
========================================================= */

.audio-metric {
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    background-color: #f5f5f5;
    margin-bottom: 10px;
}

.audio-metric-value {
    font-size: 25px;
    font-weight: bold;
}

.audio-metric-label {
    font-size: 13px;
    color: #666;
}


/* =========================================================
   Section
========================================================= */

.section-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f7f7f7;
    margin-bottom: 15px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "audio_transcript" not in st.session_state:
    st.session_state.audio_transcript = ""

if "audio_reference" not in st.session_state:
    st.session_state.audio_reference = ""

if "audio_uploaded_name" not in st.session_state:
    st.session_state.audio_uploaded_name = ""

if "audio_evaluation" not in st.session_state:
    st.session_state.audio_evaluation = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Input")

mode = st.sidebar.radio(
    "Choose input",
    [
        "Sample meeting",
        "Paste your own transcript",
        "Upload audio",
    ],
)


# ============================================================
# VARIABLES
# ============================================================

turns = []
reference_notes = []


# ============================================================
# OPTION 1 — SAMPLE MEETING
# ============================================================

if mode == "Sample meeting":

    meeting_name = st.sidebar.selectbox(
        "Sample meeting",
        list(MEETINGS.keys()),
    )

    turns = MEETINGS[meeting_name]["turns"]

    reference_notes = MEETINGS[
        meeting_name
    ]["reference_notes"]


# ============================================================
# OPTION 2 — PASTE TRANSCRIPT
# ============================================================

elif mode == "Paste your own transcript":

    st.sidebar.write(
        "Format each line as:"
    )

    st.sidebar.code(
        "Speaker: text"
    )

    raw = st.sidebar.text_area(
        "Transcript",
        value=(
            "Aditi: Let's get started.\n"
            "Ravi: I finished the report and will send it tomorrow."
        ),
        height=220,
    )

    for i, line in enumerate(raw.splitlines()):

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^([^:]+):\s*(.+)$",
            line,
        )

        if match:

            turns.append(
                (
                    i * 5,
                    match.group(1).strip(),
                    match.group(2).strip(),
                )
            )

        else:

            turns.append(
                (
                    i * 5,
                    "Speaker",
                    line,
                )
            )

    reference_notes = []


# ============================================================
# OPTION 3 — UPLOAD AUDIO
# ============================================================

else:

    st.sidebar.subheader(
        "🎵 Upload Audio"
    )

    uploaded_audio = st.sidebar.file_uploader(
        "Upload a meeting/conversation audio file",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "flac",
        ],
        help=(
            "Supported formats: WAV, MP3, M4A, OGG and FLAC."
        ),
    )

    if uploaded_audio is not None:

        st.sidebar.success(
            f"Uploaded: {uploaded_audio.name}"
        )

        st.sidebar.audio(
            uploaded_audio
        )

        # ----------------------------------------------------
        # TRANSCRIBE
        # ----------------------------------------------------

        if st.sidebar.button(
            "🎙️ Transcribe Audio",
            use_container_width=True,
        ):

            try:

                import whisper

                suffix = Path(
                    uploaded_audio.name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_audio:

                    temp_audio.write(
                        uploaded_audio.getbuffer()
                    )

                    audio_path = temp_audio.name

                with st.spinner(
                    "🎙️ Transcribing audio with Whisper..."
                ):

                    model = whisper.load_model(
                        "base"
                    )

                    result = model.transcribe(
                        audio_path
                    )

                    transcript = result.get(
                        "text",
                        "",
                    ).strip()

                st.session_state.audio_transcript = (
                    transcript
                )

                st.session_state.audio_uploaded_name = (
                    uploaded_audio.name
                )

                st.session_state.audio_evaluation = None

                st.sidebar.success(
                    "✅ Transcription completed!"
                )

            except ImportError:

                st.sidebar.error(
                    "Whisper is not installed."
                )

                st.sidebar.code(
                    "python -m pip install openai-whisper"
                )

            except Exception as error:

                st.sidebar.error(
                    f"Transcription failed: {error}"
                )

    # --------------------------------------------------------
    # REFERENCE TRANSCRIPT
    # --------------------------------------------------------

    st.sidebar.divider()

    st.sidebar.subheader(
        "📄 Reference Transcript"
    )

    reference_file = st.sidebar.file_uploader(
        "Upload reference transcript",
        type=["txt"],
        key="reference_transcript",
        help=(
            "Upload the manually prepared correct transcript "
            "for evaluating Whisper."
        ),
    )

    if reference_file is not None:

        try:

            reference_text = (
                reference_file
                .getvalue()
                .decode("utf-8")
            )

            st.session_state.audio_reference = (
                reference_text
            )

            st.sidebar.success(
                "Reference transcript loaded."
            )

        except Exception as error:

            st.sidebar.error(
                f"Could not read reference file: {error}"
            )


# ============================================================
# AUDIO TRANSCRIPT → TURNS
# ============================================================

if (
    mode == "Upload audio"
    and st.session_state.audio_transcript
):

    st.divider()

    st.header(
        "🎙️ Audio Transcription"
    )

    st.write(
        f"**Audio:** "
        f"{st.session_state.audio_uploaded_name}"
    )

    edited_transcript = st.text_area(
        "Generated Transcript",
        value=st.session_state.audio_transcript,
        height=250,
        key="editable_audio_transcript",
        help=(
            "You can edit the transcript. "
            "The edited transcript will be used for "
            "sticky-note generation."
        ),
    )

    st.session_state.audio_transcript = (
        edited_transcript
    )

    # --------------------------------------------------------
    # Convert transcript into conversational turns
    # --------------------------------------------------------

    turns = []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        edited_transcript,
    )

    for i, sentence in enumerate(sentences):

        sentence = sentence.strip()

        if sentence:

            turns.append(
                (
                    i * 5,
                    "Speaker",
                    sentence,
                )
            )

    reference_notes = []


# ============================================================
# SIDEBAR — IMPORTANCE SETTINGS
# ============================================================

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


# ============================================================
# VALIDATION
# ============================================================

if len(turns) < 2:

    if mode == "Upload audio":

        st.info(
            "👈 Upload an audio file and click "
            "**Transcribe Audio** to begin."
        )

    else:

        st.warning(
            "Please provide at least two conversational turns."
        )

    st.stop()


# ============================================================
# CACHE EXPENSIVE OPERATIONS
# ============================================================

@st.cache_resource
def load_model():

    return train_final_model()


@st.cache_data
def run_cv():

    return leave_one_meeting_out_eval()


@st.cache_data
def get_importance():

    return feature_importances()


# ============================================================
# RUN NLP PIPELINE
# ============================================================

with st.spinner(
    "Generating sticky notes..."
):

    feats = extract_features(
        turns
    )

    X = to_matrix(
        feats
    )

    clf = load_model()

    notes, probs = generate_sticky_notes(
        turns,
        feats,
        clf,
        X,
        prob_threshold=threshold,
        top_k=(
            top_k
            if top_k > 0
            else None
        ),
    )


# ============================================================
# SUMMARY
# ============================================================

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Turns",
    len(turns),
)

c2.metric(
    "Notes",
    len(notes),
)

c3.metric(
    "Threshold",
    threshold,
)

if len(probs):

    c4.metric(
        "Highest Confidence",
        f"{max(probs):.3f}",
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📌 Sticky Notes",
        "🎙️ Audio Evaluation",
        "🔎 Feature Analysis",
        "📊 Model Evaluation",
        "📄 Transcript",
    ]
)


# ============================================================
# TAB 1 — STICKY NOTES
# ============================================================

with tab1:

    st.subheader(
        f"Generated Sticky Notes ({len(notes)})"
    )

    if not notes:

        st.info(
            "No important turns found."
        )

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

            color = colors[
                i % len(colors)
            ]

            # Safely escape generated text
            speaker = html.escape(
                str(note.get("speaker", "Speaker"))
            )

            turn_range = html.escape(
                str(note.get("turn_range", ""))
            )

            confidence = html.escape(
                str(note.get("confidence", ""))
            )

            note_text = html.escape(
                str(note.get("note", ""))
            )

            with cols[i % 3]:

                st.markdown(
                    f"""
<div class="sticky-card"
     style="background:{color};">

<div class="sticky-title">
📌 {speaker}
</div>

<div class="sticky-meta">
Turns {turn_range}
&nbsp; • &nbsp;
Confidence {confidence}
</div>

<div class="sticky-body">
{note_text}
</div>

</div>
""",
                    unsafe_allow_html=True,
                )


    # --------------------------------------------------------
    # Evaluation against human reference notes
    # --------------------------------------------------------

    if reference_notes and notes:

        st.divider()

        st.subheader(
            "Evaluation vs Human Reference"
        )

        note_texts = [
            n["note"]
            for n in notes
        ]

        per_note, agg = evaluate_notes(
            note_texts,
            reference_notes,
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "ROUGE-1",
            agg["rouge1"],
        )

        m2.metric(
            "ROUGE-2",
            agg["rouge2"],
        )

        m3.metric(
            "ROUGE-L",
            agg["rougeL"],
        )

        m4.metric(
            "BLEU",
            agg["bleu"],
        )

        m5.metric(
            "METEOR",
            agg["meteor"],
        )

        with st.expander(
            "Human Reference Notes"
        ):

            for r in reference_notes:

                st.write(
                    "•",
                    r,
                )

        with st.expander(
            "Per-note Scores"
        ):

            st.dataframe(
                pd.DataFrame(per_note),
                use_container_width=True,
            )


# ============================================================
# TAB 2 — AUDIO EVALUATION
# ============================================================

with tab2:

    st.subheader(
        "🎙️ Audio Transcription Evaluation"
    )

    if mode != "Upload audio":

        st.info(
            "Select **Upload audio** from the sidebar "
            "to evaluate an audio transcription."
        )

    elif not st.session_state.audio_transcript:

        st.info(
            "Upload an audio file and click "
            "**Transcribe Audio** first."
        )

    else:

        st.write(
            f"**Audio:** "
            f"{st.session_state.audio_uploaded_name}"
        )

        st.write(
            f"**Generated transcript length:** "
            f"{len(st.session_state.audio_transcript.split())} words"
        )

        # ----------------------------------------------------
        # Generated transcript
        # ----------------------------------------------------

        with st.expander(
            "View Generated Transcript"
        ):

            st.text(
                st.session_state.audio_transcript
            )

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

        if not st.session_state.audio_reference:

            st.warning(
                "Upload a `.txt` reference transcript "
                "from the sidebar to calculate evaluation metrics."
            )

        else:

            reference_text = (
                st.session_state.audio_reference
            )

            hypothesis_text = (
                st.session_state.audio_transcript
            )

            with st.expander(
                "View Reference Transcript"
            ):

                st.text(
                    reference_text
                )

            # =================================================
            # METRIC FUNCTIONS
            # =================================================

            def normalize_text(text):

                text = str(text).lower()

                text = re.sub(
                    r"[^\w\s]",
                    " ",
                    text,
                )

                text = re.sub(
                    r"\s+",
                    " ",
                    text,
                )

                return text.strip()


            def levenshtein(
                reference,
                hypothesis,
            ):

                rows = len(reference) + 1
                cols = len(hypothesis) + 1

                matrix = [
                    [0] * cols
                    for _ in range(rows)
                ]

                for i in range(rows):
                    matrix[i][0] = i

                for j in range(cols):
                    matrix[0][j] = j

                for i in range(1, rows):

                    for j in range(1, cols):

                        if (
                            reference[i - 1]
                            == hypothesis[j - 1]
                        ):

                            cost = 0

                        else:

                            cost = 1

                        matrix[i][j] = min(
                            matrix[i - 1][j] + 1,
                            matrix[i][j - 1] + 1,
                            matrix[i - 1][j - 1] + cost,
                        )

                return matrix[-1][-1]


            def calculate_wer(
                reference,
                hypothesis,
            ):

                ref = normalize_text(
                    reference
                ).split()

                hyp = normalize_text(
                    hypothesis
                ).split()

                if not ref:

                    return 0.0

                distance = levenshtein(
                    ref,
                    hyp,
                )

                return distance / len(ref)


            def calculate_cer(
                reference,
                hypothesis,
            ):

                ref = normalize_text(
                    reference
                )

                hyp = normalize_text(
                    hypothesis
                )

                if not ref:

                    return 0.0

                distance = levenshtein(
                    list(ref),
                    list(hyp),
                )

                return distance / len(ref)


            def calculate_bleu(
                reference,
                hypothesis,
            ):

                try:

                    from nltk.translate.bleu_score import (
                        sentence_bleu,
                        SmoothingFunction,
                    )

                    ref_tokens = (
                        normalize_text(
                            reference
                        ).split()
                    )

                    hyp_tokens = (
                        normalize_text(
                            hypothesis
                        ).split()
                    )

                    if (
                        not ref_tokens
                        or not hyp_tokens
                    ):

                        return 0.0

                    return sentence_bleu(
                        [ref_tokens],
                        hyp_tokens,
                        weights=(
                            0.25,
                            0.25,
                            0.25,
                            0.25,
                        ),
                        smoothing_function=(
                            SmoothingFunction().method1
                        ),
                    )

                except Exception:

                    return 0.0


            def calculate_rouge(
                reference,
                hypothesis,
            ):

                try:

                    from rouge_score import rouge_scorer

                    scorer = rouge_scorer.RougeScorer(
                        [
                            "rouge1",
                            "rouge2",
                            "rougeL",
                        ],
                        use_stemmer=True,
                    )

                    scores = scorer.score(
                        reference,
                        hypothesis,
                    )

                    return {
                        "rouge1": scores[
                            "rouge1"
                        ].fmeasure,

                        "rouge2": scores[
                            "rouge2"
                        ].fmeasure,

                        "rougeL": scores[
                            "rougeL"
                        ].fmeasure,
                    }

                except Exception:

                    return {
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                    }


            def calculate_meteor(
                reference,
                hypothesis,
            ):

                try:

                    from nltk.translate.meteor_score import (
                        meteor_score,
                    )

                    ref_tokens = (
                        normalize_text(
                            reference
                        ).split()
                    )

                    hyp_tokens = (
                        normalize_text(
                            hypothesis
                        ).split()
                    )

                    if (
                        not ref_tokens
                        or not hyp_tokens
                    ):

                        return 0.0

                    return meteor_score(
                        [ref_tokens],
                        hyp_tokens,
                    )

                except Exception:

                    return 0.0


            # =================================================
            # CALCULATE
            # =================================================

            wer = calculate_wer(
                reference_text,
                hypothesis_text,
            )

            cer = calculate_cer(
                reference_text,
                hypothesis_text,
            )

            bleu = calculate_bleu(
                reference_text,
                hypothesis_text,
            )

            rouge = calculate_rouge(
                reference_text,
                hypothesis_text,
            )

            meteor = calculate_meteor(
                reference_text,
                hypothesis_text,
            )


            # =================================================
            # DISPLAY
            # =================================================

            st.subheader(
                "Transcription Accuracy"
            )

            m1, m2, m3, m4, m5 = st.columns(5)

            m1.metric(
                "WER",
                f"{wer:.3f}",
                help="Lower is better.",
            )

            m2.metric(
                "CER",
                f"{cer:.3f}",
                help="Lower is better.",
            )

            m3.metric(
                "BLEU",
                f"{bleu:.3f}",
                help="Higher is better.",
            )

            m4.metric(
                "ROUGE-L",
                f"{rouge['rougeL']:.3f}",
                help="Higher is better.",
            )

            m5.metric(
                "METEOR",
                f"{meteor:.3f}",
                help="Higher is better.",
            )


            # =================================================
            # ROUGE DETAILS
            # =================================================

            st.subheader(
                "ROUGE Scores"
            )

            r1, r2, rl = st.columns(3)

            r1.metric(
                "ROUGE-1",
                f"{rouge['rouge1']:.3f}",
            )

            r2.metric(
                "ROUGE-2",
                f"{rouge['rouge2']:.3f}",
            )

            rl.metric(
                "ROUGE-L",
                f"{rouge['rougeL']:.3f}",
            )


            # =================================================
            # INTERPRETATION
            # =================================================

            st.subheader(
                "Interpretation"
            )

            if wer <= 0.10:

                wer_comment = "Excellent transcription accuracy."

            elif wer <= 0.20:

                wer_comment = "Good transcription accuracy."

            elif wer <= 0.30:

                wer_comment = "Moderate transcription error."

            else:

                wer_comment = "High transcription error."


            st.info(
                f"""
**WER:** {wer:.3f} — {wer_comment}

**CER:** {cer:.3f} — lower values indicate fewer character errors.

**BLEU:** {bleu:.3f} — higher values indicate greater n-gram overlap.

**ROUGE-1:** {rouge['rouge1']:.3f}

**ROUGE-2:** {rouge['rouge2']:.3f}

**ROUGE-L:** {rouge['rougeL']:.3f}

**METEOR:** {meteor:.3f} — higher values indicate better similarity.
"""
            )


# ============================================================
# TAB 3 — FEATURE ANALYSIS
# ============================================================

with tab3:

    st.subheader(
        "Extracted Features"
    )

    df = pd.DataFrame(
        feats
    )

    df["importance_prob"] = np.round(
        probs,
        3,
    )

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

    st.subheader(
        "Feature Importances"
    )

    imp, _ = get_importance()

    imp_df = pd.DataFrame(
        imp,
        columns=[
            "Feature",
            "Importance",
        ],
    )

    st.dataframe(
        imp_df,
        use_container_width=True,
    )

    st.bar_chart(
        imp_df.set_index(
            "Feature"
        )
    )


# ============================================================
# TAB 4 — MODEL EVALUATION
# ============================================================

with tab4:

    st.subheader(
        "Classifier Performance"
    )

    cv, overall = run_cv()

    st.dataframe(
        pd.DataFrame(cv),
        use_container_width=True,
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precision",
        overall["precision"],
    )

    c2.metric(
        "Recall",
        overall["recall"],
    )

    c3.metric(
        "F1",
        overall["f1"],
    )

    c4.metric(
        "Accuracy",
        overall["accuracy"],
    )


# ============================================================
# TAB 5 — TRANSCRIPT
# ============================================================

with tab5:

    st.subheader(
        "Full Transcript"
    )

    for ts, spk, txt in turns:

        st.write(
            f"**[{ts:>4}s] {spk}:** {txt}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sticky Note Generation Project • "
    "YAKE • TF-IDF • NER • Random Forest • "
    "Whisper • WER • CER • ROUGE • BLEU • METEOR"
)
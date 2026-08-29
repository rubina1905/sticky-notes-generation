"""
generate_notes.py

Task (ii) continued - Sticky Note Generation.

Pipeline:

1. Extract features from meeting turns.
2. Use the importance classifier to calculate the probability
   that each turn is sticky-note worthy.
3. Select important turns using either:
      - an importance probability threshold, or
      - Top-K selection.
4. Merge consecutive important turns from the same speaker.
5. Compress the selected content into concise sticky notes.

The goal is to preserve:
    - decisions
    - actions
    - deadlines
    - commitments
    - problems
    - solutions
    - important project information

while removing:
    - greetings
    - filler words
    - acknowledgements
    - conversational noise.
"""

import re
import numpy as np


# ============================================================
# FILLER / DISCOURSE PATTERNS
# ============================================================

FILLER_PATTERNS = [

    # Common conversational fillers
    r"\byeah\b",
    r"\byep\b",
    r"\buh+\b",
    r"\bum+\b",
    r"\bhmm+\b",
    r"\ber+\b",

    # Discourse markers
    r"^\s*so\b,?\s*",
    r"^\s*okay\b,?\s*",
    r"^\s*ok\b,?\s*",
    r"^\s*well\b,?\s*",
    r"^\s*actually\b,?\s*",
    r"^\s*basically\b,?\s*",
    r"^\s*right\b,?\s*",

    # Conversational phrases
    r"\byou know\b",
    r"\bi mean\b",
    r"\bgreat\b,?\s*",
    r"\bnice\b,?\s*",
    r"\bcool\b,?\s*",
    r"\bgood\b,?\s*",

    # Agreement / acknowledgement
    r"\bgot it\b",
    r"\bsounds good\b",
    r"\bthanks\b",
    r"\bthank you\b",
]


# ============================================================
# LOW-INFORMATION ACKNOWLEDGEMENTS
# ============================================================

LOW_INFORMATION_PHRASES = {
    "agreed",
    "agree",
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "sounds good",
    "got it",
    "thanks",
    "thank you",
    "see ya",
    "see you",
    "cool",
    "nice",
    "great",
}


# ============================================================
# CLEAN FILLER
# ============================================================

def _clean_filler(text):
    """
    Remove common filler and discourse-marker words.

    Parameters
    ----------
    text : str
        Raw conversational utterance.

    Returns
    -------
    str
        Cleaner version of the utterance.
    """

    t = str(text).strip()

    for pattern in FILLER_PATTERNS:

        t = re.sub(
            pattern,
            "",
            t,
            flags=re.IGNORECASE,
        )

    # Remove repeated whitespace
    t = re.sub(
        r"\s+",
        " ",
        t,
    )

    # Clean punctuation spacing
    t = re.sub(
        r"\s+([,.!?])",
        r"\1",
        t,
    )

    return t.strip(" ,.")


# ============================================================
# CHECK LOW-INFORMATION TURN
# ============================================================

def _is_low_information(text):
    """
    Check whether a turn consists primarily of a simple
    acknowledgement or conversational response.

    This is used as a safety filter during note generation.
    """

    cleaned = re.sub(
        r"[.!?,]",
        "",
        str(text).lower().strip(),
    )

    return cleaned in LOW_INFORMATION_PHRASES


# ============================================================
# NORMALIZE FIRST-PERSON LANGUAGE
# ============================================================

def _normalize_first_person(text, speaker):
    """
    Convert first-person references into a standalone
    third-person style.

    Example:

        "I'll send the report tomorrow."

    becomes:

        "Ravi will send the report tomorrow."
    """

    t = text

    # Contractions first
    t = re.sub(
        r"\bI'll\b",
        f"{speaker} will",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\bI'm\b",
        f"{speaker} is",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\bI've\b",
        f"{speaker} has",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\bI'd\b",
        f"{speaker} would",
        t,
        flags=re.IGNORECASE,
    )

    # Non-contracted forms
    t = re.sub(
        r"\bI will\b",
        f"{speaker} will",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\bI have\b",
        f"{speaker} has",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"\bI am\b",
        f"{speaker} is",
        t,
        flags=re.IGNORECASE,
    )

    # Possessive
    t = re.sub(
        r"\bmy\b",
        f"{speaker}'s",
        t,
        flags=re.IGNORECASE,
    )

    # Remaining standalone I
    t = re.sub(
        r"\bI\b",
        speaker,
        t,
    )

    return t


# ============================================================
# COMPRESS TEXT
# ============================================================

def _compress(text, speaker, max_words=28):
    """
    Convert a raw conversational utterance into a concise
    standalone sticky-note sentence.

    The compression is extractive/rule-based rather than
    generative. Therefore, the original information is largely
    preserved while filler and unnecessary discourse markers
    are removed.
    """

    # --------------------------------------------------------
    # Step 1: Remove filler
    # --------------------------------------------------------

    t = _clean_filler(
        text
    )

    if not t:
        t = str(text).strip()

    # --------------------------------------------------------
    # Step 2: Normalize first-person language
    # --------------------------------------------------------

    t = _normalize_first_person(
        t,
        speaker,
    )

    # --------------------------------------------------------
    # Step 3: Remove repeated spaces
    # --------------------------------------------------------

    t = re.sub(
        r"\s+",
        " ",
        t,
    ).strip()

    # --------------------------------------------------------
    # Step 4: Remove leading punctuation
    # --------------------------------------------------------

    t = t.strip(
        " ,;:-"
    )

    # --------------------------------------------------------
    # Step 5: Limit length
    # --------------------------------------------------------

    words = t.split()

    if len(words) > max_words:

        # Try to cut at a natural punctuation boundary
        shortened = words[:max_words]

        t = " ".join(
            shortened
        )

        # Remove incomplete ending punctuation
        t = t.rstrip(
            ",;:-"
        )

        t += "..."

    # --------------------------------------------------------
    # Step 6: Capitalize
    # --------------------------------------------------------

    if t:

        t = (
            t[0].upper()
            + t[1:]
        )

    # --------------------------------------------------------
    # Step 7: Ensure sentence punctuation
    # --------------------------------------------------------

    if t and not t.endswith(
        (".", "!", "?", "...")
    ):

        t += "."

    return t


# ============================================================
# SELECT IMPORTANT SEGMENTS
# ============================================================

def select_important_segments(
    turns,
    feats,
    clf,
    X,
    prob_threshold=0.42,
    top_k=None,
):
    """
    Select important turns and merge consecutive important
    turns belonging to the same speaker.

    Parameters
    ----------
    turns : list
        Original transcript turns.

    feats : list
        Extracted feature dictionaries.

    clf : classifier
        Trained importance classifier.

    X : numpy.ndarray
        Feature matrix.

    prob_threshold : float
        Probability threshold.

    top_k : int or None
        If supplied, select the K highest-probability turns.

    Returns
    -------
    segments : list
        Tuples containing:

        (
            start_index,
            end_index,
            speaker,
            merged_text,
            average_probability
        )

    probs : numpy.ndarray
        Importance probability for every turn.
    """

    if len(turns) == 0:

        return [], np.array([])

    # ========================================================
    # PREDICT IMPORTANCE PROBABILITIES
    # ========================================================

    probs = clf.predict_proba(
        X
    )[:, 1]

    # ========================================================
    # SELECT IMPORTANT INDICES
    # ========================================================

    if top_k is not None:

        top_k = int(top_k)

        top_k = max(
            1,
            min(
                top_k,
                len(turns),
            ),
        )

        important_indices = (
            np.argsort(
                -probs
            )[:top_k]
        )

        important_idx = set(
            int(i)
            for i in important_indices
        )

    else:

        important_idx = set(
            int(i)
            for i in np.where(
                probs >= prob_threshold
            )[0]
        )

    # ========================================================
    # REMOVE SIMPLE ACKNOWLEDGEMENTS
    # ========================================================

    filtered_idx = set()

    for idx in important_idx:

        text = turns[idx][2]

        # Keep a low-information acknowledgement only if it
        # has an unusually high probability and is the only
        # useful content available.
        #
        # For normal threshold-based operation we exclude it.

        if _is_low_information(text):

            continue

        filtered_idx.add(
            idx
        )

    important_idx = filtered_idx

    # ========================================================
    # BUILD SEGMENTS
    # ========================================================

    segments = []

    i = 0
    n = len(turns)

    while i < n:

        if i not in important_idx:

            i += 1
            continue

        # ----------------------------------------------------
        # Start a new segment
        # ----------------------------------------------------

        start_idx = i

        end_idx = i

        speaker = turns[i][1]

        segment_texts = [
            turns[i][2]
        ]

        segment_probs = [
            probs[i]
        ]

        # ----------------------------------------------------
        # Merge immediately consecutive turns from same
        # speaker.
        # ----------------------------------------------------

        j = i

        while (
            j + 1 in important_idx
            and turns[j + 1][1] == speaker
        ):

            j += 1

            segment_texts.append(
                turns[j][2]
            )

            segment_probs.append(
                probs[j]
            )

        end_idx = j

        # ----------------------------------------------------
        # Create merged segment
        # ----------------------------------------------------

        merged_text = " ".join(
            segment_texts
        )

        average_probability = float(
            np.mean(
                segment_probs
            )
        )

        segments.append(
            (
                start_idx,
                end_idx,
                speaker,
                merged_text,
                average_probability,
            )
        )

        i = j + 1

    return segments, probs


# ============================================================
# GENERATE STICKY NOTES
# ============================================================

def generate_sticky_notes(
    turns,
    feats,
    clf,
    X,
    prob_threshold=0.42,
    top_k=None,
):
    """
    Generate final sticky notes.

    Returns
    -------
    notes : list of dict

        Each dictionary contains:

            speaker
            turn_range
            note
            confidence

    probs : numpy.ndarray
        Importance probability for every turn.
    """

    segments, probs = select_important_segments(
        turns=turns,
        feats=feats,
        clf=clf,
        X=X,
        prob_threshold=prob_threshold,
        top_k=top_k,
    )

    notes = []

    for (
        start_idx,
        end_idx,
        speaker,
        text,
        probability,
    ) in segments:

        note_text = _compress(
            text,
            speaker,
            max_words=28,
        )

        # Avoid empty notes
        if not note_text.strip():
            continue

        notes.append({

            "speaker": speaker,

            "turn_range": (
                start_idx,
                end_idx,
            ),

            "note": note_text,

            "confidence": round(
                probability,
                3,
            ),
        })

    return notes, probs


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    from data.sample_meetings import MEETINGS
    from features import (
        extract_features,
        to_matrix,
    )
    from classifier import (
        train_final_model,
    )

    print("=" * 70)
    print("STICKY NOTE GENERATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Select test meeting
    # --------------------------------------------------------

    meeting = MEETINGS[
        "project_standup"
    ]

    turns = meeting[
        "turns"
    ]

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    features = extract_features(
        turns
    )

    X = to_matrix(
        features
    )

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    clf = train_final_model()

    # --------------------------------------------------------
    # Generate notes
    # --------------------------------------------------------

    notes, probabilities = generate_sticky_notes(
        turns=turns,
        feats=features,
        clf=clf,
        X=X,
        prob_threshold=0.42,
    )

    # --------------------------------------------------------
    # Display notes
    # --------------------------------------------------------

    print(
        f"\nGenerated {len(notes)} sticky notes:\n"
    )

    for note in notes:

        print(
            f"[{note['speaker']}] "
            f"turns={note['turn_range']} "
            f"confidence={note['confidence']}"
        )

        print(
            f"  {note['note']}"
        )

    print(
        "\nSticky-note generation test completed successfully."
    )
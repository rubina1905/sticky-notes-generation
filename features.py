"""
features.py

Task (i) - Identify Important-Content Features.

For every conversational turn, we compute a feature vector capturing:

1. Keyword/keyphrase score using YAKE
2. TF-IDF-based semantic centrality
3. Speaker-turn features
4. Repetition of information
5. Question / answer flags
6. Decision / agreement indicators
7. Action-oriented statement indicators
8. Named-entity indicators
9. Positional/contextual features

The resulting feature vectors are used by the importance classifier
to identify turns that are suitable for sticky-note generation.
"""

import re
import numpy as np
import nltk
import yake

from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# NLTK RESOURCE SETUP
# ============================================================

def ensure_nltk_resources():
    """
    Download required NLTK resources if they are not already
    available.

    This makes the project easier to run on a new machine.
    """

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        (
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng",
        ),
        ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
        ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
        ("corpora/words", "words"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                pass


ensure_nltk_resources()


# ============================================================
# RULE-BASED LEXICONS
# ============================================================

DECISION_LEXICON = [
    "agreed",
    "agree",
    "decided",
    "decision",
    "let's go with",
    "finalize",
    "finalized",
    "we will",
    "we'll",
    "confirmed",
    "sounds good",
    "approved",
    "let's decide",
    "moving forward with",
]


ACTION_LEXICON = [
    "will",
    "need to",
    "needs to",
    "should",
    "must",
    "action item",
    "by tomorrow",
    "by monday",
    "by tuesday",
    "by wednesday",
    "by thursday",
    "by friday",
    "by end of day",
    "deadline",
    "due",
    "assign",
    "i'll",
    "i will",
    "send",
    "push",
    "prepare",
    "review",
    "register",
]


QUESTION_WORDS = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can",
    "could",
    "should",
    "is",
    "are",
    "do",
    "does",
)


# ============================================================
# NAMED ENTITY EXTRACTION
# ============================================================

def _ne_labels(text):
    """
    Return a set of named-entity labels found in the text.

    Examples:
        PERSON
        ORGANIZATION
        GPE

    NLTK's standard NE chunker does not reliably identify dates,
    so dates are additionally detected using a simple regex.
    """

    labels = set()

    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged)

        for subtree in tree:
            if isinstance(subtree, Tree):
                labels.add(subtree.label())

    except Exception:
        # NER should never prevent the complete feature pipeline
        # from running.
        pass

    # --------------------------------------------------------
    # Additional DATE detection
    # --------------------------------------------------------

    date_pattern = (
        r"\b("
        r"monday|tuesday|wednesday|thursday|friday|"
        r"saturday|sunday|"
        r"today|tomorrow|yesterday|"
        r"\d{1,2}(st|nd|rd|th)?"
        r")\b"
    )

    if re.search(date_pattern, text.lower()):
        labels.add("DATE")

    return labels


# ============================================================
# YAKE KEYWORD SCORE
# ============================================================

def _keyword_score(text, kw_extractor):
    """
    Calculate a YAKE-based keyword/keyphrase score.

    YAKE gives lower scores to better keywords.

    We therefore invert the score so that a higher value means
    stronger keyword/keyphrase importance.
    """

    text = text.strip()

    # Very short turns generally do not contain meaningful
    # keyphrases.
    if len(text.split()) < 2:
        return 0.0

    try:
        keywords = kw_extractor.extract_keywords(text)

        if not keywords:
            return 0.0

        inverted_scores = [
            1.0 / (1.0 + score)
            for _, score in keywords
        ]

        return float(np.mean(inverted_scores))

    except Exception:
        return 0.0


# ============================================================
# SAFE TF-IDF
# ============================================================

def _build_tfidf(texts):
    """
    Build a TF-IDF matrix safely.

    English stop-word removal is attempted first.

    If the transcript is too short or contains only stop words,
    a second TF-IDF configuration without stop-word removal is
    used.
    """

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf = vectorizer.fit_transform(texts)

        if tfidf.shape[1] == 0:
            raise ValueError("No TF-IDF vocabulary was created.")

        return tfidf.toarray()

    except ValueError:

        vectorizer = TfidfVectorizer()

        try:
            tfidf = vectorizer.fit_transform(texts)
            return tfidf.toarray()

        except ValueError:

            # Extremely unusual fallback.
            # Return a simple zero matrix so that the rest of
            # the application can still operate.
            return np.zeros(
                (len(texts), 1),
                dtype=float
            )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(turns):
    """
    Extract one feature dictionary per conversational turn.

    Parameters
    ----------
    turns : list
        List of:
            (timestamp, speaker, text)

    Returns
    -------
    list of dict
        Feature dictionaries aligned with the original turn order.
    """

    if not turns:
        return []

    texts = [
        str(t[2])
        for t in turns
    ]

    speakers = [
        str(t[1])
        for t in turns
    ]

    n = len(texts)

    # ========================================================
    # TF-IDF
    # ========================================================

    tfidf_arr = _build_tfidf(texts)

    # ========================================================
    # TF-IDF SEMANTIC CENTRALITY
    # ========================================================

    centroid = tfidf_arr.mean(axis=0)

    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm < 1e-9:
        centrality = np.zeros(n)

    else:
        centrality = (
            tfidf_arr @ centroid
        ) / centroid_norm

        max_centrality = np.max(centrality)

        if max_centrality > 1e-9:
            centrality = (
                centrality / max_centrality
            )

    # ========================================================
    # YAKE
    # ========================================================

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=2,
        top=5,
    )

    # ========================================================
    # FEATURE EXTRACTION PER TURN
    # ========================================================

    feats = []

    for i, (timestamp, speaker, text) in enumerate(turns):

        text = str(text)
        text_l = text.lower().strip()

        word_count = len(text.split())

        # ----------------------------------------------------
        # 1. Keyword / keyphrase score
        # ----------------------------------------------------

        kw_score = _keyword_score(
            text,
            kw_extractor
        )

        # ----------------------------------------------------
        # 2. TF-IDF semantic centrality
        # ----------------------------------------------------

        tfidf_centrality = float(
            centrality[i]
        )

        # ----------------------------------------------------
        # 3. Speaker-turn features
        # ----------------------------------------------------

        speaker_change = (
            1
            if i > 0
            and speakers[i] != speakers[i - 1]
            else 0
        )

        # Short turn = possible acknowledgement,
        # backchannel, or interruption.
        is_short_turn = (
            1
            if word_count <= 4
            else 0
        )

        # Long turn = potentially substantive contribution.
        is_long_turn = (
            1
            if word_count >= 15
            else 0
        )

        # ----------------------------------------------------
        # 4. Repetition of information
        # ----------------------------------------------------

        if i == 0:

            repetition = 0.0

        else:

            current_vector = tfidf_arr[i]

            previous_vectors = tfidf_arr[:i]

            current_norm = np.linalg.norm(
                current_vector
            )

            previous_norms = np.linalg.norm(
                previous_vectors,
                axis=1
            )

            denominator = (
                previous_norms
                * (current_norm + 1e-9)
            )

            denominator[
                denominator < 1e-9
            ] = 1e-9

            similarities = (
                previous_vectors @ current_vector
            ) / denominator

            repetition = float(
                np.max(similarities)
            ) if len(similarities) else 0.0

        # ----------------------------------------------------
        # 5. Question / answer features
        # ----------------------------------------------------

        starts_as_question = text_l.startswith(
            QUESTION_WORDS
        )

        is_question = (
            1
            if "?" in text
            or starts_as_question
            else 0
        )

        # A turn immediately following a question
        # is treated as a possible answer.
        is_answer = (
            1
            if i > 0
            and "?" in texts[i - 1]
            else 0
        )

        # ----------------------------------------------------
        # 6. Decision / agreement
        # ----------------------------------------------------

        is_decision = (
            1
            if any(
                phrase in text_l
                for phrase in DECISION_LEXICON
            )
            else 0
        )

        # ----------------------------------------------------
        # 7. Action-oriented statement
        # ----------------------------------------------------

        is_action = (
            1
            if any(
                phrase in text_l
                for phrase in ACTION_LEXICON
            )
            else 0
        )

        # ----------------------------------------------------
        # 8. Named entities
        # ----------------------------------------------------

        ner_labels = _ne_labels(text)

        has_person = (
            1
            if "PERSON" in ner_labels
            else 0
        )

        has_date = (
            1
            if "DATE" in ner_labels
            else 0
        )

        has_org_gpe = (
            1
            if (
                "ORGANIZATION" in ner_labels
                or "GPE" in ner_labels
            )
            else 0
        )

        ner_count = len(ner_labels)

        # ----------------------------------------------------
        # 9. Positional/contextual feature
        # ----------------------------------------------------

        position_norm = (
            i / max(1, n - 1)
        )

        # ----------------------------------------------------
        # Store feature vector
        # ----------------------------------------------------

        feats.append({

            "idx": i,

            "timestamp": timestamp,

            "speaker": speaker,

            "text": text,

            "word_count": word_count,

            "kw_score": round(
                kw_score,
                4
            ),

            "tfidf_centrality": round(
                tfidf_centrality,
                4
            ),

            "speaker_change": speaker_change,

            "is_short_turn": is_short_turn,

            "is_long_turn": is_long_turn,

            "repetition": round(
                repetition,
                4
            ),

            "is_question": is_question,

            "is_answer": is_answer,

            "is_decision": is_decision,

            "is_action": is_action,

            "has_person": has_person,

            "has_date": has_date,

            "has_org_gpe": has_org_gpe,

            "ner_count": ner_count,

            "position_norm": round(
                position_norm,
                4
            ),
        })

    return feats


# ============================================================
# FEATURES USED BY THE CLASSIFIER
# ============================================================

FEATURE_COLUMNS = [

    "word_count",

    "kw_score",

    "tfidf_centrality",

    "speaker_change",

    "is_short_turn",

    "is_long_turn",

    "repetition",

    "is_question",

    "is_answer",

    "is_decision",

    "is_action",

    "has_person",

    "has_date",

    "has_org_gpe",

    "ner_count",

    "position_norm",
]


# ============================================================
# CONVERT FEATURES TO NUMPY MATRIX
# ============================================================

def to_matrix(feats):
    """
    Convert feature dictionaries into a NumPy matrix.

    The column order is defined by FEATURE_COLUMNS.
    """

    if not feats:
        return np.empty(
            (0, len(FEATURE_COLUMNS)),
            dtype=float
        )

    return np.array(
        [
            [
                float(feature[c])
                for c in FEATURE_COLUMNS
            ]
            for feature in feats
        ],
        dtype=float,
    )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    from data.sample_meetings import MEETINGS

    meeting = MEETINGS["project_standup"]

    features = extract_features(
        meeting["turns"]
    )

    matrix = to_matrix(features)

    print("=" * 70)
    print("FEATURE EXTRACTION TEST")
    print("=" * 70)

    print(
        f"Number of turns : {len(features)}"
    )

    print(
        f"Number of features : {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Matrix shape : {matrix.shape}"
    )

    print("\nFeature columns:")

    for i, feature in enumerate(FEATURE_COLUMNS, 1):
        print(
            f"{i:2d}. {feature}"
        )

    print("\nFirst turn:")

    print(features[0])

    print("\nFeature matrix created successfully.")
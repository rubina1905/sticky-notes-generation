"""
evaluate.py

Task (iii): Performance Evaluation

This module evaluates the generated sticky notes against
human-written reference notes.

Metrics:

1. ROUGE-1
2. ROUGE-2
3. ROUGE-L
4. BLEU
5. METEOR

The classifier itself is evaluated separately in classifier.py
using:

- Precision
- Recall
- F1
- Accuracy
- ROC-AUC

Because the number of generated notes may differ from the
number of reference notes, each generated note is matched
with the reference note having the highest ROUGE-L score.
The final score is the average over generated notes.
"""

import re
import numpy as np

from rouge_score import rouge_scorer

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)

from nltk.translate.meteor_score import meteor_score


# ============================================================
# ROUGE SETUP
# ============================================================

_rouge = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL",
    ],
    use_stemmer=True,
)


# ============================================================
# BLEU SMOOTHING
# ============================================================

_smooth = (
    SmoothingFunction().method1
)


# ============================================================
# TOKENIZATION
# ============================================================

def _tokenize(text):
    """
    Simple regex-based tokenizer.

    We intentionally avoid nltk.word_tokenize() so that the
    evaluation does not depend on additional NLTK tokenizer
    resources such as punkt_tab.
    """

    if text is None:
        return []

    text = str(text).lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b",
        text,
    )

    return tokens


# ============================================================
# SAFE BLEU
# ============================================================

def _calculate_bleu(
    reference_tokens,
    generated_tokens,
):
    """
    Calculate sentence-level BLEU.

    BLEU-4 is used because it considers 1-gram through
    4-gram precision.

    Smoothing is applied because short sticky notes often
    contain few 3-gram or 4-gram matches.
    """

    if not reference_tokens:
        return 0.0

    if not generated_tokens:
        return 0.0

    try:

        score = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            weights=(
                0.25,
                0.25,
                0.25,
                0.25,
            ),
            smoothing_function=_smooth,
        )

        return float(score)

    except Exception:
        return 0.0


# ============================================================
# SAFE METEOR
# ============================================================

def _calculate_meteor(
    reference_tokens,
    generated_tokens,
):
    """
    Calculate METEOR.

    Newer NLTK versions expect tokenized strings rather than
    raw strings, so token lists are passed directly.
    """

    if not reference_tokens:
        return 0.0

    if not generated_tokens:
        return 0.0

    try:

        score = meteor_score(
            [reference_tokens],
            generated_tokens,
        )

        return float(score)

    except Exception:

        return 0.0


# ============================================================
# BEST MATCH
# ============================================================

def _find_best_reference(
    generated_note,
    reference_notes,
):
    """
    Find the reference note with the highest ROUGE-L F1 score.
    """

    if not reference_notes:
        return None, 0.0

    best_reference = None
    best_score = -1.0

    for reference in reference_notes:

        try:

            score = _rouge.score(
                reference,
                generated_note,
            )

            rouge_l = score[
                "rougeL"
            ].fmeasure

        except Exception:

            rouge_l = 0.0

        if rouge_l > best_score:

            best_score = rouge_l
            best_reference = reference

    return (
        best_reference,
        best_score,
    )


# ============================================================
# PER-NOTE EVALUATION
# ============================================================

def _best_match_scores(
    generated_notes,
    reference_notes,
):
    """
    Evaluate every generated note.

    For each generated note:

        1. Find best reference using ROUGE-L.
        2. Calculate ROUGE-1.
        3. Calculate ROUGE-2.
        4. Calculate ROUGE-L.
        5. Calculate BLEU-4.
        6. Calculate METEOR.

    Returns a list containing one dictionary per generated note.
    """

    per_note = []

    # --------------------------------------------------------
    # Handle missing references
    # --------------------------------------------------------

    if not reference_notes:

        for generated in generated_notes:

            per_note.append(
                {
                    "generated": generated,
                    "best_reference": None,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bleu": 0.0,
                    "meteor": 0.0,
                }
            )

        return per_note

    # --------------------------------------------------------
    # Evaluate each generated note
    # --------------------------------------------------------

    for generated in generated_notes:

        generated = str(
            generated
        ).strip()

        # Find best matching reference
        best_reference, _ = (
            _find_best_reference(
                generated,
                reference_notes,
            )
        )

        if best_reference is None:

            per_note.append(
                {
                    "generated": generated,
                    "best_reference": None,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bleu": 0.0,
                    "meteor": 0.0,
                }
            )

            continue

        # ----------------------------------------------------
        # ROUGE
        # ----------------------------------------------------

        rouge_scores = _rouge.score(
            best_reference,
            generated,
        )

        rouge1 = rouge_scores[
            "rouge1"
        ].fmeasure

        rouge2 = rouge_scores[
            "rouge2"
        ].fmeasure

        rouge_l = rouge_scores[
            "rougeL"
        ].fmeasure

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        reference_tokens = _tokenize(
            best_reference
        )

        generated_tokens = _tokenize(
            generated
        )

        # ----------------------------------------------------
        # BLEU-4
        # ----------------------------------------------------

        bleu = _calculate_bleu(
            reference_tokens,
            generated_tokens,
        )

        # ----------------------------------------------------
        # METEOR
        # ----------------------------------------------------

        meteor = _calculate_meteor(
            reference_tokens,
            generated_tokens,
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        per_note.append(
            {
                "generated": generated,

                "best_reference": best_reference,

                "rouge1": round(
                    float(rouge1),
                    3,
                ),

                "rouge2": round(
                    float(rouge2),
                    3,
                ),

                "rougeL": round(
                    float(rouge_l),
                    3,
                ),

                "bleu": round(
                    float(bleu),
                    3,
                ),

                "meteor": round(
                    float(meteor),
                    3,
                ),
            }
        )

    return per_note


# ============================================================
# AGGREGATE EVALUATION
# ============================================================

def evaluate_notes(
    generated_notes,
    reference_notes,
):
    """
    Calculate aggregate evaluation metrics.

    Parameters
    ----------
    generated_notes : list[str]
        Sticky notes produced by the system.

    reference_notes : list[str]
        Human-written reference notes.

    Returns
    -------
    per_note : list[dict]
        Individual generated-note evaluation.

    agg : dict
        Aggregate metrics.
    """

    # --------------------------------------------------------
    # Ensure lists
    # --------------------------------------------------------

    if generated_notes is None:
        generated_notes = []

    if reference_notes is None:
        reference_notes = []

    generated_notes = list(
        generated_notes
    )

    reference_notes = list(
        reference_notes
    )

    # --------------------------------------------------------
    # Per-note scores
    # --------------------------------------------------------

    per_note = _best_match_scores(
        generated_notes,
        reference_notes,
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    if per_note:

        rouge1 = float(
            np.mean(
                [
                    x["rouge1"]
                    for x in per_note
                ]
            )
        )

        rouge2 = float(
            np.mean(
                [
                    x["rouge2"]
                    for x in per_note
                ]
            )
        )

        rouge_l = float(
            np.mean(
                [
                    x["rougeL"]
                    for x in per_note
                ]
            )
        )

        bleu = float(
            np.mean(
                [
                    x["bleu"]
                    for x in per_note
                ]
            )
        )

        meteor = float(
            np.mean(
                [
                    x["meteor"]
                    for x in per_note
                ]
            )
        )

    else:

        rouge1 = 0.0
        rouge2 = 0.0
        rouge_l = 0.0
        bleu = 0.0
        meteor = 0.0

    # --------------------------------------------------------
    # Final dictionary
    # --------------------------------------------------------

    aggregate = {

        "rouge1": round(
            rouge1,
            3,
        ),

        "rouge2": round(
            rouge2,
            3,
        ),

        "rougeL": round(
            rouge_l,
            3,
        ),

        "bleu": round(
            bleu,
            3,
        ),

        "meteor": round(
            meteor,
            3,
        ),

        "num_generated": len(
            generated_notes
        ),

        "num_reference": len(
            reference_notes
        ),
    }

    return (
        per_note,
        aggregate,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STICKY NOTE EVALUATION TEST")
    print("=" * 70)

    generated = [
        "Ravi will push cleaned data to the repository by 6pm today.",
        "Meera will fix the segmentation SQL query by tomorrow.",
        "The team will finalize five business questions by Thursday.",
    ]

    reference = [
        "Ravi to push cleaned data to telco-churn-revenue-at-risk repo by 6pm today.",
        "Meera facing SQL error in segmentation query; fix is adding customer_id to group by clause.",
        "Team agreed to finalize the five business questions by Thursday.",
    ]

    per_note, aggregate = evaluate_notes(
        generated,
        reference,
    )

    print("\nPER-NOTE RESULTS")
    print("-" * 70)

    for i, result in enumerate(
        per_note,
        start=1,
    ):

        print(
            f"\nNote {i}"
        )

        print(
            f"Generated    : "
            f"{result['generated']}"
        )

        print(
            f"Best ref     : "
            f"{result['best_reference']}"
        )

        print(
            f"ROUGE-1      : "
            f"{result['rouge1']}"
        )

        print(
            f"ROUGE-2      : "
            f"{result['rouge2']}"
        )

        print(
            f"ROUGE-L      : "
            f"{result['rougeL']}"
        )

        print(
            f"BLEU-4       : "
            f"{result['bleu']}"
        )

        print(
            f"METEOR       : "
            f"{result['meteor']}"
        )

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(
        f"ROUGE-1 : {aggregate['rouge1']:.3f}"
    )

    print(
        f"ROUGE-2 : {aggregate['rouge2']:.3f}"
    )

    print(
        f"ROUGE-L : {aggregate['rougeL']:.3f}"
    )

    print(
        f"BLEU-4  : {aggregate['bleu']:.3f}"
    )

    print(
        f"METEOR  : {aggregate['meteor']:.3f}"
    )

    print(
        f"\nGenerated notes : "
        f"{aggregate['num_generated']}"
    )

    print(
        f"Reference notes : "
        f"{aggregate['num_reference']}"
    )

    print(
        "\nEvaluation test completed successfully."
    )
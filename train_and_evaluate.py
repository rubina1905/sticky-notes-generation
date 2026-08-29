"""
train_and_evaluate.py

End-to-end runner for the Sticky Note Generation project.

Pipeline:

1. Build the meeting dataset.
2. Evaluate the importance classifier using
   Leave-One-Meeting-Out Cross-Validation.
3. Calculate feature importances.
4. Train the final importance classifier.
5. Generate sticky notes for every sample meeting.
6. Evaluate generated notes against human reference notes.
7. Save the complete results to:

       outputs/report.json
       outputs/sticky_notes.txt
       outputs/feature_importances.csv
       outputs/classifier_cv_results.csv
       outputs/sticky_note_evaluation.csv
"""

import csv
import json
import os

from data.sample_meetings import MEETINGS

from features import (
    extract_features,
    to_matrix,
)

from classifier import (
    leave_one_meeting_out_eval,
    feature_importances,
    train_final_model,
)

from generate_notes import (
    generate_sticky_notes,
)

from evaluate import (
    evaluate_notes,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, filename):
    """
    Save Python dictionary/list as formatted JSON.
    """

    path = os.path.join(
        OUT_DIR,
        filename,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(rows, filename):
    """
    Save list of dictionaries to CSV.
    """

    path = os.path.join(
        OUT_DIR,
        filename,
    )

    if not rows:
        return path

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)
    print("STICKY NOTE GENERATION PROJECT")
    print("END-TO-END TRAINING AND EVALUATION")
    print("=" * 80)

    report = {}

    # ========================================================
    # STEP 1
    # CLASSIFIER EVALUATION
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 1: IMPORTANCE CLASSIFIER EVALUATION")
    print("Leave-One-Meeting-Out Cross-Validation")
    print("=" * 80)

    cv_results, overall = (
        leave_one_meeting_out_eval()
    )

    print()

    for result in cv_results:

        print(
            f"Meeting: "
            f"{result['held_out_meeting']}"
        )

        print(
            f"  Precision : "
            f"{result['precision']:.3f}"
        )

        print(
            f"  Recall    : "
            f"{result['recall']:.3f}"
        )

        print(
            f"  F1        : "
            f"{result['f1']:.3f}"
        )

        print(
            f"  Accuracy  : "
            f"{result['accuracy']:.3f}"
        )

        if "roc_auc" in result:

            print(
                f"  ROC-AUC   : "
                f"{result['roc_auc']:.3f}"
            )

        print()

    print("-" * 80)
    print("OVERALL CLASSIFIER PERFORMANCE")
    print("-" * 80)

    print(
        f"Precision : "
        f"{overall['precision']:.3f}"
    )

    print(
        f"Recall    : "
        f"{overall['recall']:.3f}"
    )

    print(
        f"F1        : "
        f"{overall['f1']:.3f}"
    )

    print(
        f"Accuracy  : "
        f"{overall['accuracy']:.3f}"
    )

    if "roc_auc" in overall:

        print(
            f"ROC-AUC   : "
            f"{overall['roc_auc']:.3f}"
        )

    report[
        "classifier_cv_per_meeting"
    ] = cv_results

    report[
        "classifier_cv_overall"
    ] = overall

    save_csv(
        cv_results,
        "classifier_cv_results.csv",
    )

    # ========================================================
    # STEP 2
    # FEATURE IMPORTANCE
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 2: FEATURE IMPORTANCE")
    print("=" * 80)

    importances, _ = (
        feature_importances()
    )

    feature_rows = []

    for feature, score in importances:

        print(
            f"{feature:<25} "
            f"{score:.4f}"
        )

        feature_rows.append(
            {
                "feature": feature,
                "importance": round(
                    float(score),
                    4,
                ),
            }
        )

    report[
        "feature_importances"
    ] = feature_rows

    save_csv(
        feature_rows,
        "feature_importances.csv",
    )

    # ========================================================
    # STEP 3
    # TRAIN FINAL MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 3: TRAIN FINAL MODEL")
    print("=" * 80)

    final_clf = (
        train_final_model()
    )

    print(
        "Final Random Forest classifier "
        "trained successfully."
    )

    # ========================================================
    # STEP 4
    # GENERATE + EVALUATE NOTES
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 4: GENERATE AND EVALUATE STICKY NOTES")
    print("=" * 80)

    all_notes_txt = []

    per_meeting_report = {}

    evaluation_rows = []

    # --------------------------------------------------------
    # Process every meeting
    # --------------------------------------------------------

    for meeting_name, meeting in (
        MEETINGS.items()
    ):

        print()
        print("-" * 80)
        print(
            f"MEETING: {meeting_name}"
        )
        print("-" * 80)

        turns = meeting[
            "turns"
        ]

        reference_notes = meeting[
            "reference_notes"
        ]

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        feats = extract_features(
            turns
        )

        X = to_matrix(
            feats
        )

        # ----------------------------------------------------
        # Generate notes
        # ----------------------------------------------------

        notes, probabilities = (
            generate_sticky_notes(
                turns=turns,
                feats=feats,
                clf=final_clf,
                X=X,
                prob_threshold=0.42,
            )
        )

        note_texts = [
            n["note"]
            for n in notes
        ]

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        per_note_scores, aggregate = (
            evaluate_notes(
                note_texts,
                reference_notes,
            )
        )

        # ----------------------------------------------------
        # Print generated notes
        # ----------------------------------------------------

        print()
        print(
            f"Generated sticky notes: "
            f"{len(notes)}"
        )

        all_notes_txt.append(
            f"\n=== {meeting_name} ==="
        )

        for note in notes:

            print(
                f"\n[{note['speaker']}] "
                f"Turns: "
                f"{note['turn_range']} "
                f"| Confidence: "
                f"{note['confidence']}"
            )

            print(
                f"  {note['note']}"
            )

            all_notes_txt.append(
                f"- "
                f"[{note['speaker']}] "
                f"(confidence={note['confidence']}) "
                f"{note['note']}"
            )

        # ----------------------------------------------------
        # Print evaluation
        # ----------------------------------------------------

        print()
        print(
            "Evaluation against human reference:"
        )

        print(
            f"  ROUGE-1 : "
            f"{aggregate['rouge1']:.3f}"
        )

        print(
            f"  ROUGE-2 : "
            f"{aggregate['rouge2']:.3f}"
        )

        print(
            f"  ROUGE-L : "
            f"{aggregate['rougeL']:.3f}"
        )

        print(
            f"  BLEU-4  : "
            f"{aggregate['bleu']:.3f}"
        )

        print(
            f"  METEOR  : "
            f"{aggregate['meteor']:.3f}"
        )

        print(
            f"  Generated: "
            f"{aggregate['num_generated']}"
        )

        print(
            f"  Reference: "
            f"{aggregate['num_reference']}"
        )

        # ----------------------------------------------------
        # Save meeting report
        # ----------------------------------------------------

        per_meeting_report[
            meeting_name
        ] = {

            "generated_notes":
                note_texts,

            "reference_notes":
                reference_notes,

            "per_note_scores":
                per_note_scores,

            "aggregate_scores":
                aggregate,
        }

        # ----------------------------------------------------
        # Flatten evaluation for CSV
        # ----------------------------------------------------

        for score in per_note_scores:

            evaluation_rows.append(
                {
                    "meeting":
                        meeting_name,

                    "generated":
                        score["generated"],

                    "best_reference":
                        score[
                            "best_reference"
                        ],

                    "rouge1":
                        score["rouge1"],

                    "rouge2":
                        score["rouge2"],

                    "rougeL":
                        score["rougeL"],

                    "bleu":
                        score["bleu"],

                    "meteor":
                        score["meteor"],
                }
            )

    # ========================================================
    # STEP 5
    # SAVE REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 5: SAVING OUTPUT FILES")
    print("=" * 80)

    report[
        "sticky_notes_by_meeting"
    ] = per_meeting_report

    report[
        "project_information"
    ] = {

        "number_of_meetings":
            len(MEETINGS),

        "model":
            "Random Forest",

        "validation":
            "Leave-One-Meeting-Out Cross-Validation",

        "importance_threshold":
            0.42,

        "evaluation_metrics": [
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
            "ROC-AUC",
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "BLEU-4",
            "METEOR",
        ],
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report_path = save_json(
        report,
        "report.json",
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    notes_path = os.path.join(
        OUT_DIR,
        "sticky_notes.txt",
    )

    with open(
        notes_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                all_notes_txt
            )
        )

    # --------------------------------------------------------
    # Evaluation CSV
    # --------------------------------------------------------

    evaluation_path = save_csv(
        evaluation_rows,
        "sticky_note_evaluation.csv",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("-" * 80)

    print(
        f"Report saved to:\n"
        f"  {report_path}"
    )

    print(
        f"\nSticky notes saved to:\n"
        f"  {notes_path}"
    )

    print(
        f"\nFeature importances saved to:\n"
        f"  {os.path.join(OUT_DIR, 'feature_importances.csv')}"
    )

    print(
        f"\nClassifier CV results saved to:\n"
        f"  {os.path.join(OUT_DIR, 'classifier_cv_results.csv')}"
    )

    print(
        f"\nSticky-note evaluation saved to:\n"
        f"  {evaluation_path}"
    )

    print()
    print("=" * 80)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
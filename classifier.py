"""
classifier.py

Task (ii) - Model Development

This module:

1. Builds the classification dataset from the annotated
   sample meeting corpus.

2. Trains a Random Forest classifier to identify
   sticky-note-worthy conversational turns.

3. Evaluates the classifier using Leave-One-Meeting-Out
   Cross-Validation (LOMO-CV).

4. Calculates:
      - Precision
      - Recall
      - F1-score
      - Accuracy
      - ROC-AUC

5. Calculates feature importance using a Random Forest
   trained on the complete sample corpus.

6. Provides a final model trained on all available
   annotated meetings for use by the Streamlit application.

IMPORTANT:
The current dataset contains only three synthetic meetings.
Therefore, the model is a proof-of-concept and not a
production-level classifier.
"""


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from features import (
    extract_features,
    to_matrix,
    FEATURE_COLUMNS,
)

from data.sample_meetings import MEETINGS


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

RF_ESTIMATORS_CV = 200
RF_ESTIMATORS_FINAL = 300

RF_MAX_DEPTH = 5


# ============================================================
# DATASET CONSTRUCTION
# ============================================================

def build_dataset():
    """
    Build the complete classification dataset.

    Returns
    -------
    dict

        meeting_name ->

        (
            X,
            y,
            features,
            turns
        )

    Where:

        X = feature matrix

        y = binary importance labels

        features = extracted feature dictionaries

        turns = original transcript turns
    """

    dataset = {}

    for meeting_name, meeting in MEETINGS.items():

        turns = meeting["turns"]

        # ----------------------------------------------
        # Extract engineered features
        # ----------------------------------------------

        features = extract_features(
            turns
        )

        X = to_matrix(
            features
        )

        # ----------------------------------------------
        # Create binary labels
        # ----------------------------------------------

        y = np.zeros(
            len(turns),
            dtype=int
        )

        # Mark human-labelled important turns as 1.
        #
        # Only use valid indices to prevent accidental
        # index errors if the annotation file changes.

        for idx in meeting["gold_important_idx"]:

            if 0 <= idx < len(turns):
                y[idx] = 1

        dataset[meeting_name] = (
            X,
            y,
            features,
            turns,
        )

    return dataset


# ============================================================
# MODEL FACTORY
# ============================================================

def _create_model(model_name="random_forest"):
    """
    Create a fresh classifier.

    Parameters
    ----------
    model_name : str
        Either:
            "random_forest"
            "logistic_regression"

    Returns
    -------
    sklearn classifier
    """

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=RF_ESTIMATORS_CV,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    elif model_name == "logistic_regression":

        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    else:

        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Use 'random_forest' or 'logistic_regression'."
        )


# ============================================================
# LEAVE-ONE-MEETING-OUT EVALUATION
# ============================================================

def leave_one_meeting_out_eval(
    model_name="random_forest"
):
    """
    Evaluate the classifier using Leave-One-Meeting-Out
    Cross-Validation.

    With three meetings:

        Fold 1:
            Train -> Meeting 2 + Meeting 3
            Test  -> Meeting 1

        Fold 2:
            Train -> Meeting 1 + Meeting 3
            Test  -> Meeting 2

        Fold 3:
            Train -> Meeting 1 + Meeting 2
            Test  -> Meeting 3

    This evaluates how well the classifier generalizes
    to a completely unseen meeting.

    Returns
    -------
    results : list of dict
        Per-meeting performance.

    overall : dict
        Overall pooled performance.
    """

    dataset = build_dataset()

    meeting_names = list(
        dataset.keys()
    )

    results = []

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    # ========================================================
    # LOOP THROUGH HELD-OUT MEETINGS
    # ========================================================

    for held_out in meeting_names:

        # ----------------------------------------------
        # Training meetings
        # ----------------------------------------------

        train_names = [
            name
            for name in meeting_names
            if name != held_out
        ]

        # ----------------------------------------------
        # Combine training data
        # ----------------------------------------------

        X_train = np.vstack(
            [
                dataset[name][0]
                for name in train_names
            ]
        )

        y_train = np.concatenate(
            [
                dataset[name][1]
                for name in train_names
            ]
        )

        # ----------------------------------------------
        # Test data
        # ----------------------------------------------

        X_test = dataset[held_out][0]

        y_test = dataset[held_out][1]

        # ----------------------------------------------
        # Create model
        # ----------------------------------------------

        clf = _create_model(
            model_name
        )

        # ----------------------------------------------
        # Train
        # ----------------------------------------------

        clf.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------
        # Predictions
        # ----------------------------------------------

        y_pred = clf.predict(
            X_test
        )

        y_prob = clf.predict_proba(
            X_test
        )[:, 1]

        # ==================================================
        # METRICS
        # ==================================================

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        # ROC-AUC requires both classes to be present.
        #
        # Normally this will be true for the meetings in
        # this project, but the check makes the code robust.

        try:

            roc_auc = roc_auc_score(
                y_test,
                y_prob,
            )

        except ValueError:

            roc_auc = 0.0

        # ==================================================
        # SAVE FOLD RESULTS
        # ==================================================

        results.append({

            "held_out_meeting": held_out,

            "precision": round(
                float(precision),
                3,
            ),

            "recall": round(
                float(recall),
                3,
            ),

            "f1": round(
                float(f1),
                3,
            ),

            "accuracy": round(
                float(accuracy),
                3,
            ),

            "roc_auc": round(
                float(roc_auc),
                3,
            ),
        })

        # ==================================================
        # SAVE FOR OVERALL METRICS
        # ==================================================

        all_y_true.extend(
            y_test.tolist()
        )

        all_y_pred.extend(
            y_pred.tolist()
        )

        all_y_prob.extend(
            y_prob.tolist()
        )

    # ========================================================
    # OVERALL POOLED METRICS
    # ========================================================

    overall_precision = precision_score(
        all_y_true,
        all_y_pred,
        zero_division=0,
    )

    overall_recall = recall_score(
        all_y_true,
        all_y_pred,
        zero_division=0,
    )

    overall_f1 = f1_score(
        all_y_true,
        all_y_pred,
        zero_division=0,
    )

    overall_accuracy = accuracy_score(
        all_y_true,
        all_y_pred,
    )

    try:

        overall_roc_auc = roc_auc_score(
            all_y_true,
            all_y_prob,
        )

    except ValueError:

        overall_roc_auc = 0.0

    overall = {

        "precision": round(
            float(overall_precision),
            3,
        ),

        "recall": round(
            float(overall_recall),
            3,
        ),

        "f1": round(
            float(overall_f1),
            3,
        ),

        "accuracy": round(
            float(overall_accuracy),
            3,
        ),

        "roc_auc": round(
            float(overall_roc_auc),
            3,
        ),
    }

    return results, overall


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importances(
    model_name="random_forest"
):
    """
    Train a classifier on all meetings and return
    feature importance values.

    For Random Forest:
        feature_importances_ is used.

    For Logistic Regression:
        absolute coefficient values are used.

    Returns
    -------
    importances : list of tuples

        [
            (feature_name, importance),
            ...
        ]

    clf : trained classifier
    """

    dataset = build_dataset()

    # ----------------------------------------------
    # Combine all meetings
    # ----------------------------------------------

    X = np.vstack(
        [
            value[0]
            for value in dataset.values()
        ]
    )

    y = np.concatenate(
        [
            value[1]
            for value in dataset.values()
        ]
    )

    # ----------------------------------------------
    # Create final model
    # ----------------------------------------------

    if model_name == "random_forest":

        clf = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS_FINAL,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    elif model_name == "logistic_regression":

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    else:

        raise ValueError(
            f"Unknown model_name: {model_name}"
        )

    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    clf.fit(
        X,
        y
    )

    # ----------------------------------------------
    # Calculate feature importance
    # ----------------------------------------------

    if model_name == "random_forest":

        values = clf.feature_importances_

    else:

        values = np.abs(
            clf.coef_[0]
        )

    # ----------------------------------------------
    # Pair feature names with importance
    # ----------------------------------------------

    importances = list(
        zip(
            FEATURE_COLUMNS,
            values,
        )
    )

    # ----------------------------------------------
    # Sort descending
    # ----------------------------------------------

    importances = sorted(
        importances,
        key=lambda item: item[1],
        reverse=True,
    )

    return importances, clf


# ============================================================
# FINAL MODEL FOR APPLICATION
# ============================================================

def train_final_model(
    model_name="random_forest"
):
    """
    Train the final classifier on the complete annotated
    sample corpus.

    This model is used by the Streamlit application to
    generate importance probabilities for new transcripts.
    """

    dataset = build_dataset()

    # ----------------------------------------------
    # Combine all meetings
    # ----------------------------------------------

    X = np.vstack(
        [
            value[0]
            for value in dataset.values()
        ]
    )

    y = np.concatenate(
        [
            value[1]
            for value in dataset.values()
        ]
    )

    # ----------------------------------------------
    # Create model
    # ----------------------------------------------

    if model_name == "random_forest":

        clf = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS_FINAL,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    elif model_name == "logistic_regression":

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    else:

        raise ValueError(
            f"Unknown model_name: {model_name}"
        )

    # ----------------------------------------------
    # Train final model
    # ----------------------------------------------

    clf.fit(
        X,
        y
    )

    return clf


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary():
    """
    Return a simple summary of the annotated dataset.

    Useful for checking class balance.
    """

    dataset = build_dataset()

    summary = []

    for meeting_name, (
        X,
        y,
        features,
        turns,
    ) in dataset.items():

        important = int(
            np.sum(y)
        )

        total = len(y)

        not_important = total - important

        percentage = (
            important / total * 100
            if total > 0
            else 0
        )

        summary.append({

            "meeting": meeting_name,

            "total_turns": total,

            "important_turns": important,

            "non_important_turns": not_important,

            "important_percentage": round(
                percentage,
                2,
            ),
        })

    return summary


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STICKY NOTE IMPORTANCE CLASSIFIER")
    print("=" * 70)

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\nDATASET SUMMARY")
    print("-" * 70)

    summary = dataset_summary()

    for row in summary:

        print(
            f"{row['meeting']:<30} "
            f"Turns={row['total_turns']:<3} "
            f"Important={row['important_turns']:<3} "
            f"Non-important={row['non_important_turns']:<3} "
            f"Important%={row['important_percentage']:.1f}%"
        )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("LEAVE-ONE-MEETING-OUT CROSS-VALIDATION")
    print("=" * 70)

    results, overall = (
        leave_one_meeting_out_eval()
    )

    for result in results:

        print(
            f"\nHeld-out meeting: "
            f"{result['held_out_meeting']}"
        )

        print(
            f"  Precision : {result['precision']:.3f}"
        )

        print(
            f"  Recall    : {result['recall']:.3f}"
        )

        print(
            f"  F1        : {result['f1']:.3f}"
        )

        print(
            f"  Accuracy  : {result['accuracy']:.3f}"
        )

        print(
            f"  ROC-AUC   : {result['roc_auc']:.3f}"
        )

    # ========================================================
    # OVERALL
    # ========================================================

    print("\n" + "-" * 70)
    print("OVERALL PERFORMANCE")
    print("-" * 70)

    print(
        f"Precision : {overall['precision']:.3f}"
    )

    print(
        f"Recall    : {overall['recall']:.3f}"
    )

    print(
        f"F1        : {overall['f1']:.3f}"
    )

    print(
        f"Accuracy  : {overall['accuracy']:.3f}"
    )

    print(
        f"ROC-AUC   : {overall['roc_auc']:.3f}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importances, _ = feature_importances()

    for feature, importance in importances:

        print(
            f"{feature:<22} "
            f"{importance:.4f}"
        )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    model = train_final_model()

    print(
        "Final Random Forest model trained successfully."
    )

    print("\nClassifier test completed successfully.")
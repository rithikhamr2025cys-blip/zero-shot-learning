"""Novel-attack anomaly detection using only known-normal training behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "login_hour",
    "failed_logins_24h",
    "files_accessed_24h",
    "usb_connected",
    "external_email_count_24h",
    "bytes_uploaded_mb_24h",
    "privilege_change",
    "geo_anomaly",
    "unique_hosts_accessed_24h",
    "sensitive_file_access_24h",
]
CATEGORICAL_FEATURES = ["department", "event_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + ["attack_label", "zero_shot_split"]


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot support the intended zero-shot evaluation."""


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    confusion: np.ndarray
    threshold: float
    calibration_normal_count: int
    training_normal_count: int
    predictions: pd.DataFrame
    detections_by_attack: pd.DataFrame


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate schema and enforce disjoint seen/unseen attack labels."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise DatasetValidationError(
            "Missing required columns: " + ", ".join(missing_columns)
        )
    if data.empty:
        raise DatasetValidationError("The uploaded dataset has no rows.")
    if data[["attack_label", "zero_shot_split"]].isna().any().any():
        raise DatasetValidationError(
            "attack_label and zero_shot_split must not contain blank values."
        )

    splits = set(data["zero_shot_split"].astype(str))
    if not {"train_seen", "test_unseen"}.issubset(splits):
        raise DatasetValidationError(
            "zero_shot_split must contain both 'train_seen' and 'test_unseen'."
        )

    train = data[data["zero_shot_split"] == "train_seen"]
    unseen = data[data["zero_shot_split"] == "test_unseen"]
    normal_train = train[train["attack_label"] == "normal"]
    unseen_attacks = unseen[unseen["attack_label"] != "normal"]
    seen_attacks = set(train.loc[train["attack_label"] != "normal", "attack_label"])
    unseen_labels = set(unseen_attacks["attack_label"])

    if len(normal_train) < 10:
        raise DatasetValidationError(
            "At least 10 train_seen normal rows are required for training, "
            "calibration, and evaluation."
        )
    if unseen_attacks.empty:
        raise DatasetValidationError(
            "test_unseen must contain at least one non-normal attack row."
        )
    overlap = seen_attacks.intersection(unseen_labels)
    if overlap:
        raise DatasetValidationError(
            "Attack labels must not appear in both train_seen and test_unseen: "
            + ", ".join(sorted(map(str, overlap)))
        )


def _build_model(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "detector",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_zero_shot(
    data: pd.DataFrame,
    target_false_positive_rate: float = 0.05,
    random_state: int = 42,
) -> EvaluationResult:
    """Fit on known-normal training rows and evaluate held-out normals vs unseen attacks.

    The attack labels and split column are used only to define evaluation sets;
    neither is passed to the fitted model.
    """
    validate_dataset(data)
    if not 0.01 <= target_false_positive_rate <= 0.20:
        raise ValueError("target_false_positive_rate must be between 0.01 and 0.20.")

    train_seen = data[data["zero_shot_split"] == "train_seen"]
    normal = train_seen[train_seen["attack_label"] == "normal"]
    unseen_attacks = data[
        (data["zero_shot_split"] == "test_unseen")
        & (data["attack_label"] != "normal")
    ]

    training_normal, remaining_normal = train_test_split(
        normal, test_size=0.4, random_state=random_state
    )
    calibration_normal, evaluation_normal = train_test_split(
        remaining_normal, test_size=0.5, random_state=random_state
    )

    model = _build_model(random_state)
    model.fit(training_normal[FEATURES])

    calibration_scores = -model.score_samples(calibration_normal[FEATURES])
    threshold = float(
        np.quantile(
            calibration_scores,
            1.0 - target_false_positive_rate,
            method="higher",
        )
    )

    evaluation = pd.concat([evaluation_normal, unseen_attacks], ignore_index=True)
    y_true = np.concatenate(
        [
            np.zeros(len(evaluation_normal), dtype=int),
            np.ones(len(unseen_attacks), dtype=int),
        ]
    )
    anomaly_scores = -model.score_samples(evaluation[FEATURES])
    y_pred = (anomaly_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, anomaly_scores)),
        "average_precision": float(average_precision_score(y_true, anomaly_scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    predictions = evaluation[
        [column for column in ["timestamp", "department", "event_type"] if column in evaluation]
    ].copy()
    predictions["actual_label"] = evaluation["attack_label"].to_numpy()
    predictions["is_unseen_attack"] = y_true.astype(bool)
    predictions["anomaly_score"] = anomaly_scores
    predictions["detected"] = y_pred.astype(bool)

    attack_predictions = predictions[predictions["is_unseen_attack"]]
    detections_by_attack = (
        attack_predictions.groupby("actual_label", as_index=False)
        .agg(rows=("detected", "size"), detected=("detected", "sum"))
        .sort_values("actual_label")
    )
    detections_by_attack["detection_rate"] = (
        detections_by_attack["detected"] / detections_by_attack["rows"]
    )

    return EvaluationResult(
        metrics=metrics,
        confusion=np.array([[tn, fp], [fn, tp]], dtype=int),
        threshold=threshold,
        calibration_normal_count=len(calibration_normal),
        training_normal_count=len(training_normal),
        predictions=predictions,
        detections_by_attack=detections_by_attack,
    )


def load_csv(file: Any) -> pd.DataFrame:
    """Read CSV input and validate its schema before analysis."""
    try:
        data = pd.read_csv(file)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise DatasetValidationError(f"Could not read the CSV file: {exc}") from exc
    validate_dataset(data)
    return data

from __future__ import annotations

from html import escape
from io import BytesIO
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from detector import DatasetValidationError, evaluate_zero_shot, load_csv


st.set_page_config(
    page_title="Zero-Shot Enterprise Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_DATASET = (
    Path.home() / "Downloads" / "zero_shot_enterprise_user_behaviour_logs (1).csv"
)
configured_path = os.environ.get("ZERO_SHOT_DATASET")
default_path = Path(configured_path) if configured_path else DEFAULT_DATASET


@st.cache_data(show_spinner=False)
def load_local_dataset(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return load_csv(path)


with st.sidebar:
    st.header("Detection settings")
    false_positive_rate = st.slider(
        "Target calibration false-positive rate",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        format="%d%%",
        help=(
            "Sets the anomaly-score threshold from a calibration subset of "
            "known-normal training rows. The measured evaluation rate can differ."
        ),
    )

st.subheader("Upload and deploy a dataset")
st.caption(
    "Choose a compatible CSV and deploy it to run detection in this local app. "
    "This does not publish the app or dataset to the internet."
)
with st.form("dataset_deployment"):
    uploaded_file = st.file_uploader("Upload a compatible CSV", type=["csv"])
    deploy_clicked = st.form_submit_button(
        "Deploy dataset", type="primary", use_container_width=True
    )

if deploy_clicked and uploaded_file is not None:
    st.session_state["deployed_dataset_content"] = uploaded_file.getvalue()
    st.session_state["deployed_dataset_name"] = uploaded_file.name
    st.session_state["deployment_notice"] = uploaded_file.name

try:
    if "deployed_dataset_content" in st.session_state:
        dataset_content = st.session_state["deployed_dataset_content"]
        data = load_csv(BytesIO(dataset_content))
        dataset_name = st.session_state["deployed_dataset_name"]
    else:
        if not default_path.is_file():
            st.warning(
                "No dataset is deployed yet and the default dataset was not found. "
                "Upload a compatible CSV above and select Deploy dataset."
            )
            st.stop()
        data = load_local_dataset(
            str(default_path), default_path.stat().st_mtime
        )
        dataset_name = default_path.name
        dataset_content = default_path.read_bytes()
    result = evaluate_zero_shot(
        data, target_false_positive_rate=false_positive_rate / 100
    )
except (DatasetValidationError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

deployment_notice = st.session_state.pop("deployment_notice", None)
status_column, download_column = st.columns([4, 1], vertical_alignment="center")
with status_column:
    if deployment_notice is not None:
        st.success(f"Deployed {deployment_notice} for analysis.")
    else:
        st.caption(f"Active dataset: `{dataset_name}`")
with download_column:
    st.download_button(
        "Download active dataset",
        data=dataset_content,
        file_name=Path(dataset_name).name,
        mime="text/csv",
        use_container_width=True,
    )

unseen = data[data["zero_shot_split"] == "test_unseen"]
unseen_attack_count = int((unseen["attack_label"] != "normal").sum())
evaluation_normal_count = int((result.predictions["actual_label"] == "normal").sum())
alert_count = int(result.predictions["detected"].sum())
no_alert_count = len(result.predictions) - alert_count
summary_cards = [
    ("Total logs analyzed", f"{len(result.predictions):,}"),
    ("Suspicious activities", f"{alert_count:,}"),
    ("Held-out novel attacks", f"{unseen_attack_count:,}"),
    ("No alert raised", f"{no_alert_count:,}"),
]

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 1.5rem; }
    .dashboard-heading { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
    .dashboard-shield {
        display: grid; place-items: center; width: 54px; height: 54px;
        flex: 0 0 54px; border-radius: 12px; background: #14254b;
        color: #8dbdff; font-size: 27px;
    }
    .stMarkdown h1.dashboard-title {
        margin: 0; color: #f5f7fb; font-size: 25px; font-weight: 700; line-height: 1.2;
    }
    .dashboard-subtitle { margin: 3px 0 0; color: #a4a8b0; font-size: 15px; }
    .summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .summary-card { min-height: 108px; padding: 17px 16px; border-radius: 12px; background: #242424; }
    .summary-label { color: #aaa; font-size: 16px; }
    .summary-value { margin-top: 7px; color: #fff; font-size: 42px; font-weight: 650; line-height: 1; }
    .dashboard-note { margin: 14px 0 22px; color: #aaa; font-size: 14px; line-height: 1.5; }
    .detection-table { width: 100%; border-collapse: collapse; color: #eee; font-size: 15px; }
    .detection-table th { padding: 12px 10px 11px 0; border-bottom: 1px solid #333; text-align: left; }
    .detection-table td { padding: 12px 10px 12px 0; border-bottom: 1px solid #222; }
    .status-pill { display: inline-block; padding: 4px 11px; border-radius: 16px; font-weight: 600; }
    .status-alert { color: #ff4949; background: #351211; }
    .status-clear { color: #00c76a; background: #092819; }
    @media (max-width: 640px) {
        .stMarkdown h1.dashboard-title { font-size: 20px; }
        .summary-card { min-height: 96px; padding: 14px; }
        .summary-label { font-size: 14px; }
        .summary-value { font-size: 36px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="dashboard-heading">
      <div class="dashboard-shield" aria-hidden="true">🛡️</div>
      <div>
        <h1 class="dashboard-title">Zero-Shot Cyber-Attack Detection</h1>
        <p class="dashboard-subtitle">Enterprise activity monitoring · model evaluation dashboard</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
cards_html = "".join(
    f'<div class="summary-card"><div class="summary-label">{escape(label)}</div>'
    f'<div class="summary-value">{escape(value)}</div></div>'
    for label, value in summary_cards
)
st.markdown(f'<div class="summary-grid">{cards_html}</div>', unsafe_allow_html=True)
st.markdown(
    f"The held-out evaluation contains {unseen_attack_count:,} attack logs and "
    f"{evaluation_normal_count:,} normal logs. The detector flagged "
    f"{result.metrics['true_positives']:,} attacks and "
    f"{result.metrics['false_positives']:,} normal logs "
    f"({result.metrics['false_positive_rate']:.1%} false-positive rate). "
    "Held-out novel attacks are dataset labels, not model predictions.",
    help=(
        "The model is trained only on known-normal behavior and flags unusual "
        "activity. Ground-truth attack labels are used only to evaluate the "
        "held-out results."
    ),
)
st.markdown(
    '<div class="dashboard-note">The table shows anonymized examples of both alerts '
    'and non-alerts. A non-alert means the detector did not flag the event; it does '
    'not guarantee that the event is benign.</div>',
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Sample detection table")
alerts = result.predictions[result.predictions["detected"]].nlargest(
    5, "anomaly_score"
)
no_alerts = result.predictions[~result.predictions["detected"]].nsmallest(
    5, "anomaly_score"
)
sample_rows = []
for index in range(max(len(alerts), len(no_alerts))):
    for group in (alerts, no_alerts):
        if index < len(group):
            sample_rows.append(group.iloc[[index]])
examples = pd.concat(sample_rows)
table_rows = []
for index, (_, prediction) in enumerate(examples.iterrows(), start=1):
    detected = bool(prediction["detected"])
    category = "Potential anomaly" if detected else "No anomaly detected"
    status = "Suspicious" if detected else "No alert"
    status_class = "status-alert" if detected else "status-clear"
    table_rows.append(
        "<tr>"
        f"<td>Log {index:03d}</td>"
        f"<td>{category}</td>"
        f'<td><span class="status-pill {status_class}">{status}</span></td>'
        "</tr>"
    )
st.markdown(
    '<table class="detection-table"><thead><tr><th>Event</th>'
    '<th>Predicted category</th><th>Detection status</th></tr></thead><tbody>'
    + "".join(table_rows)
    + "</tbody></table>",
    unsafe_allow_html=True,
)

with st.expander("Model evaluation details", expanded=False):
    st.info(
        "This implements zero-shot novelty detection, not semantic zero-shot "
        "classification. Attack labels are excluded from model training and are "
        "used only to score the held-out evaluation."
    )
    st.subheader("Held-out evaluation")
    metric_columns = st.columns(5)
    metric_columns[0].metric("ROC AUC", f"{result.metrics['roc_auc']:.3f}")
    metric_columns[1].metric("Average precision", f"{result.metrics['average_precision']:.3f}")
    metric_columns[2].metric("Recall", f"{result.metrics['recall']:.1%}")
    metric_columns[3].metric("Precision", f"{result.metrics['precision']:.1%}")
    metric_columns[4].metric("F1 score", f"{result.metrics['f1']:.3f}")
    st.caption(
        f"Threshold: {result.threshold:.4f} anomaly score · "
        f"measured false-positive rate on separate held-out normal rows: "
        f"{result.metrics['false_positive_rate']:.1%} · "
        f"confusion matrix (rows actual normal/attack, columns predicted normal/attack): "
        f"{result.confusion.tolist()}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Detection by unseen attack class")
        per_attack = result.detections_by_attack.copy()
        per_attack["detection_rate"] = per_attack["detection_rate"].map(
            lambda value: f"{value:.1%}"
        )
        per_attack = per_attack.rename(
            columns={
                "actual_label": "Unseen attack",
                "rows": "Rows",
                "detected": "Detected",
                "detection_rate": "Detection rate",
            }
        )
        st.dataframe(per_attack, width="stretch", hide_index=True)
    with right:
        st.subheader("Rows by split and label")
        label_counts = (
            data.groupby(["zero_shot_split", "attack_label"], as_index=False)
            .size()
            .rename(columns={"size": "rows"})
        )
        fig = px.bar(
            label_counts,
            x="attack_label",
            y="rows",
            color="zero_shot_split",
            barmode="group",
            labels={"attack_label": "Label", "rows": "Rows", "zero_shot_split": "Split"},
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Unseen attack anomaly scores")
    score_chart = px.histogram(
        result.predictions,
        x="anomaly_score",
        color="actual_label",
        nbins=35,
        barmode="overlay",
        opacity=0.72,
        labels={"anomaly_score": "Anomaly score (higher = more unusual)"},
    )
    score_chart.add_vline(
        x=result.threshold, line_dash="dash", line_color="red", annotation_text="Threshold"
    )
    st.plotly_chart(score_chart, width="stretch")
st.caption(
    "User identifiers are excluded from the model and are not displayed. "
    "This demonstration is not a substitute for production security monitoring."
)

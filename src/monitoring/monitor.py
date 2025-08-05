import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from datetime import datetime

# Paths
DATA_PATH = "data/diabetes.csv"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Load and split data
df = pd.read_csv(DATA_PATH)
ref_df = df.iloc[:500]  # Reference dataset
curr_df = df.iloc[500:]  # Current dataset

# Generate report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=curr_df)

# Save to HTML
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"{REPORT_DIR}/data_drift_report_{timestamp}.html"
report.save_html(report_path)

# Log to MLflow
with mlflow.start_run(run_name="evidently_data_drift_report") as run:
    mlflow.log_artifact(report_path, artifact_path="evidently_reports")
    print(f"✅ Drift report logged to MLflow: {report_path}")


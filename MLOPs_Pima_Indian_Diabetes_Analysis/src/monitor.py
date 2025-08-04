# src/pipeline/monitor.py

import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import os


def generate_drift_report(reference_path="data/diabetes.csv", production_path="data/production.csv"):
    # Load datasets
    reference_data = pd.read_csv(reference_path)
    production_data = pd.read_csv(production_path)

    # Generate drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=production_data)

    # Save report
    os.makedirs("output", exist_ok=True)
    report_path = "output/drift_report.html"
    report.save_html(report_path)

    print(f"Drift report saved to {report_path}")


if __name__ == "__main__":
    generate_drift_report()


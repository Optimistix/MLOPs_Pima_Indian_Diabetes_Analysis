from flask import Flask, jsonify, request
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)
client = MlflowClient()

@app.route("/metrics")
def get_metrics():
    experiment_name = request.args.get("experiment", "pima_best_model_training")
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        return jsonify({"error": "Experiment not found"}), 404

    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"], max_results=1)
    if not runs:
        return jsonify({"error": "No runs found"}), 404

    run = runs[0]
    metrics = {
        "accuracy": run.data.metrics.get("accuracy"),
        "roc_auc": run.data.metrics.get("roc_auc"),
        "precision": run.data.metrics.get("precision"),
        "recall": run.data.metrics.get("recall"),
        "run_id": run.info.run_id,
        "start_time": run.info.start_time,
        # Add more metrics as needed (e.g., drift)
        # "drift_score": run.data.metrics.get("drift_score"),  # if logged
    }
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


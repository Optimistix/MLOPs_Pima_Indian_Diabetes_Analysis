## Project Overview: ML Analysis of the Pima Indian Diabetes Dataset

This project focuses on applying machine learning techniques to the Pima Indian Diabetes dataset, a well-known dataset from the UCI repository. The dataset contains medical diagnostic measurements of female patients of Pima Indian heritage, aged 21 and older, with the goal of predicting the onset of diabetes.

📊 Dataset Highlights
Target variable: Outcome (1 = diabetes, 0 = no diabetes)

Features include:

Number of pregnancies

Glucose concentration

Blood pressure

Skin thickness

Insulin level

BMI

Diabetes pedigree function

Age

🎯 Objective
Use supervised machine learning to:

Train multiple classifiers (e.g., Logistic Regression, Random Forest, SVM, MLP, LightGBM)

Compare model performance (via AUC and other metrics)

Track experiments using MLflow

Visualize key metrics with Grafana

Automate the workflow with Prefect orchestration

This problem is a classic binary classification task in healthcare, with real-world implications for early detection of diabetes based on routine medical data.

Ethical Considerations
While the Pima Indian Diabetes dataset provides a useful benchmark for evaluating machine learning models in healthcare, several important considerations apply when translating such analysis to real-world clinical settings:

✅ Fairness & Bias
The dataset is ethnically homogeneous (Pima Indian women), so model performance may not generalize across other populations.

Applying models trained on this data to broader patient groups risks introducing bias or unequal predictive accuracy.

🧪 Clinical Reliability
High performance (e.g., AUC > 0.95) may indicate overfitting rather than real clinical utility.

Features like insulin level and BMI are not always measured consistently, especially in underserved settings.

🔁 Interpretability
In a clinical context, interpretability and trust in predictions is often as important as raw accuracy.

While complex models like LightGBM or MLP may perform well, simpler models (e.g., logistic regression) may be preferable for decision support.

🧷 Data Privacy
This dataset is anonymized and public, but real patient data requires strict privacy and compliance with regulations like HIPAA or GDPR.


# MLOps Diabetes Pipeline

This repository contains a complete MLOps pipeline for the Pima Indian Diabetes dataset, featuring:

- Training multiple models (LightGBM, Logistic Regression, Random Forest, SVM, MLP, etc.)
- Logging experiments and metrics to MLflow (with SQLite backend)
- Monitoring metrics via Grafana connected to MLflow's SQLite DB
- Prefect orchestration for workflow automation, scheduling, and monitoring

---

## Prerequisites

- Docker & Docker Compose installed
- Basic familiarity with Docker and MLflow

---

## Setup

### 1. Clone repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Docker images and dependencies

- The Dockerfile installs required Python packages including Prefect and MLflow.
- Prefect, MLflow, and Grafana are run as separate services in Docker Compose.

### 3. Docker Compose services

- **mlflow**: MLflow server UI, tracking SQLite database
- **grafana**: Grafana UI for visualizing MLflow metrics
- **metrics-api**: Custom metrics API container
- **prefect-server**: Prefect Orion server & UI for orchestration
- **prefect-agent**: Runs Prefect flows (e.g., training pipeline)

---

## Running the Pipeline

### Build Docker images

```bash
docker-compose build mlops-diabetes
```

### Start all services

```bash
docker-compose up -d prefect-server mlflow grafana prefect-agent metrics-api
```

### Access UIs

- MLflow UI: [http://localhost:5000](http://localhost:5000)
- Grafana UI: [http://localhost:3000](http://localhost:3000)
- Prefect UI: [http://localhost:4200](http://localhost:4200)

---

## Prefect Flow

- The Prefect flow is defined in `flow.py`, which runs `train_and_log.py` inside a Prefect task.
- The `prefect-agent` service executes this flow inside Docker.
- Use the Prefect UI to monitor flow runs, logs, and schedule future runs.

---

## Dashboard Setup

- Grafana dashboards are auto-provisioned from `./docker/grafana/dashboards/`
- Provisioning configs are in `./docker/grafana/provisioning/dashboards/`
- Dashboards visualize MLflow metrics stored in SQLite.

---

## Logs & Debugging

- To view logs for services:

```bash
docker-compose logs -f mlflow
docker-compose logs -f grafana
docker-compose logs -f prefect-server
docker-compose logs -f prefect-agent
```

- For manual dashboard import or troubleshooting, use the Grafana UI.

---

## Extending the Pipeline

- Add additional Prefect tasks for model selection, tuning, alerting, etc.
- Integrate notifications for failed runs via Prefect.
- Expand Grafana dashboards with more MLflow metrics and parameters.
- Add CI/CD and infrastructure-as-code as next steps.

---

## Troubleshooting

- If dashboards don't load, verify JSON dashboard files and provisioning YAML paths.
- Confirm Prefect and MLflow services are running and accessible.
- Rebuild Docker images after dependency or code changes.
- Check container logs for errors.

---

## License

[MIT License](LICENSE)


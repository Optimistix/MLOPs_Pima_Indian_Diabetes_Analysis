# 🩺 Pima Diabetes MLOps Pipeline

This repository implements a complete MLOps pipeline for analyzing the Pima Indian Diabetes dataset, using modern tools for experimentation, monitoring, orchestration, and visualization.

---

## 📊 Problem Statement

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


---

## 🚀 Features

- **Data Versioning & Tracking**: MLflow to track experiments, metrics, models, and artifacts.
- **Model Training**: Multiple models (Logistic Regression, Random Forest, SVM, MLP, XGBoost, LightGBM, CatBoost).
- **Best Model Selection**: Automatically selects and registers best-performing model.
- **Model Monitoring**: Evidently integrated for drift and performance monitoring.
- **Visualization**: Grafana dashboard linked to SQLite MLflow backend.
- **Orchestration**: Prefect (local and cloud) for running and scheduling flows.
- **Containers**: Dockerized environment with `docker-compose` support.
- **Extensible**: CI/CD, pre-commit hooks, unit/integration tests, and IaC to be added.

---

## 🧱 Project Structure

```
.
├── docker/
│   └── grafana/
│       ├── dashboards/
│       │   └── mlflow_dashboard.json
│       └── provisioning/
│           └── dashboards/mlflow.yaml
├── src/
│   ├── bridge/
│   │   └── metrics_api.py
│   ├── monitoring/
│   │   └── monitor.py
│   ├── training/
│   │   └── train_and_log.py
│   └── orchestration/
│       └── evidently_flow.py
├── mlflow/           # SQLite DB file mount
├── mlruns/           # MLflow artifacts
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🐳 Usage

### 1. Build and Run

```bash
docker-compose up -d --build
```

### 2. Train Models

```bash
docker-compose exec metrics-api python src/training/train_and_log.py
```

### 3. Monitor Metrics API

```bash
curl http://localhost:8000/metrics
```

### 4. Generate Monitoring Report

```bash
docker-compose exec metrics-api python src/monitoring/monitor.py
```

### 5. Run Prefect Flow (Locally)

```bash
docker-compose exec metrics-api python src/orchestration/evidently_flow.py
```

---

## ☁️ Prefect Cloud Setup

1. Log in:
   ```bash
   prefect cloud login
   ```

2. Set workspace:
   ```bash
   prefect cloud workspace set --account <ACCOUNT_ID> --workspace <WORKSPACE_ID>
   ```

3. Deploy:
   ```bash
   prefect deployment build src/orchestration/evidently_flow.py:monitoring_flow --name "Evidently Monitoring" --work-queue default
   prefect deployment apply monitoring_flow-deployment.yaml
   ```

4. Start agent:
   ```bash
   prefect agent start --pool default-agent-pool --work-queue default
   ```

---

## 📈 Grafana Setup

1. Visit [http://localhost:3000](http://localhost:3000) (default credentials: admin / admin).
2. Add `frser-sqlite-datasource` plugin.
3. Import the dashboard: `mlflow_dashboard.json`.

---

## 📦 Planned Enhancements

- [ ] CI/CD with GitHub Actions
- [ ] Unit & integration tests
- [ ] Pre-commit hooks for formatting and linting
- [ ] Infrastructure as Code (Terraform / Pulumi)
- [ ] Deployment to AWS/GCP via LocalStack or cloud-native services

---

## 📜 License

MIT License. See `LICENSE` file.

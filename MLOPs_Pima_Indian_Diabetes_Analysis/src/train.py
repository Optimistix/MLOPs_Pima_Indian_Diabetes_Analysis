# src/pipeline/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os

def train_and_log_model(csv_path="data/diabetes.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
    ])

    mlflow.set_experiment("Final_Model_Training")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model", "XGBoost with scaling")
        mlflow.log_metric("test_accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")

        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Save predictions locally
        os.makedirs("output", exist_ok=True)
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("output/predictions.csv", index=False)
        mlflow.log_artifact("output/predictions.csv")


if __name__ == "__main__":
    train_and_log_model()


import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODEL_MAP = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "MLP": MLPClassifier,
    "SVM": SVC,
    "XGBoost": XGBClassifier,
    "CatBoost": CatBoostClassifier,
    "LightGBM": LGBMClassifier
}

def load_data(path="data/pima.csv"):
    cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(path, header=None, names=cols)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, pd.NA)
    return df.fillna(df.median(numeric_only=True))

def build_pipeline(model_name):
    model_cls = MODEL_MAP[model_name]
    needs_scaling = model_name in ["Logistic Regression", "MLP", "SVM"]
    model = model_cls()

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))

    return Pipeline(steps)

if __name__ == "__main__":
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    with open("results/best_model.json") as f:
        best_model_info = json.load(f)
    model_name = best_model_info["Model"]

    mlflow.set_experiment("pima_best_model_training")
    with mlflow.start_run(run_name=f"train_{model_name}"):

        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probas = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['clf'], "predict_proba") else preds

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipeline, "model")
        print(f"✅ Trained and logged model: {model_name}")


# src/pipeline/model_selection.py

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
import mlflow
import mlflow.sklearn


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "MLP": MLPClassifier(max_iter=1000),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
    }


def evaluate_models(X, y):
    mlflow.set_experiment("Model_Comparison")
    
    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=make_scorer(accuracy_score))
            mean_score = scores.mean()
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_accuracy", mean_score)
            print(f"{name}: {mean_score:.4f}")


if __name__ == "__main__":
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    evaluate_models(X, y)


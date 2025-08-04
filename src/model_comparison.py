import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_data(path="data/pima.csv"):
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(path, header=None, names=columns)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    return df

def evaluate_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "MLP": MLPClassifier(max_iter=1000),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(verbose=0),
        "LightGBM": LGBMClassifier()
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_experiment("pima_model_comparison")

    for name, model in models.items():
        print(f"Evaluating: {name}")
        with mlflow.start_run(run_name=name):

            if name in ["Logistic Regression", "Random Forest", "MLP", "SVM", "XGBoost", "LightGBM"]:
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler() if name in ["Logistic Regression", "MLP", "SVM"] else "passthrough")
                ])
                X_proc = pd.DataFrame(numeric_pipeline.fit_transform(X), columns=X.columns)
            else:
                X_proc = X.copy()

            acc_scores = cross_val_score(model, X_proc, y, cv=cv, scoring="accuracy")
            auc_scores = cross_val_score(model, X_proc, y, cv=cv, scoring="roc_auc")

            acc_mean, acc_std = acc_scores.mean(), acc_scores.std()
            auc_mean, auc_std = auc_scores.mean(), auc_scores.std()

            mlflow.log_metric("accuracy_mean", acc_mean)
            mlflow.log_metric("accuracy_std", acc_std)
            mlflow.log_metric("roc_auc_mean", auc_mean)
            mlflow.log_metric("roc_auc_std", auc_std)

            mlflow.sklearn.log_model(model, "model")

            results.append({
                "Model": name,
                "Accuracy Mean": acc_mean,
                "Accuracy Std": acc_std,
                "ROC AUC Mean": auc_mean,
                "ROC AUC Std": auc_std
            })

    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.sort_values("ROC AUC Mean", ascending=False),
                x="ROC AUC Mean", y="Model", palette="viridis")
    plt.title("Model ROC AUC Comparison")
    plt.tight_layout()
    plt.savefig("figures/model_auc_comparison.png")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    results_df = evaluate_models(X, y)

    # Save results
    results_df.to_csv("results/model_comparison.csv", index=False)

    # Plot
    plot_results(results_df)

    # Report best model
    best_model = results_df.sort_values("ROC AUC Mean", ascending=False).iloc[0]
    print(f"\n✅ Best Model: {best_model['Model']}")
    print(best_model)


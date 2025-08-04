import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from mlflow.models.signature import infer_signature

# Load dataset
df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

best_model_name = None
best_auc = 0
best_model = None

with mlflow.start_run(run_name="Pima_Diabetes_Model_Comparison") as run:
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        mlflow.log_param(f"{name}_model_type", name)
        mlflow.log_metric(f"{name}_auc", auc)

        input_example = X_val.iloc[:2]
        signature = infer_signature(X_val, model.predict(X_val))
        mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            input_example=input_example,
            signature=signature
        )

        print(f"{name} AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model

    print(f"\nBest model: {best_model_name} with AUC {best_auc:.4f}")
    mlflow.set_tag("best_model", best_model_name)

    # Register the best model
    model_uri = f"runs:/{run.info.run_id}/{best_model_name}"
    model_name = "diabetes_best_model"

    # Create or update the registered model
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # (Optional) Transition to Staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True
    )  

print(f"\n✅ Registered '{best_model_name}' as '{model_name}' version {result.version} in stage 'Staging'")



import pytest
from src.training.train_and_log import train_model, load_data
import pandas as pd

def test_load_data_returns_dataframe():
    df = load_data("data/diabetes.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_train_model_returns_trained_model():
    df = load_data("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    model = train_model(X, y, model_type="logistic_regression")
    assert hasattr(model, "predict")


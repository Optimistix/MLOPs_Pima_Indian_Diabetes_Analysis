# src/pipeline/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def run_eda(csv_path="data/diabetes.csv"):
    df = pd.read_csv(csv_path)
    print("\n--- Dataset Head ---\n", df.head())
    print("\n--- Missing Values ---\n", df.isnull().sum())
    print("\n--- Descriptive Stats ---\n", df.describe())

    plt.figure(figsize=(10, 6))
    sns.countplot(x="Outcome", data=df)
    plt.title("Class Distribution")
    plt.savefig("output/class_distribution.png")

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("output/correlation_heatmap.png")


if __name__ == "__main__":
    run_eda()


from prefect import flow, task
from src.train import load_data, objective
from hyperopt import fmin, tpe, hp, Trials

@task
def load():
    return load_data("data/pima.csv")

@task
def preprocess(df):
    from sklearn.model_selection import train_test_split
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

@flow(name="pima_training_pipeline")
def main_pipeline():
    df = load()
    X_train, X_test, y_train, y_test = preprocess(df)
    
    from src import train
    train.X_train, train.X_test = X_train, X_test
    train.y_train, train.y_test = y_train, y_test

    space = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
    }

    trials = Trials()
    fmin(fn=train.objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

if __name__ == "__main__":
    main_pipeline()


from prefect import flow, task
import subprocess

@task
def train_and_log():
    # Run your existing train_and_log.py script as a subprocess
    result = subprocess.run(["python", "train_and_log.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")

@flow(name="MLflow Training Pipeline")
def mlflow_training_flow():
    train_and_log()

if __name__ == "__main__":
    mlflow_training_flow()


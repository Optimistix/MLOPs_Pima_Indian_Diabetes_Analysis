from prefect import flow, task
import subprocess

@task
def run_drift_report():
    result = subprocess.run(
        ["python", "src/monitoring/monitor.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Drift report failed: {result.stderr}")

@flow(name="Evidently Drift Monitoring")
def monitoring_flow():
    run_drift_report()

if __name__ == "__main__":
    monitoring_flow()


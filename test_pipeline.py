import pytest
import subprocess

def test_full_training_pipeline():
    # Run the training script as a subprocess
    result = subprocess.run(
        ["python", "src/training/train_and_log.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Best model" in result.stdout  # Adjust based on your print/logs


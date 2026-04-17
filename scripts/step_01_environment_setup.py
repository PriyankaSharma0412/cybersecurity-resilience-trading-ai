"""
Step 1: 1. Environment Setup and Reproducible Output Paths

Run: python scripts/step_01_environment_setup.py
"""

# --- Code Cell 1 ---
from pathlib import Path

IN_COLAB = False
print("Using local project storage; Google Drive is disabled for this notebook.")


# --- Code Cell 2 ---
import os
from pathlib import Path

project_root = Path.cwd()
base_path = project_root / "dissertation_outputs"

mpl_config_dir = project_root / ".mplconfig"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

folders = ["data", "models", "figures", "results"]
for folder in folders:
    (base_path / folder).mkdir(parents=True, exist_ok=True)

print(f"Base path: {base_path}")
print(f"MPLCONFIGDIR: {mpl_config_dir}")


print("\nStep 1 completed successfully!")

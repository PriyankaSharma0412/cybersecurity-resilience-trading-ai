"""
Run the consolidated supervised financial robustness pipeline.
"""

import subprocess
import sys
import time


PIPELINE_SCRIPT = "scripts/supervised_pipeline.py"


if __name__ == "__main__":
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {PIPELINE_SCRIPT}")
    print(f"{'=' * 70}")
    start = time.time()
    result = subprocess.run([sys.executable, PIPELINE_SCRIPT], capture_output=False, text=True)
    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    if result.returncode == 0:
        print(f"  PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")
    else:
        print(f"  PIPELINE FAILED WITH EXIT CODE {result.returncode} AFTER {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"{'=' * 70}")

"""
Run all 20 steps sequentially.
"""
import subprocess
import sys
import time

scripts = [
    "scripts/step_01_environment_setup.py",
    "scripts/step_02_install_import_libraries.py",
    "scripts/step_03_data_collection.py",
    "scripts/step_04_exploratory_data_analysis.py",
    "scripts/step_05_feature_engineering.py",
    "scripts/step_06_prepare_model_features.py",
    "scripts/supervised_step_07_target_split.py",
    "scripts/supervised_step_08_logistic.py",
    "scripts/supervised_step_09_random_forest.py",
    "scripts/supervised_step_10_xgboost.py",
    "scripts/supervised_step_11_evaluation.py",
    "scripts/supervised_step_12_robustness.py",
    "scripts/supervised_step_13_stress.py",
    "scripts/supervised_step_14_shap.py",
    "scripts/supervised_step_15_shap_stability.py",
    "scripts/supervised_step_16_summary.py",
    "scripts/supervised_step_17_narrative.py",
    "scripts/supervised_step_18_validation.py",
    "scripts/supervised_step_19_visuals.py",
    "scripts/supervised_step_20_chapter4_story.py",
]

failed = []
for i, script in enumerate(scripts, 1):
    print(f"\n{'='*70}")
    print(f"  RUNNING STEP {i}/{len(scripts)}: {script}")
    print(f"{'='*70}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  *** STEP {i} FAILED (exit code {result.returncode}) after {elapsed:.1f}s ***")
        failed.append((i, script))
        break  # Stop on first failure since steps depend on each other
    else:
        print(f"\n  Step {i} completed in {elapsed:.1f}s")

print(f"\n{'='*70}")
if failed:
    print(f"  PIPELINE STOPPED AT STEP {failed[0][0]}: {failed[0][1]}")
else:
    print(f"  ALL {len(scripts)} STEPS COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")

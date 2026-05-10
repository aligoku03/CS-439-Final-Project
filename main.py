# =============================================================
# main.py
# CS-439 Final Project — Rutgers University
# =============================================================
# This is the entry point for the entire project pipeline.
# It runs all 8 scripts in the correct order, from raw PDB data
# collection through to molecular visualization. Each step checks
# if its output already exists and skips it if so, meaning you
# can safely re-run main.py without redoing hours of computation.
# Simply delete a step's output file to force it to re-run.
# =============================================================
# Libraries:
#   os         : checking file and folder paths
#   sys        : accessing the current Python interpreter path
#   subprocess : running each script as a separate process
#   time       : measuring how long each step takes
# =============================================================

import os
import sys
import subprocess
import time

# setting up the base path
BASE_PATH = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'

# the correct order to run all scripts
# each step builds on the previous one
PIPELINE = [
    {
        'step'  : 1,
        'name'  : 'PDB Data Collection — EGFR, BACE1, COX2',
        'file'  : 'data_filtering.py',
        'check' : os.path.join('preprocessed_results', 'EGFR_pdb_features.csv'),
        'note'  : 'Downloads and parses mmCIF files — takes 10-30 minutes',
    },
    {
        'step'  : 2,
        'name'  : 'PDB Data Collection — HIV Protease, Thrombin',
        'file'  : 'data_api_fetch.py',
        'check' : os.path.join('preprocessed_results', 'HIV_Protease_pdb_features.csv'),
        'note'  : 'Fetches structures via RCSB API — takes 5-15 minutes',
    },
    {
        'step'  : 3,
        'name'  : 'Data Standardization',
        'file'  : 'data_standardize.py',
        'check' : os.path.join('preprocessed_results', 'Thrombin_pdb_features.csv'),
        'note'  : 'Standardizes all 5 CSVs to 44 features',
    },
    {
        'step'  : 4,
        'name'  : 'Exploratory Data Analysis',
        'file'  : 'eda.py',
        'check' : os.path.join('results', 'EDA', '01_dataset_overview.png'),
        'note'  : 'Generates 6 EDA plots across all 5 proteins',
    },
    {
        'step'  : 5,
        'name'  : 'Preprocessing — Drug Datasets + Feature Engineering',
        'file'  : 'preprocessing.py',
        'check' : os.path.join('processed_data', 'EGFR_train.csv'),
        'note'  : 'Downloads DAVIS/BACE/HIV datasets and creates Morgan fingerprints',
    },
    {
        'step'  : 6,
        'name'  : 'Baseline Models — Logistic Regression, Random Forest, XGBoost',
        'file'  : 'baseline_model.py',
        'check' : os.path.join('results', 'baseline_results.csv'),
        'note'  : 'Trains and evaluates 3 classical ML models per protein',
    },
    {
        'step'  : 7,
        'name'  : 'Graph Neural Network',
        'file'  : 'GGN.py',
        'check' : os.path.join('results', 'gnn_results.csv'),
        'note'  : 'Trains GATConv GNN for each protein — GPU recommended',
    },
    {
        'step'  : 8,
        'name'  : 'Molecular Visualization',
        'file'  : 'visualize_molecules.py',
        'check' : os.path.join('results', 'Visualization', 'EGFR_molecules.png'),
        'note'  : 'Draws 2D drug structures and molecular property plots',
    },
]


def print_header():
    print("=" * 65)
    print("  Drug-Protein Binding Prediction — CS-439 Final Project")
    print("  Rutgers University")
    print("  Team: Ali Ugur, Dipen Patel, Sarthak Gandotra")
    print("=" * 65)
    print()


def check_file_exists(base_path, check_path):
    # checking if the output file for this step already exists
    full_path = os.path.join(base_path, check_path)
    return os.path.exists(full_path)


def run_script(base_path, script_file):
    # running the script using the same python interpreter
    script_path = os.path.join(base_path, script_file)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
        cwd=base_path
    )
    return result.returncode == 0


def main():
    print_header()

    # checking that we are in the right directory
    if not os.path.exists(BASE_PATH):
        print(f"ERROR: Project folder not found at {BASE_PATH}")
        print("Please update BASE_PATH in main.py to match your setup")
        sys.exit(1)

    print(f"Project folder: {BASE_PATH}")
    print(f"Python: {sys.executable}")
    print()

    # tracking which steps were run vs skipped
    completed = []
    skipped   = []
    failed    = []

    for step in PIPELINE:
        print(f"{'='*65}")
        print(f"STEP {step['step']}/8 — {step['name']}")
        print(f"{'='*65}")
        print(f"  Script : {step['file']}")
        print(f"  Note   : {step['note']}")

        # checking if output already exists to avoid re-running
        script_path = os.path.join(BASE_PATH, step['file'])
        if not os.path.exists(script_path):
            print(f"  WARNING: {step['file']} not found — skipping")
            skipped.append(step['step'])
            print()
            continue

        if check_file_exists(BASE_PATH, step['check']):
            print(f"  Output already exists — skipping")
            print(f"  (delete {step['check']} to force re-run)")
            skipped.append(step['step'])
            print()
            continue

        # running the script
        print(f"\n  Running {step['file']}...")
        print("-" * 65)
        start_time = time.time()

        success = run_script(BASE_PATH, step['file'])

        elapsed = time.time() - start_time
        print("-" * 65)

        if success:
            print(f"  Completed in {elapsed:.1f}s")
            completed.append(step['step'])
        else:
            print(f"  FAILED after {elapsed:.1f}s")
            print(f"  Check the output above for error details")
            failed.append(step['step'])

        print()

    # printing the final summary
    print("=" * 65)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"\n  Steps completed : {completed}")
    print(f"  Steps skipped   : {skipped} (output already existed)")
    print(f"  Steps failed    : {failed}")

    print(f"\n  Results saved to:")
    print(f"    {os.path.join(BASE_PATH, 'results', 'baseline_results.csv')}")
    print(f"    {os.path.join(BASE_PATH, 'results', 'gnn_results.csv')}")
    print(f"    {os.path.join(BASE_PATH, 'results', 'figures')}")

    if failed:
        print(f"\n  WARNING: {len(failed)} step(s) failed.")
        print(f"  Run the individual scripts to see detailed errors.")
        sys.exit(1)
    else:
        print(f"\n  All steps completed successfully.")


if __name__ == '__main__':
    main()
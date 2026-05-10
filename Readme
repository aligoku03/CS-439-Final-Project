# Drug-Protein Binding Prediction with Graph Neural Networks

CS-439 Final Project — Rutgers University

## Motivation

Recent breakthroughs in computational chemistry have shown the power of machine learning in drug discovery. In September 2025, researchers at Zhejiang Province developed Bone-02, a bio-adhesive inspired by how oysters cling to surfaces — discovered after testing over 50 formulations and hundreds of animal trials. In March 2025, Japan Tobacco and D-Wave validated a quantum-AI workflow that generated more drug-like molecular structures than classical methods alone. Both stories share the same core challenge: finding the right molecule for the right biological target. This project applies machine learning to that exact problem — predicting whether a small-molecule drug will bind to a disease-relevant protein.

## What This Project Does

We compare classical machine learning baselines (Logistic Regression, Random Forest, XGBoost) against a Graph Attention Network (GATConv) that operates directly on molecular structure. Structural features from 1,592 PDB crystal structures are combined with Morgan molecular fingerprints and raw atom-bond graph representations across three disease targets.

## Proteins and Diseases

| Protein | Disease | Drug Dataset |
|---|---|---|
| EGFR | Cancer | DAVIS kinase affinity dataset |
| BACE1 | Alzheimer's | MoleculeNet BACE |
| HIV Protease | HIV/AIDS | MoleculeNet HIV |

COX2 (inflammation) and Thrombin (blood clotting) are included in the structural analysis but not in binding prediction since matched drug datasets were not available.

## Pipeline

The entire project runs through a single entry point:

```bash
python main.py
```

`main.py` runs all 8 steps in order and skips any step whose output already exists on disk. To force a step to re-run, delete its output file.

| Step | Script | What it does |
|---|---|---|
| 1 | `data_filtering.py` | Parses mmCIF files for EGFR, BACE1, COX2 |
| 2 | `data_api_fetch.py` | Fetches HIV Protease and Thrombin via RCSB API |
| 3 | `data_standardize.py` | Standardizes all 5 proteins to 44 features |
| 4 | `eda.py` | Generates 6 exploratory figures |
| 5 | `preprocessing.py` | Downloads drug datasets, computes Morgan fingerprints, 80/20 split |
| 6 | `baseline_model.py` | Trains LR, RF, XGBoost per protein |
| 7 | `GGN.py` | Trains Graph Attention Network per protein |
| 8 | `visualize_molecules.py` | Draws 2D drug structures and property plots |

Steps 1 and 2 download PDB structures and take 10-30 minutes on first run. All subsequent runs are instant since outputs are cached.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aligoku03/CS-439-Final-Project.git
cd CS-439-Final-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

**GPU (NVIDIA CUDA 12.8 — recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install PyTorch Geometric

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

### 5. Run

```bash
python main.py
```

> Note: This project requires Python 3.13. Update BASE_PATH at the top of each script to match your local folder path.

## Features

**Protein structural features (36 per protein):**
Quality metrics (resolution, R-work, R-free, B-factor, Rmerge), atom counts, crystal cell dimensions and angles, solvent content, Matthews coefficient, 20 amino acid frequency columns.

Only structures with resolution at or below 2.5 Angstroms are used, following the Scientific Reports 2021 cutoff for drug discovery.

**Drug features:**
Morgan circular fingerprints (radius 2, 2048 bits) computed via RDKit for baseline models. The GNN uses raw atom and bond graph features directly.

**Combined feature vector per example:** 2,084 dimensions (2,048 Morgan bits + 36 PDB features).

## Models

### Baselines

Three classical models trained per protein with class weighting for imbalance. Optimal decision thresholds are selected from ROC curves instead of using the default 0.5. HIV Protease additionally uses undersampling (5:1 ratio) due to 3.5% positive rate.

### Graph Attention Network

A custom DrugProteinGNN combining drug structure with protein features:

- 3 GATConv layers (16-dim atom features to 128-dim hidden) with BatchNorm and dropout 0.5
- Edge features (6-dim bond features) passed to attention layers
- Global mean pooling to produce one vector per molecule
- Concatenation with 36-dim PDB protein feature vector
- 3 fully connected layers (128 -> 64 -> 1)

Drug-level splitting is used to prevent data leakage — the same drug molecule never appears in both train and test sets.

## Results

### Baseline ROC-AUC (test set)

| Protein | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| EGFR (Cancer) | 0.776 | 0.777 | **0.778** |
| BACE1 (Alzheimer's) | 0.860 | 0.840 | **0.872** |
| HIV Protease (HIV/AIDS) | 0.768 | 0.789 | **0.809** |

### GNN ROC-AUC (test set)

| Protein | GNN | vs Best Baseline |
|---|---|---|
| EGFR (Cancer) | **0.844** | +0.066 |
| BACE1 (Alzheimer's) | 0.675 | -0.197 |
| HIV Protease (HIV/AIDS) | 0.766 | -0.043 |

The GNN improves on EGFR but does not generalize to BACE1 or HIV Protease under fixed hyperparameters — consistent with findings that GNN advantages are most reliable on balanced, structurally homogeneous datasets.

## Repository Structure

```
CS-439-Final-Project/
├── main.py                    # entry point - runs all 8 steps
├── data_filtering.py          # mmCIF file parser for EGFR, BACE1, COX2
├── data_api_fetch.py          # RCSB REST API fetcher for HIV Protease, Thrombin
├── data_standardize.py        # standardizes all 5 protein CSVs to 44 features
├── eda.py                     # exploratory data analysis - 6 figures
├── preprocessing.py           # drug datasets + Morgan fingerprints + train/test splits
├── baseline_model.py          # logistic regression, random forest, XGBoost
├── GGN.py                     # graph attention network
├── visualize_molecules.py     # 2D drug structure and property visualizations
├── requirements.txt
├── README.md
├── Preprocessed_Results/      # 5 PDB feature CSVs (one per protein)
├── processed_data/            # train/test splits (HIV files regenerated on first run)
└── results/
    ├── baseline_results.csv
    ├── gnn_results.csv
    ├── Baseline/              # ROC curves, confusion matrices, comparison plot
    ├── EDA/                   # 6 EDA figures
    ├── EGFR_gnn/              # EGFR GNN loss curve
    ├── BACE1_gnn/             # BACE1 GNN loss curve
    ├── HIV_Protease_gnn/      # HIV Protease GNN loss curve
    └── Visualization/         # molecular structure and property plots
```

## Notes

- HIV Protease train/test files are not included in the repository due to size (~325MB). They are regenerated automatically when `preprocessing.py` runs.
- The EGFR dataset contains only 68 unique drugs, making results on that protein sensitive to the random split. Drug-level splitting is applied to prevent leakage.
- BACE1 GNN underperforms the baseline — this is an honest finding reported in the paper rather than a bug.

## Authors

Ali Ugur, Dipen Patel, Sarthak Gandotra — Rutgers University, CS-439
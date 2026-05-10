# =============================================================
# GGN.py
# CS-439 Final Project — Rutgers University
# =============================================================
# This script trains a Graph Attention Network (GATConv) that
# represents each drug molecule as a graph with atoms as nodes
# and bonds as edges. It learns molecular structure patterns
# directly from the graph, combines them with PDB protein features,
# and predicts drug-protein binding. Drug-level splitting prevents
# data leakage. Undersampling handles HIV class imbalance.
# Trained on GPU if available, with protein-specific hyperparameters.
# =============================================================
# Libraries:
#   os             : file and folder management
#   random         : reproducible shuffling and undersampling
#   numpy          : numeric array operations
#   pandas         : loading processed data and saving results
#   torch          : neural network training and GPU acceleration
#   torch_geometric: graph neural network layers and data loading
#   rdkit          : converting SMILES to molecular graphs
#   sklearn        : evaluation metrics and ROC curve analysis
#   matplotlib     : plotting training loss curves
# =============================================================

import os
import pdb
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# setting up the paths
BASE_PATH = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
DATA_PATH = os.path.join(BASE_PATH, 'processed_data')
RAW_PATH  = os.path.join(BASE_PATH, 'raw_datasets')
FIG_PATH  = os.path.join(BASE_PATH, 'results')
RES_PATH  = os.path.join(BASE_PATH, 'results')

# setting random seeds so results are reproducible
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# training settings
BATCH_SIZE = 32
HIDDEN_DIM = 128

# per-protein settings based on dataset characteristics
PROTEIN_CONFIG = {
    'EGFR': {
        'epochs'       : 150,
        'learning_rate': 0.001,
        'undersample'  : False,
        'scheduler'    : False,
    },
    'BACE1': {
        'epochs'       : 150,
        'learning_rate': 0.0001,
        'undersample'  : False,
        'scheduler'    : True,
    },
    'HIV_Protease': {
        'epochs'            : 150,
        'learning_rate'     : 0.001,
        'undersample'       : True,
        'undersample_ratio' : 5,
        'scheduler'         : False,
    },
}

# using gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# this converts an atom into a feature vector
def get_atom_features(atom):
    # one-hot encoding for atom type
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other']
    symbol     = atom.GetSymbol()
    atom_type  = [1 if symbol == a else 0 for a in atom_types[:-1]]
    atom_type.append(1 if symbol not in atom_types[:-1] else 0)

    # other atom properties
    features = atom_type + [
        atom.GetDegree() / 10.0,
        atom.GetTotalValence() / 10.0,
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetFormalCharge() / 5.0,
        atom.GetTotalNumHs() / 8.0,
    ]
    return features


# this converts a bond into a feature vector
def get_bond_features(bond):
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type = [1 if bond.GetBondType() == b else 0 for b in bond_types]
    features  = bond_type + [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
    return features


# this converts a smiles string into a PyTorch Geometric graph
# we store the smiles string in the graph so we can do drug-level splitting
def smiles_to_graph(smiles, label, pdb_features):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    # getting atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    # getting bond connections and features
    edge_index = []
    edge_attr  = []
    for bond in mol.GetBonds():
        i         = bond.GetBeginAtomIdx()
        j         = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        # adding both directions since graph is undirected
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    # skipping molecules with no bonds
    if len(edge_index) == 0:
        return None

    # converting to tensors
    x          = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
    y          = torch.tensor([label], dtype=torch.float)
    # storing pdb as 2D so PyG stacks it correctly when batching
    pdb        = torch.tensor(pdb_features, dtype=torch.float).unsqueeze(0)

    # creating the graph and storing the smiles string
    # this is needed for drug-level splitting to prevent data leakage
    data        = Data(x=x, edge_index=edge_index,
                       edge_attr=edge_attr, y=y, pdb=pdb)
    data.smiles = str(smiles)

    return data


# the GNN model architecture
class DrugProteinGNN(nn.Module):
    def __init__(self, atom_feat_dim, pdb_feat_dim, hidden_dim=64):
        super(DrugProteinGNN, self).__init__()

        # three graph convolutional layers to learn drug structure
        self.conv1 = GATConv(atom_feat_dim, hidden_dim, edge_dim=6, heads=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=6, heads=1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, edge_dim=6, heads=1)

        # batch normalization to stabilize training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # fully connected layers combining drug and protein features
        self.fc1 = nn.Linear(hidden_dim + pdb_feat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch, pdb, edge_attr = (
        data.x, data.edge_index, data.batch, data.pdb, data.edge_attr
    )

        # passing through graph attention layers with edge features
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # pooling all atom features into one molecule representation
        x = global_mean_pool(x, batch)

        # pdb is already (batch_size, pdb_feat_dim) after PyG batching
        combined = torch.cat([x, pdb], dim=1)

        # final prediction layers
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = F.relu(self.fc2(combined))
        out      = self.fc3(combined)

        return out.squeeze()


# this loads the smiles from the raw dataset files
def get_smiles_for_protein(protein_name):
    if protein_name == 'EGFR':
        drugs    = pd.read_csv(os.path.join(RAW_PATH, 'davis_drugs.csv'))
        proteins = pd.read_csv(os.path.join(RAW_PATH, 'davis_proteins.csv'))
        affinity = pd.read_csv(os.path.join(RAW_PATH, 'davis_affinity.csv'))

        # filtering for egfr only using gene name
        egfr_rows = proteins[
            proteins['Gene_Name'].str.upper().str.startswith('EGFR')
        ]
        egfr_ids  = egfr_rows['Protein_Index'].tolist()
        egfr_aff  = affinity[affinity['Protein_Index'].isin(egfr_ids)]
        merged    = egfr_aff.merge(
            drugs[['Drug_Index', 'Canonical_SMILES']],
            on='Drug_Index', how='left'
        )
        # using pKd >= 7.0 threshold as per DeepDTA paper
        merged['label'] = (merged['Affinity'] >= 7.0).astype(int)
        return merged[['Canonical_SMILES', 'label']].rename(
            columns={'Canonical_SMILES': 'smiles'}
        )

    elif protein_name == 'BACE1':
        bace          = pd.read_csv(os.path.join(RAW_PATH, 'bace.csv'))
        bace['label'] = bace['Class'].astype(int)
        return bace[['mol', 'label']].rename(columns={'mol': 'smiles'})

    elif protein_name == 'HIV_Protease':
        hiv           = pd.read_csv(os.path.join(RAW_PATH, 'HIV.csv'))
        hiv['label']  = hiv['HIV_active'].astype(int)
        return hiv[['smiles', 'label']]

    return None


# this builds the graph dataset for a protein
def build_graph_dataset(protein_name, processed_df, smiles_df, config):
    print(f"  Building graphs for {protein_name}...")

    # getting the pdb feature columns
    pdb_cols = [c for c in processed_df.columns if c.startswith('pdb_')]

    # computing mean pdb features across all high quality structures
    pdb_mean = processed_df[pdb_cols].mean().values

    graphs = []
    failed = 0

    for _, row in smiles_df.iterrows():
        graph = smiles_to_graph(row['smiles'], row['label'], pdb_mean)
        if graph is not None:
            graphs.append(graph)
        else:
            failed += 1

    print(f"  Graphs built: {len(graphs)} | Failed: {failed}")

    # undersampling non-binders for HIV to fix class imbalance
    if config.get('undersample', False):
        ratio       = config.get('undersample_ratio', 5)
        binders     = [g for g in graphs if g.y.item() == 1]
        non_binders = [g for g in graphs if g.y.item() == 0]
        n_sample    = min(len(binders) * ratio, len(non_binders))
        non_binders = random.sample(non_binders, n_sample)
        graphs      = binders + non_binders
        random.shuffle(graphs)
        print(f"  After undersampling: {len(graphs)} graphs "
              f"({len(binders)} binders, {len(non_binders)} non-binders)")

    return graphs


# this splits graphs by unique drug to prevent data leakage
# without this the same drug appears in both train and test
# which inflates results because the model memorizes drug structures
def drug_level_split(graphs, train_ratio=0.8):
    # getting all unique smiles strings
    unique_smiles = list(set([g.smiles for g in graphs]))
    random.shuffle(unique_smiles)

    # splitting unique drugs 80/20
    split_idx    = int(train_ratio * len(unique_smiles))
    train_smiles = set(unique_smiles[:split_idx])
    test_smiles  = set(unique_smiles[split_idx:])

    # assigning each graph to train or test based on its drug
    train_graphs = [g for g in graphs if g.smiles in train_smiles]
    test_graphs  = [g for g in graphs if g.smiles in test_smiles]

    print(f"  Unique drugs total : {len(unique_smiles)}")
    print(f"  Train drugs        : {len(train_smiles)} "
          f"({len(train_graphs)} pairs)")
    print(f"  Test drugs         : {len(test_smiles)} "
          f"({len(test_graphs)} pairs)")

    return train_graphs, test_graphs


# this trains the model for one epoch
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out   = model(batch)
        loss  = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# this evaluates the model using the optimal threshold from the roc curve
def evaluate_model(model, loader):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            out    = model(batch)
            probs  = torch.sigmoid(out).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # finding the best threshold from the roc curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # falling back to 0.5 if threshold is inf or nan
    if np.isinf(optimal_threshold) or np.isnan(optimal_threshold):
        optimal_threshold = 0.5

    all_preds = (all_probs >= optimal_threshold).astype(int)
    print(f"  Optimal threshold: {optimal_threshold:.3f}")

    return {
        'ROC-AUC'  : roc_auc_score(all_labels, all_probs),
        'F1'       : f1_score(all_labels, all_preds, zero_division=0),
        'Precision': precision_score(all_labels, all_preds, zero_division=0),
        'Recall'   : recall_score(all_labels, all_preds, zero_division=0),
    }


# this plots the training loss curve and saves to the protein subfolder
def plot_training_loss(protein_name, losses):
    # saving to protein specific subfolder
    fig_path = os.path.join(FIG_PATH, f'{protein_name}_gnn')
    os.makedirs(fig_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='#2196F3', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{protein_name} - GNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_path, f'{protein_name}_gnn_loss.png'), dpi=150
    )
    plt.close()
    print(f"  Saved: {protein_name}_gnn/{protein_name}_gnn_loss.png")


# running the gnn for each protein
proteins    = ['EGFR', 'BACE1', 'HIV_Protease']
gnn_results = []

for protein_name in proteins:
    print(f"\n{'='*55}")
    print(f"GNN - {protein_name}")
    print(f"{'='*55}")

    config = PROTEIN_CONFIG[protein_name]

    # loading the processed data to get pdb features
    train_path = os.path.join(DATA_PATH, f'{protein_name}_train.csv')
    test_path  = os.path.join(DATA_PATH, f'{protein_name}_test.csv')

    if not os.path.exists(train_path):
        print(f"  {protein_name}_train.csv not found, skipping")
        continue

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # loading the original smiles strings
    smiles_df = get_smiles_for_protein(protein_name)
    if smiles_df is None:
        continue

    print(f"  Total drug-protein pairs: {len(smiles_df)}")
    print(f"  Epochs: {config['epochs']} | "
          f"LR: {config['learning_rate']} | "
          f"Undersample: {config['undersample']}")

    # building graphs
    all_graphs = build_graph_dataset(
        protein_name, train_df, smiles_df, config
    )

    if len(all_graphs) < 10:
        print(f"  Not enough graphs, skipping")
        continue

    # splitting by unique drug to prevent the same drug
    # appearing in both train and test — this avoids data leakage
    print(f"\n  Splitting by unique drug (prevents data leakage)...")
    train_graphs, test_graphs = drug_level_split(all_graphs, train_ratio=0.8)

    if len(test_graphs) == 0:
        print(f"  No test graphs after split, skipping")
        continue

    print(f"  Train pairs: {len(train_graphs)} | "
          f"Test pairs: {len(test_graphs)}")

    # creating data loaders
    train_loader = DataLoader(
        train_graphs, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader  = DataLoader(
        test_graphs, batch_size=BATCH_SIZE, shuffle=False
    )

    # getting feature dimensions
    atom_feat_dim = all_graphs[0].x.shape[1]
    pdb_feat_dim  = all_graphs[0].pdb.shape[1]
    print(f"  Atom features: {atom_feat_dim} | PDB features: {pdb_feat_dim}")

    # setting up the model
    model = DrugProteinGNN(
        atom_feat_dim, pdb_feat_dim, HIDDEN_DIM
    ).to(device)

    # calculating class weight for imbalanced data
    labels     = [g.y.item() for g in all_graphs]
    pos_count  = sum(labels)
    neg_count  = len(labels) - pos_count
    pos_weight = torch.tensor(
        [neg_count / pos_count], dtype=torch.float
    ).to(device)

    # using weighted binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # adding learning rate scheduler for bace1
    scheduler = None
    if config.get('scheduler', False):
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        print(f"  Using LR scheduler (step=30, gamma=0.5)")

    # training loop
    print(f"\n  Training for {config['epochs']} epochs...")
    losses = []

    for epoch in range(config['epochs']):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        losses.append(loss)

        if scheduler:
            scheduler.step()

        # printing progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{config['epochs']} | "
                  f"Loss: {loss:.4f} | LR: {lr_now:.6f}")

    # evaluating on test set
    print(f"\n  Evaluating on test set...")
    metrics            = evaluate_model(model, test_loader)
    metrics['Protein'] = protein_name
    metrics['Model']   = 'GNN'
    gnn_results.append(metrics)

    print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")
    print(f"  F1-Score : {metrics['F1']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")

    # plotting training loss
    plot_training_loss(protein_name, losses)

    # saving the trained model
    model_path = os.path.join(RES_PATH, f'{protein_name}_gnn.pt')
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {protein_name}_gnn.pt")


# printing the final summary
print(f"\n{'='*55}")
print("GNN RESULTS SUMMARY")
print(f"{'='*55}")

if gnn_results:
    results_df = pd.DataFrame(gnn_results)
    results_df = results_df[
        ['Protein', 'Model', 'ROC-AUC', 'F1', 'Precision', 'Recall']
    ]

    for col in ['ROC-AUC', 'F1', 'Precision', 'Recall']:
        results_df[col] = results_df[col].round(4)

    print(results_df.to_string(index=False))

    # saving results
    gnn_results_path = os.path.join(RES_PATH, 'gnn_results.csv')
    results_df.to_csv(gnn_results_path, index=False)
    print(f"\nResults saved to: gnn_results.csv")
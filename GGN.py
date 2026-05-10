import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
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
FIG_PATH  = os.path.join(BASE_PATH, 'results', 'figures')
RES_PATH  = os.path.join(BASE_PATH, 'results')

# training settings
EPOCHS        = 50
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
HIDDEN_DIM    = 64
RANDOM_STATE  = 42

# using gpu if available, otherwise cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# this converts an atom into a feature vector
def get_atom_features(atom):
    # one-hot encoding for atom type (most common atoms in drug molecules)
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
    # bond type as one-hot encoding
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
def smiles_to_graph(smiles, label, pdb_features):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    # getting atom features for each atom in the molecule
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

        # adding both directions since the graph is undirected
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    # if molecule has no bonds skip it
    if len(edge_index) == 0:
        return None

    # converting everything to tensors
    x          = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
    y          = torch.tensor([label], dtype=torch.float)

    # storing pdb as shape (1, num_features) so PyG stacks them correctly
    # when batching — this makes pdb shape (batch_size, num_features)
    pdb = torch.tensor(pdb_features, dtype=torch.float).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pdb=pdb)


# the GNN model architecture
class DrugProteinGNN(nn.Module):
    def __init__(self, atom_feat_dim, pdb_feat_dim, hidden_dim=64):
        super(DrugProteinGNN, self).__init__()

        # three graph convolutional layers to learn drug structure
        self.conv1 = GCNConv(atom_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # batch normalization to stabilize training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # fully connected layers combining drug and protein features
        self.fc1 = nn.Linear(hidden_dim + pdb_feat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch, pdb = data.x, data.edge_index, data.batch, data.pdb

        # passing through graph conv layers with relu activation
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # pooling all atom features into one molecule representation
        # x goes from (total_atoms, hidden) to (batch_size, hidden)
        x = global_mean_pool(x, batch)

        # pdb is already (batch_size, pdb_feat_dim) after PyG batching
        combined = torch.cat([x, pdb], dim=1)

        # final prediction layers
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = F.relu(self.fc2(combined))
        out      = self.fc3(combined)

        return out.squeeze()


# this loads the processed train/test csv for a protein
def load_smiles_and_pdb(protein_name):
    train_path = os.path.join(DATA_PATH, f'{protein_name}_train.csv')
    test_path  = os.path.join(DATA_PATH, f'{protein_name}_test.csv')

    if not os.path.exists(train_path):
        print(f"  {protein_name}_train.csv not found, skipping")
        return None, None

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


# this loads the original smiles strings from the raw dataset files
def get_smiles_for_protein(protein_name):
    if protein_name == 'EGFR':
        # loading davis drugs and affinity data for egfr
        drugs    = pd.read_csv(os.path.join(RAW_PATH, 'davis_drugs.csv'))
        proteins = pd.read_csv(os.path.join(RAW_PATH, 'davis_proteins.csv'))
        affinity = pd.read_csv(os.path.join(RAW_PATH, 'davis_affinity.csv'))

        # filtering for egfr only using gene name
        egfr_rows = proteins[proteins['Gene_Name'].str.upper().str.startswith('EGFR')]
        egfr_ids  = egfr_rows['Protein_Index'].tolist()

        # getting drug-protein pairs for egfr
        egfr_aff = affinity[affinity['Protein_Index'].isin(egfr_ids)]
        merged   = egfr_aff.merge(
            drugs[['Drug_Index', 'Canonical_SMILES']],
            on='Drug_Index', how='left'
        )
        merged['label'] = (merged['Affinity'] >= 7.0).astype(int)
        return merged[['Canonical_SMILES', 'label']].rename(
            columns={'Canonical_SMILES': 'smiles'}
        )

    elif protein_name == 'BACE1':
        bace = pd.read_csv(os.path.join(RAW_PATH, 'bace.csv'))
        bace['label'] = bace['Class'].astype(int)
        return bace[['mol', 'label']].rename(columns={'mol': 'smiles'})

    elif protein_name == 'HIV_Protease':
        hiv = pd.read_csv(os.path.join(RAW_PATH, 'HIV.csv'))
        hiv['label'] = hiv['HIV_active'].astype(int)
        return hiv[['smiles', 'label']]

    return None


# this builds the graph dataset for a protein
def build_graph_dataset(protein_name, processed_df, smiles_df):
    print(f"  Building graphs for {protein_name}...")

    # getting the pdb feature columns
    pdb_cols = [c for c in processed_df.columns if c.startswith('pdb_')]

    # computing mean pdb features across the processed data
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
    return graphs


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


# this evaluates the model on the test set
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
    all_preds  = (all_probs >= 0.5).astype(int)

    return {
        'ROC-AUC'  : roc_auc_score(all_labels, all_probs),
        'F1'       : f1_score(all_labels, all_preds, zero_division=0),
        'Precision': precision_score(all_labels, all_preds, zero_division=0),
        'Recall'   : recall_score(all_labels, all_preds, zero_division=0),
    }


# this plots the training loss over epochs
def plot_training_loss(protein_name, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='#2196F3', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{protein_name} - GNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, f'{protein_name}_gnn_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved: {protein_name}_gnn_loss.png")


# running the gnn for each protein
proteins    = ['EGFR', 'BACE1', 'HIV_Protease']
gnn_results = []

for protein_name in proteins:
    print(f"\n{'='*50}")
    print(f"GNN - {protein_name}")
    print(f"{'='*50}")

    # loading the processed data to get pdb features
    train_df, test_df = load_smiles_and_pdb(protein_name)
    if train_df is None:
        continue

    # loading the original smiles strings
    smiles_df = get_smiles_for_protein(protein_name)
    if smiles_df is None:
        continue

    print(f"  Total drug-protein pairs: {len(smiles_df)}")

    # building graphs for all drug-protein pairs
    all_graphs = build_graph_dataset(protein_name, train_df, smiles_df)

    if len(all_graphs) < 10:
        print(f"  Not enough graphs, skipping {protein_name}")
        continue

    # splitting into train and test 80/20
    split        = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:split]
    test_graphs  = all_graphs[split:]
    print(f"  Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")

    # creating data loaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

    # getting feature dimensions
    # pdb is stored as (1, pdb_feat_dim) so we use shape[1] not shape[0]
    atom_feat_dim = all_graphs[0].x.shape[1]
    pdb_feat_dim  = all_graphs[0].pdb.shape[1]
    print(f"  Atom features: {atom_feat_dim} | PDB features: {pdb_feat_dim}")

    # setting up the model
    model = DrugProteinGNN(atom_feat_dim, pdb_feat_dim, HIDDEN_DIM).to(device)

    # calculating class weight for imbalanced data
    labels     = [g.y.item() for g in all_graphs]
    pos_count  = sum(labels)
    neg_count  = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float).to(device)

    # using weighted binary cross entropy loss to handle class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    print(f"\n  Training for {EPOCHS} epochs...")
    losses = []

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        losses.append(loss)

        # printing progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    # evaluating on test set
    print(f"\n  Evaluating on test set...")
    metrics = evaluate_model(model, test_loader)
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
print(f"\n{'='*50}")
print("GNN RESULTS SUMMARY")
print(f"{'='*50}")

if gnn_results:
    results_df = pd.DataFrame(gnn_results)
    results_df = results_df[['Protein', 'Model', 'ROC-AUC',
                              'F1', 'Precision', 'Recall']]

    for col in ['ROC-AUC', 'F1', 'Precision', 'Recall']:
        results_df[col] = results_df[col].round(4)

    print(results_df.to_string(index=False))

    # saving results
    gnn_results_path = os.path.join(RES_PATH, 'gnn_results.csv')
    results_df.to_csv(gnn_results_path, index=False)
    print(f"\nResults saved to: gnn_results.csv")

print(f"\nNext: compare GNN results with baseline models")
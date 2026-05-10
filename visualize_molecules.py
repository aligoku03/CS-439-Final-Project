import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# setting up the paths
BASE_PATH = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
RAW_PATH  = os.path.join(BASE_PATH, 'raw_datasets')
FIG_PATH  = os.path.join(BASE_PATH, 'results', 'figures')

os.makedirs(FIG_PATH, exist_ok=True)


# this loads smiles and labels for each protein
def load_protein_data(protein_name):
    if protein_name == 'EGFR':
        drugs    = pd.read_csv(os.path.join(RAW_PATH, 'davis_drugs.csv'))
        proteins = pd.read_csv(os.path.join(RAW_PATH, 'davis_proteins.csv'))
        affinity = pd.read_csv(os.path.join(RAW_PATH, 'davis_affinity.csv'))

        egfr_rows = proteins[proteins['Gene_Name'].str.upper().str.startswith('EGFR')]
        egfr_ids  = egfr_rows['Protein_Index'].tolist()
        egfr_aff  = affinity[affinity['Protein_Index'].isin(egfr_ids)]
        merged    = egfr_aff.merge(
            drugs[['Drug_Index', 'Canonical_SMILES']],
            on='Drug_Index', how='left'
        )
        merged['label'] = (merged['Affinity'] >= 7.0).astype(int)
        return merged[['Canonical_SMILES', 'label', 'Affinity']].rename(
            columns={'Canonical_SMILES': 'smiles', 'Affinity': 'affinity'}
        )

    elif protein_name == 'BACE1':
        bace = pd.read_csv(os.path.join(RAW_PATH, 'bace.csv'))
        bace['label']    = bace['Class'].astype(int)
        bace['affinity'] = bace['pIC50']
        return bace[['mol', 'label', 'affinity']].rename(columns={'mol': 'smiles'})

    elif protein_name == 'HIV_Protease':
        hiv = pd.read_csv(os.path.join(RAW_PATH, 'HIV.csv'))
        hiv['label']    = hiv['HIV_active'].astype(int)
        hiv['affinity'] = hiv['HIV_active']
        return hiv[['smiles', 'label', 'affinity']]

    return None


# this draws a molecule and returns it as a PIL image
def smiles_to_image(smiles, size=(400, 300)):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    # generating 2D coordinates for the molecule
    AllChem.Compute2DCoords(mol)

    # drawing the molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # converting svg to png using rdkit
    drawer2 = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer2.DrawMolecule(mol)
    drawer2.FinishDrawing()
    png_data = drawer2.GetDrawingText()

    img = Image.open(BytesIO(png_data))
    return img


# this gets the molecular formula and weight for a smiles string
def get_mol_info(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 'N/A', 0
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    weight  = round(Descriptors.MolWt(mol), 2)
    return formula, weight


# this draws a grid of binder vs non-binder molecules for one protein
def visualize_binders_vs_nonbinders(protein_name, df, n_samples=4):
    print(f"  Visualizing {protein_name}...")

    # getting binders and non-binders
    binders     = df[df['label'] == 1].head(n_samples)
    non_binders = df[df['label'] == 0].head(n_samples)

    # making sure we have enough samples
    n_bind    = len(binders)
    n_nonbind = len(non_binders)
    n_cols    = max(n_bind, n_nonbind)

    if n_cols == 0:
        print(f"  No samples found for {protein_name}, skipping")
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 9))

    # making axes always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # drawing binders in the top row
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_bind:
            row    = binders.iloc[i]
            img    = smiles_to_image(row['smiles'])
            formula, weight = get_mol_info(row['smiles'])
            if img:
                ax.imshow(img)
                ax.set_title(
                    f'Binder\n{formula}\nMW: {weight} g/mol',
                    fontsize=9, fontweight='bold', color='green'
                )
            else:
                ax.text(0.5, 0.5, 'Invalid SMILES',
                        ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # drawing non-binders in the bottom row
    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_nonbind:
            row    = non_binders.iloc[i]
            img    = smiles_to_image(row['smiles'])
            formula, weight = get_mol_info(row['smiles'])
            if img:
                ax.imshow(img)
                ax.set_title(
                    f'Non-Binder\n{formula}\nMW: {weight} g/mol',
                    fontsize=9, fontweight='bold', color='red'
                )
            else:
                ax.text(0.5, 0.5, 'Invalid SMILES',
                        ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # adding row labels
    fig.text(0.01, 0.73, 'BINDERS', va='center', rotation='vertical',
             fontsize=13, fontweight='bold', color='green')
    fig.text(0.01, 0.27, 'NON-BINDERS', va='center', rotation='vertical',
             fontsize=13, fontweight='bold', color='red')

    disease = {
        'EGFR'        : 'Cancer',
        'BACE1'       : "Alzheimer's",
        'HIV_Protease': 'HIV/AIDS'
    }

    plt.suptitle(
        f'{protein_name} ({disease.get(protein_name, "")}) —'
        f' Drug Structures: Binders vs Non-Binders',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    save_path = os.path.join(FIG_PATH, f'{protein_name}_molecules.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {protein_name}_molecules.png")


# this draws a property comparison between binders and non-binders
def visualize_molecular_properties(protein_name, df):
    print(f"  Plotting molecular properties for {protein_name}...")

    # calculating properties for each molecule
    mw_binders     = []
    mw_nonbinders  = []
    log_binders    = []
    log_nonbinders = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row['smiles']))
        if mol is None:
            continue
        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        if row['label'] == 1:
            mw_binders.append(mw)
            log_binders.append(logp)
        else:
            mw_nonbinders.append(mw)
            log_nonbinders.append(logp)

    if not mw_binders or not mw_nonbinders:
        print(f"  Not enough data for {protein_name}, skipping properties")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # molecular weight distribution
    axes[0].hist(mw_binders, bins=20, alpha=0.7, color='green',
                 label=f'Binders (n={len(mw_binders)})', edgecolor='white')
    axes[0].hist(mw_nonbinders, bins=20, alpha=0.7, color='red',
                 label=f'Non-Binders (n={len(mw_nonbinders)})', edgecolor='white')
    axes[0].set_xlabel('Molecular Weight (g/mol)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Molecular Weight Distribution')
    axes[0].legend()
    axes[0].axvline(x=500, color='black', linestyle='--',
                    alpha=0.5, label="Lipinski's rule (500 Da)")

    # logP distribution
    axes[1].hist(log_binders, bins=20, alpha=0.7, color='green',
                 label=f'Binders (n={len(log_binders)})', edgecolor='white')
    axes[1].hist(log_nonbinders, bins=20, alpha=0.7, color='red',
                 label=f'Non-Binders (n={len(log_nonbinders)})', edgecolor='white')
    axes[1].set_xlabel('LogP (lipophilicity)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('LogP Distribution')
    axes[1].legend()
    axes[1].axvline(x=5, color='black', linestyle='--',
                    alpha=0.5, label="Lipinski's rule (LogP=5)")

    disease = {
        'EGFR'        : 'Cancer',
        'BACE1'       : "Alzheimer's",
        'HIV_Protease': 'HIV/AIDS'
    }
    plt.suptitle(
        f'{protein_name} ({disease.get(protein_name, "")}) —'
        f' Molecular Properties: Binders vs Non-Binders',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(FIG_PATH, f'{protein_name}_properties.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {protein_name}_properties.png")


# running the visualizations for all proteins
proteins = ['EGFR', 'BACE1', 'HIV_Protease']

for protein_name in proteins:
    print(f"\n{'='*50}")
    print(f"Visualizing: {protein_name}")
    print(f"{'='*50}")

    df = load_protein_data(protein_name)
    if df is None:
        print(f"  Could not load data for {protein_name}")
        continue

    binders     = df[df['label'] == 1]
    non_binders = df[df['label'] == 0]
    print(f"  Binders: {len(binders)} | Non-binders: {len(non_binders)}")

    # drawing molecular structures
    visualize_binders_vs_nonbinders(protein_name, df, n_samples=4)

    # drawing molecular property distributions
    visualize_molecular_properties(protein_name, df)

print(f"\n{'='*50}")
print("VISUALIZATION COMPLETE")
print(f"{'='*50}")
print(f"\nFigures saved to: {FIG_PATH}")
print("Files created:")
for protein_name in proteins:
    print(f"  {protein_name}_molecules.png")
    print(f"  {protein_name}_properties.png")
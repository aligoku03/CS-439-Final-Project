import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_PATH         = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
DATA_PATH         = os.path.join(BASE_PATH, 'processed_data')
RAW_PATH          = os.path.join(BASE_PATH, 'raw_datasets')
RESOLUTION_CUTOFF = 2.5
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
MORGAN_RADIUS     = 2
MORGAN_BITS       = 2048
BINDING_THRESHOLD = 1000  # nM

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RAW_PATH,  exist_ok=True)

# Direct download URLs — no DeepChem needed
DATASET_URLS = {
    'HIV': {
        'url'    : 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
        'file'   : 'HIV.csv',
        'protein': 'HIV_Protease',
        'smiles_col': 'smiles',
        'label_col' : 'HIV_active',
    },
    'BACE': {
        'url'    : 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
        'file'   : 'bace.csv',
        'protein': 'BACE1',
        'smiles_col': 'mol',
        'label_col' : 'Class',
    },
}
# ─────────────────────────────────────────────────────────────────────────────


# ── HELPERS ───────────────────────────────────────────────────────────────────
def download_file(url, dest_path):
    """Download a file from URL if not already cached."""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f'    Already cached ({size/1024:.0f} KB) — skipping download')
        return True
    try:
        print(f'    Downloading from {url}...')
        urllib.request.urlretrieve(url, dest_path)
        size = os.path.getsize(dest_path)
        print(f'    Downloaded successfully ({size/1024:.0f} KB)')
        return True
    except Exception as e:
        print(f'    Download failed: {e}')
        return False


def smiles_to_morgan(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """Convert SMILES string to Morgan fingerprint vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)


def get_protein_feature_vector(protein_name):
    """
    Load PDB structural features for a protein.
    Filter to high quality structures (<=2.5Å).
    Return mean feature vector.
    """
    path = os.path.join(BASE_PATH, f'{protein_name}_pdb_features.csv')
    if not os.path.exists(path):
        print(f'    WARNING: {protein_name}_pdb_features.csv not found')
        return None, []

    df = pd.read_csv(path)
    df = df[df['resolution'] <= RESOLUTION_CUTOFF].copy()

    exclude_cols = [
        'pdb_id', 'protein', 'title', 'keywords', 'method',
        'space_group', 'organism', 'protein_sequence'
    ]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_vector = df[feature_cols].mean().fillna(0)
    print(f'    {protein_name}: {len(df)} structures | '
          f'{len(feature_cols)} PDB features')
    return feature_vector, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — COMPUTE PROTEIN FEATURE VECTORS FROM PDB
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: PROTEIN FEATURE VECTORS FROM PDB")
print("=" * 60)

protein_features     = {}
protein_feature_cols = {}

for protein in ['EGFR', 'BACE1', 'HIV_Protease']:
    print(f'\n  {protein}:')
    vec, cols = get_protein_feature_vector(protein)
    if vec is not None:
        protein_features[protein]     = vec
        protein_feature_cols[protein] = cols

print(f'\n  Ready: {list(protein_features.keys())}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DOWNLOAD DRUG DATASETS DIRECTLY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: DOWNLOADING DRUG DATASETS")
print("        (No DeepChem needed)")
print("=" * 60)

drug_datasets = {}

# ── HIV and BACE from MoleculeNet S3 ─────────────────────────────────────────
for name, config in DATASET_URLS.items():
    print(f'\n  {name} ({config["protein"]}):')
    dest = os.path.join(RAW_PATH, config['file'])
    ok   = download_file(config['url'], dest)

    if ok:
        df = pd.read_csv(dest)
        print(f'    Raw data: {len(df)} rows, columns: {df.columns.tolist()}')

        smiles_col = config['smiles_col']
        label_col  = config['label_col']

        if smiles_col not in df.columns:
            # Try to find the SMILES column
            for col in df.columns:
                if 'smiles' in col.lower() or 'mol' in col.lower():
                    smiles_col = col
                    break

        if label_col not in df.columns:
            for col in df.columns:
                if 'class' in col.lower() or 'active' in col.lower() \
                        or 'label' in col.lower():
                    label_col = col
                    break

        result_df = pd.DataFrame({
            'smiles' : df[smiles_col],
            'label'  : pd.to_numeric(df[label_col], errors='coerce').astype(int),
            'protein': config['protein']
        }).dropna()

        drug_datasets[config['protein']] = result_df
        binders = result_df['label'].sum()
        print(f'    Loaded: {len(result_df)} compounds | '
              f'binders={binders} | non-binders={len(result_df)-binders}')


# ── DAVIS for EGFR ────────────────────────────────────────────────────────────
print('\n  DAVIS (EGFR):')

davis_drugs_url    = 'https://raw.githubusercontent.com/dingyan20/Davis-Dataset-for-DTA-Prediction/main/drugs.csv'
davis_proteins_url = 'https://raw.githubusercontent.com/dingyan20/Davis-Dataset-for-DTA-Prediction/main/proteins.csv'
davis_affinity_url = 'https://raw.githubusercontent.com/dingyan20/Davis-Dataset-for-DTA-Prediction/main/drug_protein_affinity.csv'

drugs_path    = os.path.join(RAW_PATH, 'davis_drugs.csv')
proteins_path = os.path.join(RAW_PATH, 'davis_proteins.csv')
affinity_path = os.path.join(RAW_PATH, 'davis_affinity.csv')

ok1 = download_file(davis_drugs_url,    drugs_path)
ok2 = download_file(davis_proteins_url, proteins_path)
ok3 = download_file(davis_affinity_url, affinity_path)

if ok1 and ok2 and ok3:
    drugs    = pd.read_csv(drugs_path)
    proteins = pd.read_csv(proteins_path)
    affinity = pd.read_csv(affinity_path)

    print(f'    Drugs    : {len(drugs)} | columns: {drugs.columns.tolist()}')
    print(f'    Proteins : {len(proteins)} | columns: {proteins.columns.tolist()}')
    print(f'    Affinity : {len(affinity)} | columns: {affinity.columns.tolist()}')

    # Filter for EGFR
    egfr_rows = proteins[proteins.apply(
        lambda r: 'EGFR' in str(r.values).upper(), axis=1
    )]
    print(f'    EGFR matches: {len(egfr_rows)} proteins')
    print(egfr_rows)

    if len(egfr_rows) > 0:
        egfr_ids = egfr_rows['Protein_Index'].tolist()

        # Get affinity rows for EGFR
        egfr_affinity = affinity[affinity['Protein_Index'].isin(egfr_ids)]

        # Merge with drug SMILES
        egfr_merged = egfr_affinity.merge(
        drugs[['Drug_Index', 'Canonical_SMILES']], on='Drug_Index', how='left'
)

        # Binary label: Kd <= 1000 nM = binder
        egfr_merged['label'] = (
            egfr_merged['Affinity'] >= 7.0
        ).astype(int)

        davis_df = pd.DataFrame({
            'smiles' : egfr_merged['Canonical_SMILES'],
            'label'  : egfr_merged['label'],
            'protein': 'EGFR'
        }).dropna()

        drug_datasets['EGFR'] = davis_df
        binders = davis_df['label'].sum()
        print(f'    Loaded: {len(davis_df)} pairs | '
              f'binders={binders} | non-binders={len(davis_df)-binders}')
    else:
        print('    EGFR not found in DAVIS proteins')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MORGAN FINGERPRINTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: MORGAN FINGERPRINTS")
print(f"        Radius={MORGAN_RADIUS} | Bits={MORGAN_BITS}")
print("=" * 60)

fingerprint_datasets = {}

for protein_name, drug_df in drug_datasets.items():
    print(f'\n  {protein_name} ({len(drug_df)} compounds)...')

    fingerprints  = []
    valid_indices = []
    failed        = 0

    for idx, row in drug_df.iterrows():
        fp = smiles_to_morgan(row['smiles'])
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)
        else:
            failed += 1

    fp_cols  = [f'fp_{i}' for i in range(MORGAN_BITS)]
    fp_df    = pd.DataFrame(fingerprints, columns=fp_cols)
    valid_df = drug_df.loc[valid_indices].reset_index(drop=True)
    combined = pd.concat([valid_df.reset_index(drop=True), fp_df], axis=1)

    fingerprint_datasets[protein_name] = combined
    print(f'    Valid: {len(combined)} | Failed: {failed}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COMBINE DRUG + PROTEIN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: COMBINING DRUG + PROTEIN FEATURES")
print("=" * 60)

combined_datasets = {}

for protein_name, fp_df in fingerprint_datasets.items():
    if protein_name not in protein_features:
        print(f'  WARNING: No PDB features for {protein_name} — skipping')
        continue

    pdb_vec  = protein_features[protein_name]
    pdb_cols = protein_feature_cols[protein_name]

    pdb_df = pd.DataFrame(
        [pdb_vec.values] * len(fp_df),
        columns=[f'pdb_{c}' for c in pdb_cols]
    )

    combined = pd.concat([
        fp_df[['smiles', 'label', 'protein']],
        fp_df[[c for c in fp_df.columns if c.startswith('fp_')]],
        pdb_df
    ], axis=1)

    combined_datasets[protein_name] = combined
    n_total = MORGAN_BITS + len(pdb_cols)

    print(f'\n  {protein_name}:')
    print(f'    Examples         : {len(combined)}')
    print(f'    Drug features    : {MORGAN_BITS} (Morgan FP)')
    print(f'    Protein features : {len(pdb_cols)} (PDB structural)')
    print(f'    Total features   : {n_total}')
    print(f'    Binders (1)      : {combined["label"].sum()}')
    print(f'    Non-binders (0)  : {(combined["label"]==0).sum()}')
    print(f'    Class balance    : {combined["label"].mean()*100:.1f}% binders')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN/TEST SPLIT 80/20
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: TRAIN/TEST SPLIT (80/20)")
print("=" * 60)

split_datasets = {}

for protein_name, df in combined_datasets.items():
    X = df.drop(columns=['smiles', 'label', 'protein'])
    y = df['label']

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = y
        )
    except ValueError:
        # Not enough samples for stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE
        )

    split_datasets[protein_name] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
    }

    print(f'\n  {protein_name}:')
    print(f'    Train: {len(X_train)} | '
          f'binders={y_train.sum()} | non-binders={(y_train==0).sum()}')
    print(f'    Test : {len(X_test)}  | '
          f'binders={y_test.sum()} | non-binders={(y_test==0).sum()}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE FILES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: SAVING PROCESSED DATASETS")
print("=" * 60)

for protein_name, splits in split_datasets.items():
    train_df = splits['X_train'].copy()
    train_df.insert(0, 'label', splits['y_train'].values)

    test_df  = splits['X_test'].copy()
    test_df.insert(0, 'label', splits['y_test'].values)

    train_path = os.path.join(DATA_PATH, f'{protein_name}_train.csv')
    test_path  = os.path.join(DATA_PATH, f'{protein_name}_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f'\n  {protein_name}:')
    print(f'    {protein_name}_train.csv — '
          f'{len(train_df)} rows × {len(train_df.columns)} cols')
    print(f'    {protein_name}_test.csv  — '
          f'{len(test_df)} rows × {len(test_df.columns)} cols')


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f'\n  Proteins   : {list(split_datasets.keys())}')
print(f'  Resolution : ≤{RESOLUTION_CUTOFF}Å filter applied')
print(f'  Fingerprint: Morgan radius={MORGAN_RADIUS}, bits={MORGAN_BITS}')
print(f'  Split      : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test')
print(f'  Output     : {DATA_PATH}')

if split_datasets:
    p    = list(split_datasets.keys())[0]
    n    = len(split_datasets[p]['X_train'].columns)
    npdb = len(protein_feature_cols[p])
    print(f'\n  Feature breakdown:')
    print(f'    Morgan fingerprint : {MORGAN_BITS}')
    print(f'    PDB structural     : {npdb}')
    print(f'    Total              : {n}')

print('\n  Files saved:')
for p in split_datasets:
    print(f'    {p}_train.csv')
    print(f'    {p}_test.csv')

print('\n  Next: 04_baseline_models.py')
import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_PATH   = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

PROTEINS = {
    'EGFR'        : 'EGFR_pdb_features.csv',
    'BACE1'       : 'BACE1_pdb_features.csv',
    'COX2'        : 'COX2_pdb_features.csv',
    'HIV_Protease': 'HIV_Protease_pdb_features.csv',
    'Thrombin'    : 'Thrombin_pdb_features.csv',
}

NUMERIC_COLS = [
    'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
    'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
    'cell_length_a', 'cell_length_b', 'cell_length_c',
    'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
    'solvent_content', 'matthews_coeff',
]
# ─────────────────────────────────────────────────────────────────────────────


# ── HELPER ────────────────────────────────────────────────────────────────────
def compute_aa_composition(sequence):
    """
    Convert a protein sequence string into 20 amino acid frequency columns.
    Returns a dict of {aa_X: frequency_%} for all 20 amino acids.
    """
    if pd.isna(sequence) or not str(sequence).strip():
        return {f'aa_{aa}': np.nan for aa in AMINO_ACIDS}

    sequence = str(sequence).upper().strip()
    length   = len(sequence)

    if length == 0:
        return {f'aa_{aa}': np.nan for aa in AMINO_ACIDS}

    return {
        f'aa_{aa}': round(sequence.count(aa) / length * 100, 4)
        for aa in AMINO_ACIDS
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("DATA STANDARDIZATION")
print("Goal: 44 consistent features across all 5 proteins")
print("=" * 60)

all_dfs = {}

for protein_name, filename in PROTEINS.items():
    filepath = os.path.join(BASE_PATH, filename)

    if not os.path.exists(filepath):
        print(f"\n  WARNING: {filename} not found — skipping {protein_name}")
        continue

    print(f"\n{'─'*60}")
    print(f"Processing: {protein_name}")
    print(f"{'─'*60}")

    df = pd.read_csv(filepath)
    print(f"  Loaded   : {len(df)} structures, {len(df.columns)} features")

    # ── FIX 1: Add amino acid columns if missing ──────────────────────────────
    aa_cols_present = [f'aa_{aa}' for aa in AMINO_ACIDS
                       if f'aa_{aa}' in df.columns]

    if len(aa_cols_present) == 20:
        print(f"  AA cols  : Already present ✅")
    else:
        print(f"  AA cols  : Missing — calculating from protein_sequence...")

        if 'protein_sequence' not in df.columns:
            print(f"  WARNING  : No protein_sequence column found!")
        else:
            # Calculate amino acid composition for each row
            aa_data = df['protein_sequence'].apply(compute_aa_composition)
            aa_df   = pd.DataFrame(aa_data.tolist())

            # Add to main dataframe
            for aa in AMINO_ACIDS:
                col = f'aa_{aa}'
                df[col] = aa_df[col]

            filled = df['aa_L'].notna().sum()
            print(f"  AA cols  : Added 20 columns | "
                  f"filled for {filled}/{len(df)} rows ✅")

    # ── FIX 2: Convert numeric columns ───────────────────────────────────────
    for col in NUMERIC_COLS + [f'aa_{aa}' for aa in AMINO_ACIDS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── FIX 3: Fill missing numeric values with per-protein median ────────────
    cols_to_fill = NUMERIC_COLS + [f'aa_{aa}' for aa in AMINO_ACIDS]
    filled_report = []

    for col in cols_to_fill:
        if col not in df.columns:
            continue
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
                filled_report.append(
                    f"{col} ({missing_before} filled with {median_val:.3f})"
                )

    if filled_report:
        print(f"  Filled   :")
        for msg in filled_report:
            print(f"    → {msg}")
    else:
        print(f"  Fill     : No numeric missing values ✅")

    # ── FIX 4: Standardize column order ──────────────────────────────────────
    base_cols = [
        'pdb_id', 'protein', 'title', 'keywords', 'method',
        'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
        'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
        'cell_length_a', 'cell_length_b', 'cell_length_c',
        'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
        'space_group', 'solvent_content', 'matthews_coeff',
        'organism', 'protein_sequence',
    ]
    aa_cols   = [f'aa_{aa}' for aa in AMINO_ACIDS]
    col_order = [c for c in base_cols + aa_cols if c in df.columns]

    # Add any remaining columns not in our order
    remaining = [c for c in df.columns if c not in col_order]
    col_order += remaining

    df = df[col_order]

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(filepath, index=False)
    print(f"  Saved    : {len(df)} structures, {len(df.columns)} features → {filename}")

    all_dfs[protein_name] = df


# ── FINAL VALIDATION ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL VALIDATION — ALL 5 PROTEINS")
print(f"{'='*60}")

reference_cols = None

for protein_name, df in all_dfs.items():
    res    = df['resolution'].dropna()
    seq    = df['protein_sequence'].notna().sum()
    aa_ok  = all(f'aa_{aa}' in df.columns for aa in AMINO_ACIDS)
    miss   = df.isnull().sum().sum()

    print(f"\n  {protein_name}:")
    print(f"    Structures   : {len(df)}")
    print(f"    Features     : {len(df.columns)}")
    print(f"    AA cols      : {'✅ All 20 present' if aa_ok else '❌ Missing'}")
    print(f"    Avg res      : {res.mean():.2f} Å" if len(res) > 0 else "    Avg res      : N/A")
    print(f"    Has sequence : {seq}/{len(df)}")
    print(f"    Missing vals : {miss}")

    # Check column consistency
    if reference_cols is None:
        reference_cols = set(df.columns)
        print(f"    Columns      : ✅ Reference set ({len(df.columns)} cols)")
    else:
        diff = reference_cols.symmetric_difference(set(df.columns))
        if diff:
            print(f"    Columns      : ⚠️  Differs from reference: {diff}")
        else:
            print(f"    Columns      : ✅ Matches reference")

total = sum(len(df) for df in all_dfs.values())
print(f"\n{'='*60}")
print(f"Total structures : {total}")
print(f"Features per row : {len(list(all_dfs.values())[0].columns)}")
print(f"{'='*60}")
print("\nAll 5 files standardized and saved.")
print("Next Step: 03_preprocessing.py — Feature engineering + model prep")
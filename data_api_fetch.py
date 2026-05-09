import requests
import pandas as pd
import time
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_PATH = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'

PROTEINS = {
    'HIV_Protease': {
        'uniprot_id' : 'P03367',
        'description': 'HIV-1 Protease — HIV/AIDS',
        'output_file': 'HIV_Protease_pdb_features.csv',
    },
    'Thrombin': {
        'uniprot_id' : 'P00734',
        'description': 'Thrombin — Blood Clotting',
        'output_file': 'Thrombin_pdb_features.csv',
    },
}

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
# ─────────────────────────────────────────────────────────────────────────────


# ── HELPERS ───────────────────────────────────────────────────────────────────
def safe_get(data, *keys, default=None):
    """Safely navigate nested dict keys."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            data = data[key] if len(data) > key else default
        else:
            return default
    return data


def get_pdb_ids_for_uniprot(uniprot_id):
    """Get all PDB IDs linked to a UniProt ID via RCSB search API."""
    url   = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type"      : "terminal",
            "service"   : "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers"
                    ".reference_sequence_identifiers"
                    ".database_accession"
                ),
                "operator" : "exact_match",
                "value"    : uniprot_id
            }
        },
        "return_type"    : "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1000}}
    }
    try:
        r = requests.post(url, json=query, timeout=15)
        if r.status_code == 200:
            results = r.json().get('result_set', [])
            return [x['identifier'] for x in results]
    except Exception as e:
        print(f"  Search error: {e}")
    return []


def fetch_entry(pdb_id):
    """Fetch main entry data from RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def fetch_polymer_entity(pdb_id, entity_id=1):
    """Fetch polymer entity data (sequence, organism) from RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def sequence_to_aa_composition(sequence):
    """Convert protein sequence to amino acid frequency features."""
    if not sequence:
        return {f'aa_{aa}': None for aa in AMINO_ACIDS}
    sequence = str(sequence).upper()
    length   = len(sequence)
    if length == 0:
        return {f'aa_{aa}': None for aa in AMINO_ACIDS}
    return {f'aa_{aa}': round(sequence.count(aa) / length * 100, 4)
            for aa in AMINO_ACIDS}


def extract_all_features(pdb_id, protein_name, entry_data, entity_data):
    """
    Extract ALL features matching the mmCIF file format:
    - Basic info
    - Quality metrics (resolution, r_free, r_work, b_iso_mean, rmerge)
    - Crystal features (cell dimensions, angles, space group)
    - Atom counts
    - Solvent content, Matthews coefficient
    - Protein sequence and amino acid composition
    - Organism
    """
    record = {
        'pdb_id' : pdb_id,
        'protein': protein_name,
    }

    if not entry_data:
        return record

    # ── Basic info ────────────────────────────────────────────────────────────
    record['title']    = safe_get(entry_data, 'struct', 'title')
    record['keywords'] = safe_get(entry_data, 'struct_keywords',
                                  'pdbx_keywords')

    # ── Experimental method ───────────────────────────────────────────────────
    exptl = safe_get(entry_data, 'exptl', default=[{}])
    record['method'] = safe_get(exptl, 0, 'method')

    # ── Resolution ────────────────────────────────────────────────────────────
    refine = safe_get(entry_data, 'refine', default=[{}])
    record['resolution'] = safe_get(refine, 0, 'ls_d_res_high')

    # ── R-work and R-free ─────────────────────────────────────────────────────
    record['r_work'] = safe_get(refine, 0, 'ls_R_factor_R_work')
    record['r_free'] = safe_get(refine, 0, 'ls_R_factor_R_free')

    # ── B-factor mean ─────────────────────────────────────────────────────────
    record['b_iso_mean'] = safe_get(refine, 0, 'B_iso_mean')

    # ── Rmerge ────────────────────────────────────────────────────────────────
    reflns = safe_get(entry_data, 'reflns', default=[{}])
    record['rmerge'] = safe_get(reflns, 0, 'pdbx_Rmerge_I_obs')

    # ── Atom counts ───────────────────────────────────────────────────────────
    refine_hist = safe_get(entry_data, 'refine_hist', default=[{}])
    record['num_atoms_protein'] = safe_get(refine_hist, 0,
                                           'pdbx_number_atoms_protein')
    record['num_atoms_solvent'] = safe_get(refine_hist, 0,
                                           'number_atoms_solvent')
    record['num_atoms_total']   = safe_get(refine_hist, 0,
                                           'number_atoms_total')

    # ── Crystal cell dimensions ───────────────────────────────────────────────
    cell = safe_get(entry_data, 'cell', default={})
    record['cell_length_a']    = safe_get(cell, 'length_a')
    record['cell_length_b']    = safe_get(cell, 'length_b')
    record['cell_length_c']    = safe_get(cell, 'length_c')
    record['cell_angle_alpha'] = safe_get(cell, 'angle_alpha')
    record['cell_angle_beta']  = safe_get(cell, 'angle_beta')
    record['cell_angle_gamma'] = safe_get(cell, 'angle_gamma')

    # ── Space group ───────────────────────────────────────────────────────────
    symmetry = safe_get(entry_data, 'symmetry', default={})
    record['space_group'] = safe_get(symmetry, 'space_group_name_H_M')

    # ── Solvent content and Matthews coefficient ──────────────────────────────
    exptl_crystal = safe_get(entry_data, 'exptl_crystal', default=[{}])
    record['solvent_content'] = safe_get(exptl_crystal, 0,
                                         'density_percent_sol')
    record['matthews_coeff']  = safe_get(exptl_crystal, 0,
                                         'density_Matthews')

    # ── Organism ──────────────────────────────────────────────────────────────
    if entity_data:
        src_list = safe_get(entity_data,
                            'rcsb_entity_source_organism', default=[])
        record['organism'] = safe_get(src_list, 0, 'scientific_name')

        # ── Protein sequence ──────────────────────────────────────────────────
        sequence = safe_get(entity_data, 'entity_poly',
                            'pdbx_seq_one_letter_code_can')
        record['protein_sequence'] = sequence

        # ── Amino acid composition ────────────────────────────────────────────
        aa_features = sequence_to_aa_composition(sequence)
        record.update(aa_features)
    else:
        record['organism']         = None
        record['protein_sequence'] = None
        record.update({f'aa_{aa}': None for aa in AMINO_ACIDS})

    return record


# ── MAIN FETCH FUNCTION ───────────────────────────────────────────────────────
def fetch_protein(protein_name, config):
    """
    Fetch all PDB structural features for a protein.
    Matches the exact column format of the mmCIF-extracted files.
    """
    output_path = os.path.join(BASE_PATH, config['output_file'])

    # Skip if already done
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        # Check if this is the old format (missing crystal features)
        has_cell = 'cell_length_a' in existing.columns
        if has_cell:
            print(f"\n{protein_name}: Already fetched with full features — "
                  f"{len(existing)} structures")
            print("  Delete the CSV to re-fetch.")
            return existing
        else:
            print(f"\n{protein_name}: Old format detected — re-fetching "
                  f"with full crystal features...")
            os.remove(output_path)

    print(f"\n{'='*60}")
    print(f"Fetching: {protein_name}")
    print(f"Disease : {config['description']}")
    print(f"UniProt : {config['uniprot_id']}")
    print(f"{'='*60}")

    # Step 1 — Get all PDB IDs
    print(f"\nStep 1: Searching for all {protein_name} structures...")
    pdb_ids = get_pdb_ids_for_uniprot(config['uniprot_id'])
    print(f"  Found {len(pdb_ids)} PDB entries")

    if not pdb_ids:
        print(f"  No structures found.")
        return None

    # Step 2 — Fetch features for each structure
    print(f"\nStep 2: Fetching features...")
    print(f"  Each structure = 2 API calls (entry + polymer entity)")
    est_mins = len(pdb_ids) * 0.7 / 60
    print(f"  Estimated time: ~{est_mins:.0f} minutes\n")

    records = []
    failed  = 0

    for i, pdb_id in enumerate(pdb_ids):

        # Call 1: Main entry data
        entry_data = fetch_entry(pdb_id)
        time.sleep(0.2)

        # Call 2: Polymer entity data (sequence + organism)
        entity_data = fetch_polymer_entity(pdb_id, entity_id=1)
        time.sleep(0.2)

        if entry_data:
            record = extract_all_features(
                pdb_id, protein_name, entry_data, entity_data
            )
            records.append(record)
            res = record.get('resolution', 'N/A')
            seq = 'YES' if record.get('protein_sequence') else 'NO'
        else:
            failed += 1
            res = 'FAILED'
            seq = 'NO'

        # Progress every 25 structures
        if (i + 1) % 25 == 0 or (i + 1) == len(pdb_ids):
            print(f"  [{i+1:4d}/{len(pdb_ids)}] {pdb_id} — "
                  f"res={res} seq={seq} | "
                  f"fetched={len(records)} failed={failed}")

    # Step 3 — Clean and save
    print(f"\nStep 3: Cleaning and saving...")

    df = pd.DataFrame(records)

    # Convert numeric columns
    numeric_cols = [
        'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
        'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
        'cell_length_a', 'cell_length_b', 'cell_length_c',
        'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
        'solvent_content', 'matthews_coeff',
    ] + [f'aa_{aa}' for aa in AMINO_ACIDS]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Reorder columns to match mmCIF files exactly + extra API columns
    col_order = [
        'pdb_id', 'protein', 'title', 'keywords', 'method',
        'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
        'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
        'cell_length_a', 'cell_length_b', 'cell_length_c',
        'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
        'space_group', 'solvent_content', 'matthews_coeff',
        'organism', 'protein_sequence',
    ] + [f'aa_{aa}' for aa in AMINO_ACIDS]

    # Only keep columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df        = df[col_order]

    df.to_csv(output_path, index=False)

    # Summary
    print(f"\n  Results for {protein_name}:")
    print(f"  {'─'*40}")
    print(f"  Structures saved    : {len(df)}")
    print(f"  Features per row    : {len(df.columns)}")
    print(f"  Failed fetches      : {failed}")

    res = df['resolution'].dropna()
    if len(res) > 0:
        print(f"  Avg resolution      : {res.mean():.2f} Å")
        print(f"  Pass ≤2.5Å          : "
              f"{(res<=2.5).sum()} ({(res<=2.5).sum()/len(res)*100:.0f}%)")

    seq_count = df['protein_sequence'].notna().sum()
    print(f"  Has sequence        : {seq_count}/{len(df)}")

    cell_count = df['cell_length_a'].notna().sum()
    print(f"  Has cell dims       : {cell_count}/{len(df)}")

    print(f"\n  Saved to: {config['output_file']}")
    return df


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("RCSB PDB API FETCH — FULL FEATURE VERSION")
    print("Proteins : HIV Protease + Thrombin")
    print("Features : Matches mmCIF file format exactly")
    print("=" * 60)
    print(f"Output   : {BASE_PATH}\n")
    print("Note: Old CSV files will be automatically replaced")
    print("      if they are missing crystal features.\n")

    results = {}
    for protein_name, config in PROTEINS.items():
        df = fetch_protein(protein_name, config)
        if df is not None:
            results[protein_name] = df

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — ALL 5 PROTEINS")
    print(f"{'='*60}")

    existing_files = {
        'EGFR' : 'EGFR_pdb_features.csv',
        'BACE1': 'BACE1_pdb_features.csv',
        'COX2' : 'COX2_pdb_features.csv',
    }

    all_proteins = {}
    for name, fname in existing_files.items():
        path = os.path.join(BASE_PATH, fname)
        if os.path.exists(path):
            all_proteins[name] = pd.read_csv(path)

    all_proteins.update(results)

    for name, df in all_proteins.items():
        res    = df['resolution'].dropna() if 'resolution' in df.columns \
                 else pd.Series()
        passed = (res <= 2.5).sum() if len(res) > 0 else 0
        print(f"\n  {name:15s}: {len(df):4d} structures | "
              f"{len(df.columns):2d} features | "
              f"avg res {res.mean():.2f}Å | "
              f"{passed/len(res)*100:.0f}% pass ≤2.5Å"
              if len(res) > 0 else
              f"\n  {name:15s}: {len(df):4d} structures | "
              f"{len(df.columns):2d} features")

    total = sum(len(df) for df in all_proteins.values())
    print(f"\n  Total structures across all 5 proteins: {total}")
    print(f"\nNext Step: 03_preprocessing.py")
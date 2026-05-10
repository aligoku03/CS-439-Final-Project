# =============================================================
# data_api_fetch.py
# CS-439 Final Project - Rutgers University
# =============================================================
# This script fetches all available PDB crystal structures for
# HIV Protease and Thrombin directly from the RCSB PDB REST
# API, since those proteins have too many entries to download
# manually like we did for EGFR, BACE1, and COX2.
# It extracts the same 24 structural features and saves one
# csv per protein to the project folder.
# =============================================================
# Libraries:
#   requests : sending HTTP requests to the RCSB PDB API
#   pandas   : organizing and saving the fetched data
#   time     : adding delays between API calls to avoid limits
#   os       : file and folder navigation
# =============================================================

import requests
import pandas as pd
import time
import os

# setting up the base path and proteins to fetch
BASE_PATH = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'

PROTEINS = {
    'HIV_Protease': {
        'uniprot_id' : 'P03367',
        'description': 'HIV-1 Protease - HIV/AIDS',
        'output_file': 'HIV_Protease_pdb_features.csv',
    },
    'Thrombin': {
        'uniprot_id' : 'P00734',
        'description': 'Thrombin - Blood Clotting',
        'output_file': 'Thrombin_pdb_features.csv',
    },
}

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


# this safely walks through nested dict and list keys without crashing
def safe_get(data, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            data = data[key] if len(data) > key else default
        else:
            return default
    return data


# this gets all pdb ids linked to a uniprot id using the rcsb search api
def get_pdb_ids_for_uniprot(uniprot_id):
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


# this fetches the main entry data for a single pdb id
def fetch_entry(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


# this fetches the polymer entity data which has the sequence and organism
def fetch_polymer_entity(pdb_id, entity_id=1):
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


# this converts a protein sequence into 20 amino acid frequency features
def sequence_to_aa_composition(sequence):
    if not sequence:
        return {f'aa_{aa}': None for aa in AMINO_ACIDS}
    sequence = str(sequence).upper()
    length   = len(sequence)
    if length == 0:
        return {f'aa_{aa}': None for aa in AMINO_ACIDS}
    return {f'aa_{aa}': round(sequence.count(aa) / length * 100, 4)
            for aa in AMINO_ACIDS}


# this pulls all the features we want from the api response and returns one row
def extract_all_features(pdb_id, protein_name, entry_data, entity_data):
    record = {
        'pdb_id' : pdb_id,
        'protein': protein_name,
    }

    if not entry_data:
        return record

    # getting basic info like title and keywords
    record['title']    = safe_get(entry_data, 'struct', 'title')
    record['keywords'] = safe_get(entry_data, 'struct_keywords',
                                  'pdbx_keywords')

    # getting the experimental method
    exptl = safe_get(entry_data, 'exptl', default=[{}])
    record['method'] = safe_get(exptl, 0, 'method')

    # getting the resolution in angstroms
    refine = safe_get(entry_data, 'refine', default=[{}])
    record['resolution'] = safe_get(refine, 0, 'ls_d_res_high')

    # getting r-work and r-free which measure how well the model fits the data
    record['r_work'] = safe_get(refine, 0, 'ls_R_factor_R_work')
    record['r_free'] = safe_get(refine, 0, 'ls_R_factor_R_free')

    # getting the b-factor mean which measures atomic flexibility
    record['b_iso_mean'] = safe_get(refine, 0, 'B_iso_mean')

    # getting rmerge which is a diffraction data quality metric
    reflns = safe_get(entry_data, 'reflns', default=[{}])
    record['rmerge'] = safe_get(reflns, 0, 'pdbx_Rmerge_I_obs')

    # getting atom counts for protein, solvent, and total
    refine_hist = safe_get(entry_data, 'refine_hist', default=[{}])
    record['num_atoms_protein'] = safe_get(refine_hist, 0,
                                           'pdbx_number_atoms_protein')
    record['num_atoms_solvent'] = safe_get(refine_hist, 0,
                                           'number_atoms_solvent')
    record['num_atoms_total']   = safe_get(refine_hist, 0,
                                           'number_atoms_total')

    # getting the crystal cell dimensions and angles
    cell = safe_get(entry_data, 'cell', default={})
    record['cell_length_a']    = safe_get(cell, 'length_a')
    record['cell_length_b']    = safe_get(cell, 'length_b')
    record['cell_length_c']    = safe_get(cell, 'length_c')
    record['cell_angle_alpha'] = safe_get(cell, 'angle_alpha')
    record['cell_angle_beta']  = safe_get(cell, 'angle_beta')
    record['cell_angle_gamma'] = safe_get(cell, 'angle_gamma')

    # getting the space group which describes the crystal symmetry
    symmetry = safe_get(entry_data, 'symmetry', default={})
    record['space_group'] = safe_get(symmetry, 'space_group_name_H_M')

    # getting solvent content and matthews coefficient
    exptl_crystal = safe_get(entry_data, 'exptl_crystal', default=[{}])
    record['solvent_content'] = safe_get(exptl_crystal, 0,
                                         'density_percent_sol')
    record['matthews_coeff']  = safe_get(exptl_crystal, 0,
                                         'density_Matthews')

    # getting the organism and sequence from the polymer entity response
    if entity_data:
        src_list = safe_get(entity_data,
                            'rcsb_entity_source_organism', default=[])
        record['organism'] = safe_get(src_list, 0, 'scientific_name')

        # getting the canonical one-letter protein sequence
        sequence = safe_get(entity_data, 'entity_poly',
                            'pdbx_seq_one_letter_code_can')
        record['protein_sequence'] = sequence

        # computing the 20 amino acid composition features
        aa_features = sequence_to_aa_composition(sequence)
        record.update(aa_features)
    else:
        record['organism']         = None
        record['protein_sequence'] = None
        record.update({f'aa_{aa}': None for aa in AMINO_ACIDS})

    return record


# this fetches all pdb structures for one protein and saves them to csv
def fetch_protein(protein_name, config):
    output_path = os.path.join(BASE_PATH, 'preprocessed_results', config['output_file'])

    # skipping if we already fetched this one
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        # checking if the file has the new crystal feature columns
        has_cell = 'cell_length_a' in existing.columns
        if has_cell:
            print(f"\n{protein_name}: Already fetched with full features - "
                  f"{len(existing)} structures")
            print("  Delete the CSV to re-fetch.")
            return existing
        else:
            print(f"\n{protein_name}: Old format detected - re-fetching "
                  f"with full crystal features...")
            os.remove(output_path)

    print(f"\n{'='*60}")
    print(f"Fetching: {protein_name}")
    print(f"Disease : {config['description']}")
    print(f"UniProt : {config['uniprot_id']}")
    print(f"{'='*60}")

    # getting the list of all pdb ids for this protein
    print(f"\nStep 1: Searching for all {protein_name} structures...")
    pdb_ids = get_pdb_ids_for_uniprot(config['uniprot_id'])
    print(f"  Found {len(pdb_ids)} PDB entries")

    if not pdb_ids:
        print(f"  No structures found.")
        return None

    # fetching features for each structure
    print(f"\nStep 2: Fetching features...")
    print(f"  Each structure = 2 API calls (entry + polymer entity)")
    est_mins = len(pdb_ids) * 0.7 / 60
    print(f"  Estimated time: ~{est_mins:.0f} minutes\n")

    records = []
    failed  = 0

    for i, pdb_id in enumerate(pdb_ids):

        # first call gets the main entry data
        entry_data = fetch_entry(pdb_id)
        time.sleep(0.2)

        # second call gets the polymer entity which has the sequence
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

        # printing progress every 25 structures
        if (i + 1) % 25 == 0 or (i + 1) == len(pdb_ids):
            print(f"  [{i+1:4d}/{len(pdb_ids)}] {pdb_id} - "
                  f"res={res} seq={seq} | "
                  f"fetched={len(records)} failed={failed}")

    # cleaning everything up and saving to csv
    print(f"\nStep 3: Cleaning and saving...")

    df = pd.DataFrame(records)

    # converting all numeric columns from strings to actual numbers
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

    # putting columns in a consistent order so all proteins match
    col_order = [
        'pdb_id', 'protein', 'title', 'keywords', 'method',
        'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
        'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
        'cell_length_a', 'cell_length_b', 'cell_length_c',
        'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
        'space_group', 'solvent_content', 'matthews_coeff',
        'organism', 'protein_sequence',
    ] + [f'aa_{aa}' for aa in AMINO_ACIDS]

    # only keeping columns that actually exist in the dataframe
    col_order = [c for c in col_order if c in df.columns]
    df        = df[col_order]

    df.to_csv(output_path, index=False)

    # printing a summary of what we got
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


if __name__ == '__main__':
    print("=" * 60)
    print("RCSB PDB API FETCH - FULL FEATURE VERSION")
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

    # loading the existing csv files for the other 3 proteins
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - ALL 5 PROTEINS")
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
    
import os
import pandas as pd
import numpy as np

# setting up the paths and protein folders to process
main_folder = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
proteins = {
    'EGFR'  : 'EGFR_dataset',
    'BACE1' : 'BACE1_dataset',
    'COX2'  : 'COX2_dataset',
}
output_csv     = os.path.join(main_folder, 'pdb_features.csv')
output_folder  = main_folder


# this parses the protein sequence out of a cif file's lines
# handles both single-line and multiline (semicolon-delimited) formats
def parse_sequence(lines):
    seq_lines = []
    in_seq    = False

    for i, line in enumerate(lines):
        # the trigger line we are looking for
        if '_entity_poly.pdbx_seq_one_letter_code_can' in line:
            rest = line.split('pdbx_seq_one_letter_code_can', 1)[-1].strip()

            # if the value is on the same line we just return it directly
            if rest and not rest.startswith(';') and not rest.startswith('_') \
                    and rest not in ('', '?', '.'):
                return rest.strip("'").replace('\n', '').strip()

            in_seq = True
            continue

        if in_seq:
            stripped = line.strip()

            # the sequence block is over once we hit a new field
            if stripped.startswith('_') or stripped.startswith('loop_') \
                    or stripped.startswith('#'):
                break

            # semicolons mark the start and end of the multiline value
            if stripped == ';':
                if seq_lines:   # closing semicolon means we are done
                    break
                continue        # opening semicolon means start collecting

            # skipping any blank lines before the content begins
            if not stripped:
                continue

            seq_lines.append(stripped)

    if seq_lines:
        return ''.join(seq_lines).strip("'").strip()
    return None


# this extracts all the features we want from a single cif file
def extract_features(filepath, pdb_id, protein_name):

    # starting with all fields set to none and filling them in as we find them
    record = {
        'pdb_id'            : pdb_id,
        'protein'           : protein_name,
        'title'             : None,
        'keywords'          : None,
        'method'            : None,
        'resolution'        : None,
        'r_work'            : None,
        'r_free'            : None,
        'b_iso_mean'        : None,
        'rmerge'            : None,
        'num_atoms_protein' : None,
        'num_atoms_solvent' : None,
        'num_atoms_total'   : None,
        'cell_length_a'     : None,
        'cell_length_b'     : None,
        'cell_length_c'     : None,
        'cell_angle_alpha'  : None,
        'cell_angle_beta'   : None,
        'cell_angle_gamma'  : None,
        'space_group'       : None,
        'solvent_content'   : None,
        'matthews_coeff'    : None,
        'organism'          : None,
        'protein_sequence'  : None,
    }

    try:
        with open(filepath, 'r', errors='ignore') as f:
            lines = f.readlines()

        # parsing the protein sequence using the helper above
        record['protein_sequence'] = parse_sequence(lines)

        # going through each line and pulling out the fields we care about
        for line in lines:
            l = line.strip()
            if not l or l.startswith('#'):
                continue

            # title of the structure
            if '_struct.title' in l and len(l.split()) > 1:
                record['title'] = l.split('title', 1)[-1].strip().strip("'")

            # keywords describing the structure
            elif '_struct_keywords.pdbx_keywords' in l and len(l.split()) > 1:
                record['keywords'] = l.split('keywords', 1)[-1].strip().strip("'")

            # experimental method, handles both old and new cif formats
            elif '_exptl.method' in l and 'crystals' not in l:
                val = l.split('method', 1)[-1].strip().strip("'")
                if val and '_details' not in val and val not in ['.', '?', '']:
                    record['method'] = val
                elif record['method'] is None:
                    # try to infer from the refine line below
                    pass

            # if method is still missing we try to infer it from refine id
            elif '_refine.pdbx_refine_id' in l:
                val = l.split('pdbx_refine_id', 1)[-1].strip().strip("'")
                if val and val not in ['.', '?', ''] and record['method'] is None:
                    record['method'] = val

            # resolution in angstroms
            elif '_refine.ls_d_res_high' in l and 'error' not in l and 'low' not in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['resolution'] = parts[-1]

            # r-work measures how well the model fits the data
            elif '_refine.ls_R_factor_R_work' in l and 'free' not in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['r_work'] = parts[-1]

            # r-free is the same but on held-out reflections
            elif '_refine.ls_R_factor_R_free' in l \
                    and 'error' not in l and 'details' not in l \
                    and 'percent' not in l and 'number' not in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['r_free'] = parts[-1]

            # b-factor mean measures atomic flexibility
            elif '_refine.B_iso_mean' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['b_iso_mean'] = parts[-1]

            # rmerge is a diffraction data quality metric
            elif '_reflns.pdbx_Rmerge_I_obs' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['rmerge'] = parts[-1]

            # atom counts for protein, solvent, and total
            elif '_refine_hist.pdbx_number_atoms_protein' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['num_atoms_protein'] = parts[-1]

            elif '_refine_hist.number_atoms_solvent' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['num_atoms_solvent'] = parts[-1]

            elif '_refine_hist.number_atoms_total' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['num_atoms_total'] = parts[-1]

            # crystal cell dimensions (a, b, c lengths)
            elif '_cell.length_a ' in l:
                parts = l.split()
                if len(parts) > 1: record['cell_length_a'] = parts[-1]

            elif '_cell.length_b ' in l:
                parts = l.split()
                if len(parts) > 1: record['cell_length_b'] = parts[-1]

            elif '_cell.length_c ' in l:
                parts = l.split()
                if len(parts) > 1: record['cell_length_c'] = parts[-1]

            # crystal cell angles (alpha, beta, gamma)
            elif '_cell.angle_alpha ' in l:
                parts = l.split()
                if len(parts) > 1: record['cell_angle_alpha'] = parts[-1]

            elif '_cell.angle_beta ' in l:
                parts = l.split()
                if len(parts) > 1: record['cell_angle_beta'] = parts[-1]

            elif '_cell.angle_gamma ' in l:
                parts = l.split()
                if len(parts) > 1: record['cell_angle_gamma'] = parts[-1]

            # space group describes the crystal symmetry
            elif '_symmetry.space_group_name_H-M' in l:
                record['space_group'] = l.split('H-M', 1)[-1].strip().strip("'")

            # solvent content and matthews coefficient describe crystal packing
            elif '_exptl_crystal.density_percent_sol' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['solvent_content'] = parts[-1]

            elif '_exptl_crystal.density_Matthews' in l:
                parts = l.split()
                if len(parts) > 1 and parts[-1] not in ['.', '?']:
                    record['matthews_coeff'] = parts[-1]

            # organism the protein came from
            elif '_entity_src_gen.pdbx_gene_src_scientific_name' in l:
                val = l.split('name', 1)[-1].strip().strip("'")
                if val and val not in ['.', '?', '']:
                    record['organism'] = val

    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")

    return record


# step 1 is walking through every cif file and extracting features
all_records = []

for protein_name, dataset_folder in proteins.items():
    protein_folder = os.path.join(main_folder, dataset_folder)
    print(f"\n{'='*50}")
    print(f"Processing {protein_name}")
    print(f"{'='*50}")

    count = 0
    for root, dirs, files in os.walk(protein_folder):
        for filename in files:
            if filename.endswith('.cif'):
                filepath = os.path.join(root, filename)
                pdb_id   = filename.replace('.cif', '')
                record   = extract_features(filepath, pdb_id, protein_name)
                all_records.append(record)
                count += 1
                print(f"  [{count}] {pdb_id} - "
                      f"resolution: {record['resolution']}, "
                      f"seq: {'YES' if record['protein_sequence'] else 'MISSING'}")

    print(f"\n  Done {protein_name}: {count} structures")

df = pd.DataFrame(all_records)
df.to_csv(output_csv, index=False)
print(f"\nRaw features saved: {output_csv}")


# step 2 is cleaning up the missing values
print(f"\n{'='*50}")
print("CLEANING MISSING VALUES")
print(f"{'='*50}")

# converting numeric columns from strings to actual numbers
numeric_cols = [
    'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
    'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
    'cell_length_a', 'cell_length_b', 'cell_length_c',
    'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma',
    'solvent_content', 'matthews_coeff'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# this fixes missing methods by inferring from resolution
# structures with resolution are xray, ones without are nmr
def fix_method(row):
    if pd.notna(row['method']) and row['method'] not in ['.', '?', '']:
        return row['method']
    if pd.notna(row['resolution']):
        return 'X-RAY DIFFRACTION'
    return 'SOLUTION NMR'

df['method'] = df.apply(fix_method, axis=1)

# filling missing organism values with unknown
df['organism'] = df['organism'].fillna('Unknown')

# filling missing numeric values with the median for that protein
for col in numeric_cols:
    df[col] = df.groupby('protein')[col].transform(
        lambda x: x.fillna(x.median())
    )

# checking how many sequences are still missing per protein
print("\nMissing sequences per protein:")
for protein in ['EGFR', 'BACE1', 'COX2']:
    sub = df[df['protein'] == protein]
    missing = sub['protein_sequence'].isna().sum()
    print(f"  {protein}: {missing}/{len(sub)} missing")

# dropping any rows that still don't have a sequence
before = len(df)
df = df.dropna(subset=['protein_sequence'])
after  = len(df)
print(f"\nDropped {before - after} rows with missing protein sequences")
print(f"Remaining: {after} structures")

# final report on remaining missing values
print("\nFinal missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# step 3 is splitting the combined dataframe into 3 separate csvs
print(f"\n{'='*50}")
print("SAVING 3 SEPARATE CSV FILES")
print(f"{'='*50}")

for protein in ['EGFR', 'BACE1', 'COX2']:
    protein_df   = df[df['protein'] == protein].copy().reset_index(drop=True)
    output_path  = os.path.join(output_folder, f'{protein}_pdb_features.csv')
    protein_df.to_csv(output_path, index=False)

    print(f"\n{protein}:")
    print(f"  Structures : {len(protein_df)}")
    print(f"  Features   : {len(protein_df.columns)}")
    print(f"  Saved to   : {output_path}")
    print(f"  Missing    : {protein_df.isnull().sum().sum()} total NaN values")
    print(f"  Methods    : {protein_df['method'].value_counts().to_dict()}")

print(f"\n{'='*50}")
print("ALL DONE!")
print(f"{'='*50}")
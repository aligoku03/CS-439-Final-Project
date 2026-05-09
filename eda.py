import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
import warnings
import os
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_PATH         = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
FIG_PATH          = os.path.join(BASE_PATH, 'results', 'figures')
RESOLUTION_CUTOFF = 2.5    # Scientific Reports 2021: 2.0–2.5Å for drug discovery

PROTEINS = ['EGFR', 'BACE1', 'COX2', 'HIV_Protease', 'Thrombin']

COLORS = {
    'EGFR'        : '#2196F3',  # blue
    'BACE1'       : '#4CAF50',  # green
    'COX2'        : '#FF5722',  # orange-red
    'HIV_Protease': '#9C27B0',  # purple
    'Thrombin'    : '#FF9800',  # amber
}

DISEASE = {
    'EGFR'        : 'Cancer',
    'BACE1'       : "Alzheimer's",
    'COX2'        : 'Inflammation',
    'HIV_Protease': 'HIV/AIDS',
    'Thrombin'    : 'Blood Clotting',
}

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_NAMES    = {
    'A':'Ala','C':'Cys','D':'Asp','E':'Glu','F':'Phe',
    'G':'Gly','H':'His','I':'Ile','K':'Lys','L':'Leu',
    'M':'Met','N':'Asn','P':'Pro','Q':'Gln','R':'Arg',
    'S':'Ser','T':'Thr','V':'Val','W':'Trp','Y':'Tyr'
}
os.makedirs(FIG_PATH, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── HELPERS ───────────────────────────────────────────────────────────────────
def save(filename):
    try:
        plt.savefig(os.path.join(FIG_PATH, filename), dpi=150)
    except Exception:
        plt.savefig(os.path.join(FIG_PATH, filename), dpi=100)
    plt.close()
    print(f'  Saved: {filename}')

def get_aa_composition(df):
    compositions = []
    for seq in df['protein_sequence'].dropna():
        seq    = str(seq).upper()
        length = len(seq)
        if length > 0:
            comp = {aa: seq.count(aa) / length * 100 for aa in AMINO_ACIDS}
            compositions.append(comp)
    if not compositions:
        return pd.Series({aa: 0 for aa in AMINO_ACIDS})
    return pd.DataFrame(compositions).mean()

def label(protein):
    return f'{protein}\n({DISEASE[protein]})'


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING ALL 5 PROTEIN DATASETS")
print("=" * 60)

dfs = {}
for protein in PROTEINS:
    path = os.path.join(BASE_PATH, f'{protein}_pdb_features.csv')
    if os.path.exists(path):
        dfs[protein] = pd.read_csv(path)
        dfs[protein]['organism'] = dfs[protein]['organism'].str.title()
        print(f'  {protein:15s}: {len(dfs[protein]):4d} structures, '
              f'{len(dfs[protein].columns)} features')
    else:
        print(f'  WARNING: {protein}_pdb_features.csv not found — skipping')

df_all = pd.concat(dfs.values(), ignore_index=True)
loaded = list(dfs.keys())
print(f'\n  Total: {len(df_all)} structures across {len(dfs)} proteins')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 1: DATASET OVERVIEW")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Structure counts
counts   = [len(dfs[p]) for p in loaded]
x_labels = [label(p) for p in loaded]
colors   = [COLORS[p] for p in loaded]

bars = axes[0].bar(x_labels, counts, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 8, str(count),
                 ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0].set_title('PDB Structures per Protein', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Number of Structures')
axes[0].set_ylim(0, max(counts) * 1.2)
axes[0].tick_params(axis='x', labelsize=8)

# Experimental method breakdown
method_data = df_all.groupby(['protein', 'method']).size().unstack(fill_value=0)
method_colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
method_data.plot(kind='bar', ax=axes[1], color=method_colors[:len(method_data.columns)],
                 edgecolor='white', linewidth=1)
axes[1].set_title('Experimental Methods per Protein', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Count')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=30, labelsize=8)
axes[1].legend(title='Method', fontsize=7)

# Average sequence length
seq_lengths = [dfs[p]['protein_sequence'].dropna().apply(len).mean()
               for p in loaded]
bars = axes[2].bar(x_labels, seq_lengths, color=colors,
                    edgecolor='white', linewidth=1.5)
for bar, length in zip(bars, seq_lengths):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5, f'{length:.0f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[2].set_title('Average Sequence Length', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Amino Acids')
axes[2].set_ylim(0, max(seq_lengths) * 1.2)
axes[2].tick_params(axis='x', labelsize=8)

save('01_dataset_overview.png')

for p in loaded:
    res = dfs[p]['resolution'].dropna()
    print(f'  {p:15s}: {len(dfs[p])} structures | '
          f'avg res {res.mean():.2f}Å | '
          f'seq {dfs[p]["protein_sequence"].notna().sum()}/{len(dfs[p])}')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — RESOLUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 2: RESOLUTION ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
for protein in loaded:
    res = dfs[protein]['resolution'].dropna()
    axes[0].hist(res, bins=25, alpha=0.5, color=COLORS[protein],
                 label=f'{protein} ({res.mean():.2f}Å)', edgecolor='white')
axes[0].axvline(x=RESOLUTION_CUTOFF, color='red', linestyle='--',
                linewidth=2, label=f'{RESOLUTION_CUTOFF}Å cutoff\n(Scientific Reports 2021)')
axes[0].set_xlabel('Resolution (Å)')
axes[0].set_ylabel('Count')
axes[0].set_title('Resolution Distribution\n(lower = higher quality structure)',
                  fontweight='bold')
axes[0].legend(fontsize=7)

# Boxplot
res_data = [dfs[p]['resolution'].dropna().values for p in loaded]
bp = axes[1].boxplot(res_data, labels=[label(p) for p in loaded],
                     patch_artist=True)
for patch, protein in zip(bp['boxes'], loaded):
    patch.set_facecolor(COLORS[protein])
    patch.set_alpha(0.7)
axes[1].axhline(y=RESOLUTION_CUTOFF, color='red', linestyle='--',
                linewidth=2, label=f'{RESOLUTION_CUTOFF}Å cutoff')
axes[1].set_ylabel('Resolution (Å)')
axes[1].set_title('Resolution Comparison\n(HIV Protease has the highest quality structures)',
                  fontweight='bold')
axes[1].tick_params(axis='x', labelsize=8)
axes[1].legend(fontsize=9)

# Annotate pass rates
for idx, protein in enumerate(loaded, start=1):
    res    = dfs[protein]['resolution'].dropna()
    passed = (res <= RESOLUTION_CUTOFF).sum()
    axes[1].text(idx, res.max() + 0.05, f'{passed/len(res)*100:.0f}%',
                 ha='center', fontsize=8, color=COLORS[protein], fontweight='bold')

save('02_resolution_analysis.png')

print(f'\n  Resolution filter (≤{RESOLUTION_CUTOFF}Å):')
for protein in loaded:
    res    = dfs[protein]['resolution'].dropna()
    passed = (res <= RESOLUTION_CUTOFF).sum()
    print(f'  {protein:15s}: {passed}/{len(res)} ({passed/len(res)*100:.0f}%) pass')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 3: CORRELATION HEATMAP")
print("=" * 60)

numeric_cols = [
    'resolution', 'r_work', 'r_free', 'b_iso_mean', 'rmerge',
    'num_atoms_protein', 'num_atoms_solvent', 'num_atoms_total',
    'cell_length_a', 'cell_length_b', 'cell_length_c',
    'solvent_content', 'matthews_coeff'
]

# Show heatmap for all 5 proteins — 2 rows x 3 cols (last spot empty)
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes_flat = axes.flatten()

for i, protein in enumerate(loaded):
    ax   = axes_flat[i]
    corr = dfs[protein][numeric_cols].corr()
    im   = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels([c.replace('_', '\n') for c in numeric_cols],
                       fontsize=5, rotation=45, ha='right')
    ax.set_yticklabels([c.replace('_', ' ') for c in numeric_cols], fontsize=5)
    ax.set_title(f'{protein} ({DISEASE[protein]})\nFeature Correlations',
                 fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

# Hide unused subplot
if len(loaded) < 6:
    for j in range(len(loaded), 6):
        axes_flat[j].set_visible(False)

save('03_correlation_heatmap.png')
print('  Done.')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — AMINO ACID COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 4: AMINO ACID COMPOSITION")
print("=" * 60)

comp_data = {p: get_aa_composition(dfs[p]) for p in loaded}
comp_df   = pd.DataFrame(comp_data)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Heatmap
im = axes[0].imshow(comp_df.values, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(len(loaded)))
axes[0].set_yticks(range(20))
axes[0].set_xticklabels([label(p) for p in loaded], fontsize=9, fontweight='bold')
axes[0].set_yticklabels([f'{aa} ({AA_NAMES[aa]})' for aa in AMINO_ACIDS], fontsize=8)
axes[0].set_title('Amino Acid Composition per Protein (%)',
                  fontweight='bold', fontsize=13)
plt.colorbar(im, ax=axes[0], label='Frequency (%)')
for i in range(20):
    for j in range(len(loaded)):
        val = comp_df.values[i, j]
        axes[0].text(j, i, f'{val:.1f}', ha='center', va='center',
                     fontsize=6, color='black' if val < 7 else 'white')

# Top 5 bar chart
width = 0.15
x     = np.arange(5)
for idx, protein in enumerate(loaded):
    top5 = comp_df[protein].nlargest(5)
    axes[1].bar(x + idx * width, top5.values, width,
                label=protein, color=COLORS[protein],
                alpha=0.85, edgecolor='white')

axes[1].set_xticks(x + width * (len(loaded) - 1) / 2)
top5_labels = [f'{aa} ({AA_NAMES[aa]})'
               for aa in comp_df[loaded[0]].nlargest(5).index]
axes[1].set_xticklabels(top5_labels, rotation=15, ha='right', fontsize=9)
axes[1].set_title('Top 5 Most Frequent Amino Acids per Protein',
                  fontweight='bold', fontsize=13)
axes[1].set_ylabel('Frequency (%)')
axes[1].legend(title='Protein', fontsize=8)

save('04_amino_acid_composition.png')

print('\n  Top 5 amino acids per protein:')
for protein in loaded:
    top5 = comp_df[protein].nlargest(5)
    print(f'  {protein:15s}: '
          f'{", ".join([f"{aa}({v:.1f}%)" for aa, v in top5.items()])}')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — R-FREE VS RESOLUTION SCATTER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 5: R-FREE VS RESOLUTION SCATTER")
print("=" * 60)

# 2 rows x 3 cols for 5 proteins
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes_flat = axes.flatten()

for i, protein in enumerate(loaded):
    ax = axes_flat[i]

    xray_mask = dfs[protein]['method'].str.contains('X-ray|X-RAY', case=False, na=False)
    df_xray   = dfs[protein][xray_mask].dropna(subset=['resolution', 'r_free'])

    if len(df_xray) > 0:
        scatter = ax.scatter(
            df_xray['resolution'], df_xray['r_free'],
            c=df_xray['b_iso_mean'], cmap='coolwarm',
            alpha=0.6, s=25, edgecolors='none'
        )
        plt.colorbar(scatter, ax=ax, label='B-factor', fraction=0.046)

        if len(df_xray) > 5:
            z      = np.polyfit(df_xray['resolution'], df_xray['r_free'], 1)
            x_line = np.linspace(df_xray['resolution'].min(),
                                 df_xray['resolution'].max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color='black',
                    linewidth=1.5, alpha=0.7)

        corr = df_xray['resolution'].corr(df_xray['r_free'])
        ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.axvline(x=RESOLUTION_CUTOFF, color='green', linestyle='--',
               alpha=0.7, linewidth=1.2)
    ax.axhline(y=0.25, color='red', linestyle='--',
               alpha=0.7, linewidth=1.2)
    ax.set_xlabel('Resolution (Å)', fontsize=9)
    ax.set_ylabel('R-free', fontsize=9)
    ax.set_title(f'{protein} ({DISEASE[protein]})\nn={len(df_xray)} X-ray structures',
                 fontweight='bold', fontsize=10)

# Hide unused subplot
for j in range(len(loaded), 6):
    axes_flat[j].set_visible(False)

save('05_rfree_vs_resolution.png')

print('\n  Pearson r (R-free vs Resolution):')
for protein in loaded:
    xray_mask = dfs[protein]['method'].str.contains('X-ray|X-RAY', case=False, na=False)
    df_xray   = dfs[protein][xray_mask].dropna(subset=['resolution', 'r_free'])
    if len(df_xray) > 5:
        corr = df_xray['resolution'].corr(df_xray['r_free'])
        print(f'  {protein:15s}: r = {corr:.3f}')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — B-FACTOR VIOLIN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PLOT 6: B-FACTOR VIOLIN")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Violin plot
b_data = [dfs[p]['b_iso_mean'].dropna().values for p in loaded]
parts  = axes[0].violinplot(b_data,
                             positions=range(1, len(loaded) + 1),
                             showmedians=True, showextrema=True)
for pc, protein in zip(parts['bodies'], loaded):
    pc.set_facecolor(COLORS[protein])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)
axes[0].set_xticks(range(1, len(loaded) + 1))
axes[0].set_xticklabels([label(p) for p in loaded], fontsize=8)
axes[0].set_ylabel('Mean B-factor (Å²)', fontsize=11)
axes[0].set_title('B-factor Distribution per Protein\n'
                  '(higher = more atomic flexibility)',
                  fontweight='bold', fontsize=12)
axes[0].axhline(y=30, color='gray', linestyle='--',
                alpha=0.5, label='B=30 reference')
axes[0].legend(fontsize=9)

# B-factor vs resolution scatter
for protein in loaded:
    xray_mask = dfs[protein]['method'].str.contains('X-ray|X-RAY', case=False, na=False)
    df_xray   = dfs[protein][xray_mask].dropna(subset=['resolution', 'b_iso_mean'])
    axes[1].scatter(df_xray['resolution'], df_xray['b_iso_mean'],
                    color=COLORS[protein], alpha=0.35, s=15,
                    edgecolors='none', label=protein)

axes[1].set_xlabel('Resolution (Å)', fontsize=11)
axes[1].set_ylabel('Mean B-factor (Å²)', fontsize=11)
axes[1].set_title('B-factor vs Resolution\n'
                  '(higher resolution → lower B-factor expected)',
                  fontweight='bold', fontsize=12)
axes[1].legend(title='Protein', fontsize=8)
axes[1].axvline(x=RESOLUTION_CUTOFF, color='red', linestyle='--',
                alpha=0.6, label=f'{RESOLUTION_CUTOFF}Å')

save('06_bfactor_analysis.png')

print('\n  B-factor stats per protein:')
for protein in loaded:
    b = dfs[protein]['b_iso_mean'].dropna()
    print(f'  {protein:15s}: mean={b.mean():.1f} | '
          f'median={b.median():.1f} | std={b.std():.1f}')


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA COMPLETE — SUMMARY")
print("=" * 60)

for protein in loaded:
    df     = dfs[protein]
    res    = df['resolution'].dropna()
    passed = (res <= RESOLUTION_CUTOFF).sum()
    seq    = df['protein_sequence'].dropna().apply(len)
    print(f'\n  {protein} ({DISEASE[protein]}):')
    print(f'    Structures       : {len(df)}')
    print(f'    Avg resolution   : {res.mean():.2f} Å')
    print(f'    Pass ≤{RESOLUTION_CUTOFF}Å       : '
          f'{passed}/{len(res)} ({passed/len(res)*100:.0f}%)')
    print(f'    Avg seq length   : {seq.mean():.0f} amino acids')

print(f'\n  Total structures : {len(df_all)}')
print(f'  Features per row : {len(list(dfs.values())[0].columns)}')
print(f'\n  6 figures saved to: {FIG_PATH}')
for i, name in enumerate([
    'dataset_overview', 'resolution_analysis', 'correlation_heatmap',
    'amino_acid_composition', 'rfree_vs_resolution', 'bfactor_analysis'
], start=1):
    print(f'    {i:02d}_{name}.png')
print('\n  Next Step: 03_preprocessing.py')
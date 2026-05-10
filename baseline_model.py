# =============================================================
# baseline_model.py
# CS-439 Final Project — Rutgers University
# =============================================================
# This script trains and evaluates three classical machine learning
# models (Logistic Regression, Random Forest, XGBoost) for each of
# the three target proteins. It handles class imbalance using class
# weighting and undersampling for HIV Protease, applies optimal
# decision thresholds from ROC curves, and saves ROC curves,
# confusion matrices, and a full comparison figure to results/Baseline/.
# =============================================================
# Libraries:
#   os           : file and folder management
#   numpy        : numeric array operations
#   pandas       : loading processed data and saving results
#   matplotlib   : plotting ROC curves and confusion matrices
#   sklearn      : logistic regression, random forest, metrics, scaling
#   xgboost      : gradient boosting classifier
# =============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve,
    classification_report
)
from sklearn.preprocessing      import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# trying to import xgboost, skipping that model if it isn't installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

# setting random seed for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# setting up the paths and configuration
BASE_PATH = r'C:\Users\aligo\OneDrive\Desktop\Protein_Machine_Learning'
DATA_PATH = os.path.join(BASE_PATH, 'processed_data')
RES_PATH  = os.path.join(BASE_PATH, 'results')
FIG_PATH  = os.path.join(BASE_PATH, 'results', 'Baseline')

os.makedirs(RES_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# the proteins we are training models for and their associated diseases
PROTEINS = ['EGFR', 'BACE1', 'HIV_Protease']
DISEASE  = {
    'EGFR'        : 'Cancer',
    'BACE1'       : "Alzheimer's",
    'HIV_Protease': 'HIV/AIDS',
}

# colors used for each model in the plots
COLORS = {
    'Logistic Regression': '#2196F3',
    'Random Forest'      : '#4CAF50',
    'XGBoost'            : '#FF5722',
}

# per-protein config — controls undersampling and threshold strategy
# balanced datasets use 0.5, imbalanced use optimal threshold from roc curve
PROTEIN_CONFIG = {
    'EGFR': {
        'undersample'      : False,
        'use_optimal_thresh': True,   # 19.8% binders - imbalanced
    },
    'BACE1': {
        'undersample'      : False,
        'use_optimal_thresh': False,  # 45.7% binders - balanced, 0.5 is fine
    },
    'HIV_Protease': {
        'undersample'       : True,
        'undersample_ratio' : 5,      # 5 non-binders per binder
        'use_optimal_thresh': True,   # still imbalanced even after sampling
    },
}


# this loads the train and test csvs for a protein
def load_data(protein_name):
    train_path = os.path.join(DATA_PATH, f'{protein_name}_train.csv')
    test_path  = os.path.join(DATA_PATH, f'{protein_name}_test.csv')

    if not os.path.exists(train_path):
        print(f'  WARNING: {protein_name}_train.csv not found - skipping')
        return None, None, None, None

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # dropping non-feature columns to get just the feature matrix
    drop_cols = ['label', 'smiles', 'protein']
    X_train   = train.drop(columns=[c for c in drop_cols if c in train.columns])
    y_train   = train['label']
    X_test    = test.drop(columns=[c for c in drop_cols if c in test.columns])
    y_test    = test['label']

    return X_train, y_train, X_test, y_test


# this undersamples the majority class to fix severe class imbalance
# only applied to HIV protease since it has 96.5% non-binders
def undersample(X_train, y_train, ratio=5):
    # separating binders and non-binders
    binder_idx     = np.where(y_train == 1)[0]
    non_binder_idx = np.where(y_train == 0)[0]

    # randomly sampling ratio * num_binders non-binders
    n_sample           = min(len(binder_idx) * ratio, len(non_binder_idx))
    sampled_non_binder = np.random.choice(non_binder_idx, n_sample, replace=False)

    # combining and shuffling
    keep_idx   = np.concatenate([binder_idx, sampled_non_binder])
    np.random.shuffle(keep_idx)

    X_resampled = X_train.iloc[keep_idx]
    y_resampled = y_train.iloc[keep_idx]

    print(f'  After undersampling: {len(y_resampled)} samples '
          f'({y_resampled.sum()} binders, '
          f'{(y_resampled == 0).sum()} non-binders)')

    return X_resampled, y_resampled


# this computes balanced class weights for handling imbalanced data
def get_class_weight(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


# this finds the best classification threshold using the roc curve
# handles edge cases where threshold could be inf or nan
def get_optimal_threshold(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    optimal_idx       = np.argmax(tpr - fpr)
    threshold         = thresholds[optimal_idx]

    # falling back to 0.5 if threshold is inf or nan
    if np.isinf(threshold) or np.isnan(threshold):
        return 0.5
    return threshold


# this computes all the evaluation metrics for one model
def evaluate(model_name, y_test, y_pred, y_prob):
    return {
        'Model'    : model_name,
        'ROC-AUC'  : roc_auc_score(y_test, y_prob),
        'F1'       : f1_score(y_test, y_pred, zero_division=0),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall'   : recall_score(y_test, y_pred, zero_division=0),
        'Accuracy' : accuracy_score(y_test, y_pred),
    }


# this applies the right threshold strategy based on protein config
def apply_threshold(y_test, y_prob, use_optimal):
    if use_optimal:
        threshold = get_optimal_threshold(y_test, y_prob)
    else:
        threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, threshold


# this plots the roc curves for all models for one protein
def plot_roc_curves(protein_name, roc_data):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, linewidth=2,
                color=COLORS.get(model_name, 'gray'),
                label=f'{model_name} (AUC={auc:.3f})')

    # diagonal reference line for a random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{protein_name} ({DISEASE[protein_name]})\nROC Curves',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_PATH,
                     f'{protein_name}_roc_curves.png'), dpi=150
    )
    plt.close()
    print(f'  Saved: {protein_name}_roc_curves.png')


# this plots a confusion matrix for each model side by side
def plot_confusion_matrices(protein_name, cm_data):
    n_models  = len(cm_data)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cm_data.items()):
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-binder', 'Binder'], fontsize=9)
        ax.set_yticklabels(['Non-binder', 'Binder'], fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(model_name, fontweight='bold', fontsize=10)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'{protein_name} ({DISEASE[protein_name]})\nConfusion Matrices',
                 fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_PATH,
                     f'{protein_name}_confusion_matrices.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f'  Saved: {protein_name}_confusion_matrices.png')


# creating the baseline subfolder for figures
os.makedirs(FIG_PATH, exist_ok=True)

# training and evaluating all models for each protein
all_results = []

for protein_name in PROTEINS:
    print(f"\n{'='*60}")
    print(f"PROTEIN: {protein_name} ({DISEASE[protein_name]})")
    print(f"{'='*60}")

    config = PROTEIN_CONFIG[protein_name]

    # loading the train and test data
    X_train, y_train, X_test, y_test = load_data(protein_name)
    if X_train is None:
        continue

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Binders in train : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  Binders in test  : {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print(f"  Features         : {X_train.shape[1]}")
    print(f"  Undersample      : {config['undersample']} | "
          f"Optimal threshold: {config['use_optimal_thresh']}")

    # applying undersampling to training data if configured
    if config['undersample']:
        ratio            = config.get('undersample_ratio', 5)
        X_train, y_train = undersample(X_train, y_train, ratio)

    # computing class weights after undersampling
    cw = get_class_weight(y_train)
    print(f"  Class weights    : {cw}")

    # scaling features for logistic regression
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    roc_data = {}
    cm_data  = {}
    use_opt  = config['use_optimal_thresh']

    # model 1 — logistic regression
    print(f"\n  [1/3] Logistic Regression...")
    lr = LogisticRegression(
        max_iter     = 1000,
        class_weight = 'balanced',
        random_state = RANDOM_STATE,
        C            = 1.0
    )
    lr.fit(X_train_scaled, y_train)
    y_prob_lr          = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr, thresh  = apply_threshold(y_test, y_prob_lr, use_opt)

    metrics_lr            = evaluate('Logistic Regression', y_test, y_pred_lr, y_prob_lr)
    metrics_lr['Protein'] = protein_name
    all_results.append(metrics_lr)

    fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
    roc_data['Logistic Regression'] = (fpr, tpr, metrics_lr['ROC-AUC'])
    cm_data['Logistic Regression']  = confusion_matrix(y_test, y_pred_lr)

    print(f"    Threshold: {thresh:.3f}")
    print(f"    ROC-AUC  : {metrics_lr['ROC-AUC']:.4f}")
    print(f"    F1-Score : {metrics_lr['F1']:.4f}")
    print(f"    Precision: {metrics_lr['Precision']:.4f}")
    print(f"    Recall   : {metrics_lr['Recall']:.4f}")

    # model 2 — random forest
    print(f"\n  [2/3] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators = 100,
        class_weight = 'balanced',
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        max_depth    = 10,
    )
    rf.fit(X_train, y_train)
    y_prob_rf          = rf.predict_proba(X_test)[:, 1]
    y_pred_rf, thresh  = apply_threshold(y_test, y_prob_rf, use_opt)

    metrics_rf            = evaluate('Random Forest', y_test, y_pred_rf, y_prob_rf)
    metrics_rf['Protein'] = protein_name
    all_results.append(metrics_rf)

    fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
    roc_data['Random Forest'] = (fpr, tpr, metrics_rf['ROC-AUC'])
    cm_data['Random Forest']  = confusion_matrix(y_test, y_pred_rf)

    print(f"    Threshold: {thresh:.3f}")
    print(f"    ROC-AUC  : {metrics_rf['ROC-AUC']:.4f}")
    print(f"    F1-Score : {metrics_rf['F1']:.4f}")
    print(f"    Precision: {metrics_rf['Precision']:.4f}")
    print(f"    Recall   : {metrics_rf['Recall']:.4f}")

    # model 3 — xgboost
    if XGBOOST_AVAILABLE:
        print(f"\n  [3/3] XGBoost...")

        neg              = (y_train == 0).sum()
        pos              = (y_train == 1).sum()
        scale_pos_weight = neg / pos

        xgb = XGBClassifier(
            n_estimators     = 100,
            max_depth        = 6,
            learning_rate    = 0.1,
            scale_pos_weight = scale_pos_weight,
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
            eval_metric      = 'logloss',
            verbosity        = 0,
        )
        xgb.fit(X_train, y_train)
        y_prob_xgb          = xgb.predict_proba(X_test)[:, 1]
        y_pred_xgb, thresh  = apply_threshold(y_test, y_prob_xgb, use_opt)

        metrics_xgb            = evaluate('XGBoost', y_test, y_pred_xgb, y_prob_xgb)
        metrics_xgb['Protein'] = protein_name
        all_results.append(metrics_xgb)

        fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
        roc_data['XGBoost'] = (fpr, tpr, metrics_xgb['ROC-AUC'])
        cm_data['XGBoost']  = confusion_matrix(y_test, y_pred_xgb)

        print(f"    Threshold: {thresh:.3f}")
        print(f"    ROC-AUC  : {metrics_xgb['ROC-AUC']:.4f}")
        print(f"    F1-Score : {metrics_xgb['F1']:.4f}")
        print(f"    Precision: {metrics_xgb['Precision']:.4f}")
        print(f"    Recall   : {metrics_xgb['Recall']:.4f}")
    else:
        print(f"\n  [3/3] XGBoost - skipped (not installed)")

    # generating the plots
    print(f"\n  Generating plots...")
    plot_roc_curves(protein_name, roc_data)
    plot_confusion_matrices(protein_name, cm_data)

    # printing the best model classification report
    print(f"\n  Best model classification report:")
    best_model_name = max(
        [m for m in ['Logistic Regression', 'Random Forest', 'XGBoost']
         if any(r['Model'] == m and r['Protein'] == protein_name
                for r in all_results)],
        key=lambda m: next(
            r['ROC-AUC'] for r in all_results
            if r['Model'] == m and r['Protein'] == protein_name
        )
    )
    if best_model_name == 'Logistic Regression':
        y_pred_best = y_pred_lr
    elif best_model_name == 'Random Forest':
        y_pred_best = y_pred_rf
    else:
        y_pred_best = y_pred_xgb

    print(f"  ({best_model_name})")
    print(classification_report(
        y_test, y_pred_best,
        target_names=['Non-binder', 'Binder'],
        zero_division=0
    ))


# building the final summary table
print(f"\n{'='*60}")
print("FINAL RESULTS - ALL PROTEINS & MODELS")
print(f"{'='*60}")

results_df = pd.DataFrame(all_results)
results_df = results_df[['Protein', 'Model', 'ROC-AUC', 'F1',
                          'Precision', 'Recall', 'Accuracy']]

display_df = results_df.copy()
for col in ['ROC-AUC', 'F1', 'Precision', 'Recall', 'Accuracy']:
    display_df[col] = display_df[col].round(4)

print(display_df.to_string(index=False))

# saving the results
results_path = os.path.join(RES_PATH, 'baseline_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: baseline_results.csv")


# making a comparison plot
print(f"\nGenerating comparison plots...")

metrics_to_plot = ['ROC-AUC', 'F1', 'Precision', 'Recall']
fig, axes       = plt.subplots(2, 2, figsize=(16, 12))
axes_flat       = axes.flatten()

models   = results_df['Model'].unique()
proteins = results_df['Protein'].unique()
x        = np.arange(len(proteins))
width    = 0.25

for ax, metric in zip(axes_flat, metrics_to_plot):
    for i, model in enumerate(models):
        model_df = results_df[results_df['Model'] == model]
        values   = [
            model_df[model_df['Protein'] == p][metric].values[0]
            if len(model_df[model_df['Protein'] == p]) > 0 else 0
            for p in proteins
        ]
        bars = ax.bar(x + i * width, values, width,
                      label=model,
                      color=COLORS.get(model, 'gray'),
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [f'{p}\n({DISEASE[p]})' for p in proteins], fontsize=9
    )
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} by Protein and Model',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('Baseline Model Comparison - All Proteins',
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_PATH, 'baseline_comparison.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print(f"Saved: baseline_comparison.png")


# final summary
print(f"\n{'='*60}")
print("BASELINE MODELS COMPLETE")
print(f"{'='*60}")

for protein in proteins:
    protein_results = results_df[results_df['Protein'] == protein]
    best            = protein_results.loc[protein_results['ROC-AUC'].idxmax()]
    print(f"\n  {protein} ({DISEASE[protein]}):")
    print(f"    Best model : {best['Model']}")
    print(f"    ROC-AUC    : {best['ROC-AUC']:.4f}")
    print(f"    F1-Score   : {best['F1']:.4f}")

print(f"\n  Figures saved to : {FIG_PATH}")
print(f"  Results saved to : {results_path}")
print(f"\n  Next: GNN.py - Graph Neural Network")
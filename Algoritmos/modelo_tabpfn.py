import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from tabpfn import TabPFNClassifier

print("\n" + "="*80)
print("TabPFN FOUNDATION MODEL FOR ALZHEIMER'S DISEASE CLASSIFICATION")
print("="*80)
print("Model Type: Pre-trained Transformer (In-Context Learning)")
print("Task: Multi-class Classification (AD, MCI, NL)")
print("Dataset: TADPOLE (ADNI)")
print("="*80 + "\n")

#%% CONFIGURATION AND HYPERPARAMETERS

# Create directories
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Hyperparameters Dictionary (for scientific documentation)
HYPERPARAMETERS = {
    "model_name": "TabPFN",
    "model_type": "Foundation Model (Pre-trained Transformer)",
    "n_ensemble_configurations": 32,  # Default TabPFN ensemble size
    "device": "cpu",  # TabPFN optimized for CPU
    "n_features_selected": 100,  # Maximum features for TabPFN
    "feature_selection_method": "ANOVA F-statistic (SelectKBest)",
    "standardization": "StandardScaler (mean=0, std=1)",
    "train_test_val_split": "60-20-20 stratified",
    "random_state": 42,
    "class_mapping": {
        "Dementia": "AD",
        "MCI": "MCI",
        "NL": "NL"
    },
    "rationale": {
        "n_ensemble": "Default value balances prediction accuracy with computational efficiency",
        "n_features": "TabPFN constraint; ANOVA F-statistic selects most discriminative features",
        "standardization": "Required for fair feature comparison in ANOVA and model input",
        "no_hyperparameter_tuning": "TabPFN uses fixed pre-trained weights; no gradient descent",
        "stratified_split": "Maintains class proportions across train/val/test sets"
    }
}

# Save hyperparameters
with open("results/tabpfn_hyperparameters.json", "w") as f:
    json.dump(HYPERPARAMETERS, f, indent=4)

print("Hyperparameters saved to: results/tabpfn_hyperparameters.json")
print(f"Feature Selection: Top {HYPERPARAMETERS['n_features_selected']} features using {HYPERPARAMETERS['feature_selection_method']}")
print(f"Ensemble Size: {HYPERPARAMETERS['n_ensemble_configurations']} configurations")
print(f"Device: {HYPERPARAMETERS['device'].upper()}\n")

#%% DATA LOADING

print("="*80)
print("LOADING DATASETS")
print("="*80)

train_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv")
validation_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_VALIDATION.csv")
test_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TEST.csv")

print(f"Train set:      {len(train_df)} samples")
print(f"Validation set: {len(validation_df)} samples")
print(f"Test set:       {len(test_df)} samples")
print(f"Total:          {len(train_df) + len(validation_df) + len(test_df)} samples\n")

# Drop ID columns
train_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
validation_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
test_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")

target = 'DX'

#%% CLASS MAPPING AND FILTERING

print("="*80)
print("CLASS MAPPING AND FILTERING")
print("="*80)

def mapear_dx(val):
    """
    Map diagnosis values to standardized class labels.
    
    Handles both text labels (from TADPOLE) and legacy numeric codes.
    Filters out transition states and unstable diagnoses.
    
    Returns:
        "AD": Alzheimer's Disease (Dementia)
        "MCI": Mild Cognitive Impairment
        "NL": Normal/Cognitively Normal
        "OTHER": Invalid or transition states (to be filtered)
    """
    val_str = str(val).strip()
    
    # Primary text labels from TADPOLE
    if val_str == "Dementia": return "AD"
    if val_str == "MCI": return "MCI"
    if val_str == "NL": return "NL"
    
    # Legacy numeric codes (if present - 0=AD, 1=MCI, 2=NL)
    if val_str in ["0", "0.0", "AD"]: return "AD"
    if val_str in ["1", "1.0"]: return "MCI"
    if val_str in ["2", "2.0"]: return "NL"
    
    # Transition states or invalid (to be filtered)
    return "OTHER"

# Apply mapping
train_df[target] = train_df[target].apply(mapear_dx)
validation_df[target] = validation_df[target].apply(mapear_dx)
test_df[target] = test_df[target].apply(mapear_dx)

# Filter out invalid classes
train_df = train_df[train_df[target] != "OTHER"].reset_index(drop=True)
validation_df = validation_df[validation_df[target] != "OTHER"].reset_index(drop=True)
test_df = test_df[test_df[target] != "OTHER"].reset_index(drop=True)

print("Class mapping applied:")
print("  Dementia → AD (Alzheimer's Disease)")
print("  MCI → MCI (Mild Cognitive Impairment)")
print("  NL → NL (Normal/Cognitively Normal)")
print("\nTransition states and invalid diagnoses filtered out.\n")

#%% CLASS DISTRIBUTION ANALYSIS

print("="*80)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*80)

def print_class_distribution(df, dataset_name):
    """Print class distribution with percentages."""
    print(f"\n{dataset_name}:")
    conteo = df[target].value_counts(sort=False)
    total = len(df)
    for clase in sorted(conteo.index):
        count = conteo[clase]
        pct = (count / total) * 100
        print(f"  {clase}: {count:4d} samples ({pct:5.2f}%)")

print_class_distribution(train_df, "Training Set")
print_class_distribution(validation_df, "Validation Set")
print_class_distribution(test_df, "Test Set")

print("\n" + "-"*80)
print("NOTE: Class imbalance is inherent to clinical Alzheimer's datasets.")
print("MCI is typically the most prevalent stage in research cohorts.")
print("Stratified splitting ensures proportional representation across splits.")
print("-"*80 + "\n")

#%% LABEL ENCODING

print("="*80)
print("LABEL ENCODING")
print("="*80)

# LabelEncoder creates alphabetical mapping: AD=0, MCI=1, NL=2
le = LabelEncoder()
y_train = le.fit_transform(train_df[target])
y_validation = le.transform(validation_df[target])
y_test = le.transform(test_df[target])

# Extract feature matrices
X_train_all = train_df.drop(columns=[target])
X_validation_all = validation_df.drop(columns=[target])
X_test_all = test_df.drop(columns=[target])

etiquetas_clases = [str(c) for c in le.classes_]
print(f"Classes encoded: {etiquetas_clases}")
print(f"Encoding: {dict(zip(etiquetas_clases, range(len(etiquetas_clases))))}\n")

#%% FEATURE STANDARDIZATION

print("="*80)
print("FEATURE STANDARDIZATION")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_validation_scaled = scaler.transform(X_validation_all)
X_test_scaled = scaler.transform(X_test_all)

print("StandardScaler applied (mean=0, std=1)")
print(f"Features standardized: {X_train_all.shape[1]}")
print("Rationale: Ensures fair comparison in ANOVA F-statistic and model input\n")

#%% FEATURE SELECTION

print("="*80)
print("FEATURE SELECTION")
print("="*80)

# TabPFN constraint: maximum 100 features
N_FEATURES = HYPERPARAMETERS['n_features_selected']

selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_validation_selected = selector.transform(X_validation_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = X_train_all.columns[selected_indices].tolist()

print(f"Method: ANOVA F-statistic (SelectKBest)")
print(f"Features selected: {N_FEATURES} out of {X_train_all.shape[1]}")
print(f"Rationale: TabPFN computational constraint; ANOVA identifies most discriminative features")
print(f"\nTop 10 selected features:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"  {i:2d}. {feat}")

# Save selected features list for use in other models
with open("results/tabpfn_selected_features.txt", "w") as f:
    for feature in selected_features:
        f.write(f"{feature}\n")
print("Selected features list saved to: results/tabpfn_selected_features.txt")

# Save as JSON for easy loading in Python
with open("results/tabpfn_selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=4)
print("Selected features JSON saved to: results/tabpfn_selected_features.json\n")

#%% MODEL TRAINING

print("="*80)
print("TABPFN MODEL TRAINING")
print("="*80)

print("Initializing TabPFN classifier...")
print(f"  N_ensemble_configurations: {HYPERPARAMETERS['n_ensemble_configurations']}")
print(f"  Device: {HYPERPARAMETERS['device']}")
print("\nNote: TabPFN uses pre-trained weights; no gradient descent or epochs.\n")

# Initialize TabPFN
model = TabPFNClassifier(
    device=HYPERPARAMETERS['device'],
    N_ensemble_configurations=HYPERPARAMETERS['n_ensemble_configurations']
)

print("Training TabPFN (in-context learning)...")
training_start = datetime.now()

# Fit model
model.fit(X_train_selected, y_train)

training_time = (datetime.now() - training_start).total_seconds()
print(f"Training completed in {training_time:.2f} seconds\n")

#%% VALIDATION SET EVALUATION

print("="*80)
print("VALIDATION SET EVALUATION")
print("="*80)

y_validation_pred = model.predict(X_validation_selected)
y_validation_pred_proba = model.predict_proba(X_validation_selected)

# Metrics
val_acc = accuracy_score(y_validation, y_validation_pred)
val_precision = precision_score(y_validation, y_validation_pred, average='weighted', zero_division=0)
val_recall = recall_score(y_validation, y_validation_pred, average='weighted', zero_division=0)
val_f1 = f1_score(y_validation, y_validation_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {val_acc:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1-Score:  {val_f1:.4f}\n")

print("Classification Report (Validation):")
print(classification_report(y_validation, y_validation_pred, target_names=etiquetas_clases, zero_division=0))

#%% TEST SET EVALUATION

print("="*80)
print("TEST SET EVALUATION (FINAL PERFORMANCE)")
print("="*80)

y_test_pred = model.predict(X_test_selected)
y_test_pred_proba = model.predict_proba(X_test_selected)

# Metrics
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}\n")

print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred, target_names=etiquetas_clases, zero_division=0))

#%% CONFUSION MATRIX

print("="*80)
print("GENERATING CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=etiquetas_clases,
            yticklabels=etiquetas_clases,
            cbar_kws={'label': 'Count'})
plt.title('TabPFN - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Diagnosis', fontsize=12)
plt.ylabel('True Diagnosis', fontsize=12)
plt.tight_layout()
plt.savefig('figures/TabPFN_Confusion_Matrix.svg', format='svg', bbox_inches='tight')
plt.close()

print("Saved: figures/TabPFN_Confusion_Matrix.svg\n")

#%% ROC CURVES (ONE-VS-REST)

print("="*80)
print("GENERATING ROC CURVES")
print("="*80)

# Binarize labels for ROC (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=range(len(etiquetas_clases)))

# Handle binary case
n_classes_proba = y_test_pred_proba.shape[1]
if n_classes_proba == 2 and y_test_bin.ndim == 2 and y_test_bin.shape[1] == 1:
    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes_proba):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_test_pred_proba[:, i])

# Plot
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i in range(n_classes_proba):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{etiquetas_clases[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('TabPFN - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/TabPFN_ROC_Curves.svg', format='svg', bbox_inches='tight')
plt.close()

print("Saved: figures/TabPFN_ROC_Curves.svg")
print("\nAUC Scores:")
for i in range(n_classes_proba):
    print(f"  {etiquetas_clases[i]}: {roc_auc[i]:.4f}")
print()

#%% SAVE RESULTS

print("="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    "model": "TabPFN",
    "timestamp": datetime.now().isoformat(),
    "training_time_seconds": training_time,
    "hyperparameters": HYPERPARAMETERS,
    "data_splits": {
        "train_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df)
    },
    "validation_metrics": {
        "accuracy": float(val_acc),
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1_score": float(val_f1)
    },
    "test_metrics": {
        "accuracy": float(test_acc),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(test_f1)
    },
    "roc_auc_scores": {etiquetas_clases[i]: float(roc_auc[i]) for i in range(n_classes_proba)},
    "confusion_matrix": cm.tolist(),
    "class_labels": etiquetas_clases
}

with open("results/tabpfn_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to: results/tabpfn_results.json")

# Save model artifacts
joblib.dump(model, "models/tabpfn_model.pkl")
joblib.dump(scaler, "models/tabpfn_scaler.pkl")
joblib.dump(selector, "models/tabpfn_selector.pkl")
joblib.dump(le, "models/tabpfn_label_encoder.pkl")

print("Model saved to: models/tabpfn_model.pkl")
print("Scaler saved to: models/tabpfn_scaler.pkl")
print("Feature selector saved to: models/tabpfn_selector.pkl")
print("Label encoder saved to: models/tabpfn_label_encoder.pkl")

print("\n" + "="*80)
print("TABPFN MODEL TRAINING COMPLETE")
print("="*80)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test F1-Score: {test_f1:.4f}")
print("="*80 + "\n")
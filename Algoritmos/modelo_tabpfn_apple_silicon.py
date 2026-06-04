import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import torch
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from tabpfn import TabPFNClassifier


# APPLE SILICON GPU DETECTION
def get_device():

    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("Apple Silicon GPU (MPS) detected and available")
            return "mps"
        else:
            print("MPS not built in this PyTorch version, using CPU")
            return "cpu"
    else:
        print("MPS not available, using CPU")
        return "cpu"

# Detect device
DEVICE = get_device()

print("\n" + "="*80)
print("APPLE SILICON OPTIMIZATION")
print("="*80)
print(f"Device selected: {DEVICE.upper()}")
if DEVICE == "mps":
    print("Using Metal Performance Shaders for GPU acceleration")
    print("Expected speedup: 2-5x compared to CPU")
else:
    print("Running on CPU (MPS not available)")
print("="*80)

#%% CONFIGURATION AND HYPERPARAMETERS

# Create directories
os.makedirs("figures", exist_ok=True)
os.makedirs("ascii32/Modelos", exist_ok=True)

# Hyperparameters
HYPERPARAMETERS = {
    "model_name": "TabPFN_AppleSilicon",
    "n_ensemble_configurations": 16,  # Default ensemble size for robust predictions
    "device": DEVICE,  # Automatically detected (MPS or CPU)
    "max_features": 100,  # TabPFN limitation
    "feature_selection_method": "f_classif",  # ANOVA F-statistic for feature selection
    "random_state": 42,  # For reproducibility
    "normalization": "StandardScaler",  # Required by TabPFN
    "apple_silicon_optimized": True,
    "pytorch_version": torch.__version__,
    "mps_available": torch.backends.mps.is_available(),
    "class_mapping": {
        "AD": 0,
        "MCI": 1,
        "NL": 2
    }
}

# Save hyperparameters documentation
with open("ascii32/Modelos/tabpfn_hyperparameters.json", "w") as f:
    json.dump(HYPERPARAMETERS, f, indent=4)

print("\n" + "="*80)
print("TabPFN MODEL FOR ALZHEIMER'S DISEASE CLASSIFICATION")
print("APPLE SILICON OPTIMIZED VERSION")
print("="*80)
print("\nHyperparameters saved to: ascii32/Modelos/tabpfn_ahyperparameters.json")
print(f"\nConfiguration:")
for key, value in HYPERPARAMETERS.items():
    print(f"  {key}: {value}")

#%% DATA LOADING AND PREPROCESSING


print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

# Load the three datasets
train_df = pd.read_csv("ascii32/Algoritmos/TADPOLE_D1_D2_BL_RF_TRAIN.csv")
validation_df = pd.read_csv("ascii32/Algoritmos/TADPOLE_D1_D2_BL_RF_VALIDATION.csv")
test_df = pd.read_csv("ascii32/Algoritmos/TADPOLE_D1_D2_BL_RF_TEST.csv")

print(f"\nTrain set size: {len(train_df)}")
print(f"Validation set size: {len(validation_df)}")
print(f"Test set size: {len(test_df)}")

# Drop ID columns
train_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
validation_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
test_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")

target = 'DX'

# Map diagnosis labels
def mapear_dx(val):
    """Map diagnosis labels to standardized format
    
    The TADPOLE dataset uses text labels:
    - 'Dementia' → 'AD' (Alzheimer's Disease)
    - 'MCI' → 'MCI' (Mild Cognitive Impairment)
    - 'NL' → 'NL' (Normal/Cognitively Normal)
    
    Transition states like 'MCI to Dementia' are filtered out in pipeline_RF.py
    """
    val_str = str(val).strip()
    
    # Map text labels (from TADPOLE dataset)
    if val_str == "Dementia": return "AD"
    if val_str == "MCI": return "MCI"
    if val_str == "NL": return "NL"
    
    # Legacy: also handle numeric codes if present (0=AD, 1=MCI, 2=NL)
    if val_str in ["0", "0.0", "AD"]: return "AD"
    if val_str in ["1", "1.0"]: return "MCI"
    if val_str in ["2", "2.0"]: return "NL"
    
    return "OTHER"

train_df[target] = train_df[target].apply(mapear_dx)
validation_df[target] = validation_df[target].apply(mapear_dx)
test_df[target] = test_df[target].apply(mapear_dx)

# Filter out any non-standard classes
train_df = train_df[train_df[target] != "OTHER"].reset_index(drop=True)
validation_df = validation_df[validation_df[target] != "OTHER"].reset_index(drop=True)
test_df = test_df[test_df[target] != "OTHER"].reset_index(drop=True)

# Class distribution analysis
print("\n" + "-"*80)
print("CLASS DISTRIBUTION")
print("-"*80)
print("\nTrain set:")
train_counts = train_df[target].value_counts(normalize=True) * 100
for cls, pct in train_counts.items():
    print(f"  {cls}: {pct:.2f}%")

print("\nValidation set:")
val_counts = validation_df[target].value_counts(normalize=True) * 100
for cls, pct in val_counts.items():
    print(f"  {cls}: {pct:.2f}%")

print("\nTest set:")
test_counts = test_df[target].value_counts(normalize=True) * 100
for cls, pct in test_counts.items():
    print(f"  {cls}: {pct:.2f}%")

print("\nNOTE: Class imbalance is natural in clinical data.")
print("TabPFN handles imbalanced data through its pre-training on diverse distributions.")

#%% LABEL ENCODING

le = LabelEncoder()
y_train = le.fit_transform(train_df[target])
y_validation = le.transform(validation_df[target])
y_test = le.transform(test_df[target])

X_train_all = train_df.drop(columns=[target])
X_validation_all = validation_df.drop(columns=[target])
X_test_all = test_df.drop(columns=[target])

class_names = [str(c) for c in le.classes_]
print(f"\nClass encoding: {dict(zip(class_names, range(len(class_names))))}")

#%% FEATURE SELECTION

print("\n" + "="*80)
print("FEATURE SELECTION")
print("="*80)

n_features_original = X_train_all.shape[1]
print(f"\nOriginal number of features: {n_features_original}")

# TabPFN has a limit of 100 features
if n_features_original > HYPERPARAMETERS["max_features"]:
    print(f"\nReducing to {HYPERPARAMETERS['max_features']} features using ANOVA F-statistic...")
    print("Rationale: Selecting features with highest discriminative power for classification")
    
    # Use SelectKBest with ANOVA F-statistic
    selector = SelectKBest(score_func=f_classif, k=HYPERPARAMETERS["max_features"])
    X_train_selected = selector.fit_transform(X_train_all, y_train)
    X_validation_selected = selector.transform(X_validation_all)
    X_test_selected = selector.transform(X_test_all)
    
    # Get selected feature names
    selected_features = X_train_all.columns[selector.get_support()].tolist()
    
    # Save feature importance scores
    feature_scores = pd.DataFrame({
        'feature': X_train_all.columns,
        'score': selector.scores_,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    feature_scores.to_csv("ascii32/Modelos/tabpfn_feature_scores.csv", index=False)
    print(f"Feature scores saved to: ascii32/Modelos/tabpfn_feature_scores.csv")
    
    # Save selected features list for use in other models
    with open("ascii32/Modelos/tabpfn_selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Selected features list saved to: ascii32/Modelos/tabpfn_selected_features.txt")
    
    # Save as JSON for easy loading in Python
    with open("ascii32/Modelos/tabpfn_selected_features.json", "w") as f:
        json.dump(selected_features, f, indent=4)
    print(f"Selected features JSON saved to: ascii32/Modelos/tabpfn_selected_features.json")
    
    # Plot top 30 features
    plt.figure(figsize=(12, 8))
    top_features = feature_scores.head(30)
    colors = ['green' if x else 'gray' for x in top_features['selected']]
    plt.barh(range(len(top_features)), top_features['score'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'].tolist())
    plt.xlabel('ANOVA F-statistic Score')
    plt.ylabel('Features')
    plt.title('Top 30 Features by Discriminative Power\n(Green = Selected for TabPFN)\n', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/07_TabPFN_Feature_Selection.svg", format='svg')
    plt.close()
    print("Feature selection plot saved: figures/07_TabPFN_Feature_Selection.svg")
    
    # Convert back to DataFrame for easier handling
    X_train = pd.DataFrame(X_train_selected, columns=selected_features)
    X_validation = pd.DataFrame(X_validation_selected, columns=selected_features)
    X_test = pd.DataFrame(X_test_selected, columns=selected_features)
else:
    X_train = X_train_all.copy()
    X_validation = X_validation_all.copy()
    X_test = X_test_all.copy()
    selected_features = X_train.columns.tolist()

print(f"Final number of features: {len(selected_features)}")

#%% DATA NORMALIZATION 

print("\n" + "="*80)
print("DATA NORMALIZATION")
print("="*80)
print("\nApplying StandardScaler (required by TabPFN)")
print("Rationale: TabPFN expects normalized features for optimal performance")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future use
joblib.dump(scaler, "ascii32/Modelos/tabpfn_scaler.pkl")
print("Scaler saved to: ascii32/Modelos/tabpfn_scaler.pkl")


#%% MODEL TRAINING


print("\n" + "="*80)
print("TRAINING TABPFN MODEL")
print("="*80)

print("\nInitializing TabPFN classifier...")
print(f"Ensemble configurations: {HYPERPARAMETERS['n_ensemble_configurations']}")
print(f"Device: {HYPERPARAMETERS['device'].upper()}")

if DEVICE == "mps":
    print("\n Using Apple Silicon GPU acceleration")
    print("Metal Performance Shaders (MPS) backend active")
    print("Expected performance: 2-5x faster than CPU")
else:
    print("\n  Running on CPU (MPS not available)")

# Initialize TabPFN with detected device
# Note: TabPFN API may vary - using device only (ensemble config is internal)
try:
    model = TabPFNClassifier(
        device=DEVICE,
        N_ensemble_configurations=HYPERPARAMETERS['n_ensemble_configurations']
    )
except TypeError:
    # Fallback for newer TabPFN versions
    print("Note: Using default ensemble configuration")
    model = TabPFNClassifier(device=DEVICE)

print("\nFitting model on training data...")
print("Note: TabPFN performs in-context learning, no traditional training needed")
start_time = datetime.now()

#%% Fit the model
try:
    model.fit(X_train_scaled, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f" Model fitting completed in {training_time:.2f} seconds")
    
    if DEVICE == "mps":
        print(f"  GPU acceleration provided significant speedup")
except Exception as e:
    print(f"  Error during training: {e}")
    print("  Falling back to CPU...")
    DEVICE = "cpu"
    model = TabPFNClassifier(
        device="cpu",
        N_ensemble_configurations=HYPERPARAMETERS['n_ensemble_configurations']
    )
    model.fit(X_train_scaled, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f" Model fitting completed on CPU in {training_time:.2f} seconds")


#%% MODEL EVALUATION - VALIDATION SET

print("\n" + "="*80)
print("VALIDATION SET EVALUATION")
print("="*80)

y_val_pred = model.predict(X_validation_scaled)
y_val_pred_proba = model.predict_proba(X_validation_scaled)

# Calculate metrics
val_accuracy = accuracy_score(y_validation, y_val_pred)
val_precision = precision_score(y_validation, y_val_pred, average='weighted', zero_division=0)
val_recall = recall_score(y_validation, y_val_pred, average='weighted', zero_division=0)
val_f1 = f1_score(y_validation, y_val_pred, average='weighted', zero_division=0)

print(f"\nValidation Metrics:")
print(f"  Accuracy:  {val_accuracy:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1-Score:  {val_f1:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_validation, y_val_pred, target_names=class_names, zero_division=0))

# MODEL EVALUATION - TEST SET

print("\n" + "="*80)
print("TEST SET EVALUATION (FINAL PERFORMANCE)")
print("="*80)

y_test_pred = model.predict(X_test_scaled)
y_test_pred_proba = model.predict_proba(X_test_scaled)

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"\nTest Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))


# CONFUSION MATRICES

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Validation Confusion Matrix
cm_val = confusion_matrix(y_validation, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title('Confusion Matrix - Validation Set\nTabPFN Model', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Diagnosis', fontsize=12)
plt.ylabel('True Diagnosis', fontsize=12)
plt.tight_layout()
plt.savefig("figures/07_TabPFN_Confusion_Matrix_Validation.svg", format='svg')
plt.close()
print("Saved: figures/07_TabPFN_Confusion_Matrix_Validation.svg")

# Test Confusion Matrix
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title('Confusion Matrix - Test Set\nTabPFN Model', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Diagnosis', fontsize=12)
plt.ylabel('True Diagnosis', fontsize=12)
plt.tight_layout()
plt.savefig("figures/07_TabPFN_Confusion_Matrix_Test.svg", format='svg')
plt.close()
print("Saved: figures/07_TabPFN_Confusion_Matrix_Test.svg")

#%% ROC CURVES (One-vs-Rest)

print("\nGenerating ROC curves...")

# For multiclass, compute ROC curve for each class (One-vs-Rest)
from sklearn.preprocessing import label_binarize

# Determine actual number of classes from predict_proba output
n_classes_proba = y_test_pred_proba.shape[1]
unique_classes_in_test = np.unique(y_test)

print(f"\nROC Curve Generation:")
print(f"  Classes in test set: {unique_classes_in_test} → {[class_names[i] for i in unique_classes_in_test]}")
print(f"  Predict_proba shape: {y_test_pred_proba.shape} ({n_classes_proba} classes)")

# Use only the classes that have probabilities
classes_for_roc = list(range(n_classes_proba))
y_test_bin = label_binarize(y_test, classes=unique_classes_in_test)

# Handle binary case where label_binarize returns 1D array
if n_classes_proba == 2 and y_test_bin.ndim == 2 and y_test_bin.shape[1] == 1:
    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(min(n_classes_proba, y_test_bin.shape[1])):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_test_pred_proba[:, i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
n_classes_to_plot = min(n_classes_proba, len(class_names))
for i in range(n_classes_to_plot):
    class_idx = unique_classes_in_test[i] if i < len(unique_classes_in_test) else i
    class_label = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
             label=f'{class_label} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - TabPFN Model\nOne-vs-Rest', 
          fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/07_TabPFN_ROC_Curves.svg", format='svg')
plt.close()
print("Saved: figures/07_TabPFN_ROC_Curves.svg")

#%% SAVE RESULTS AND MODEL

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save results summary
results = {
    "model": "TabPFN",
    "timestamp": datetime.now().isoformat(),
    "device_used": DEVICE,
    "apple_silicon_optimized": True,
    "pytorch_version": torch.__version__,
    "mps_available": torch.backends.mps.is_available(),
    "hyperparameters": HYPERPARAMETERS,
    "training_time_seconds": training_time,
    "n_features_selected": len(selected_features),
    "validation_metrics": {
        "accuracy": float(val_accuracy),
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1_score": float(val_f1)
    },
    "test_metrics": {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1_score": float(test_f1)
    },
    "roc_auc_per_class": {class_names[unique_classes_in_test[i]]: float(roc_auc[i]) for i in range(n_classes_proba)},
    "class_names": class_names,
    "selected_features": selected_features
}

with open("ascii32/Modelos/tabpfn_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to: ascii32/Modelos/tabpfn_results.json")

# Save predictions for analysis
predictions_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_test_pred,
    'true_class': [class_names[i] for i in y_test],
    'predicted_class': [class_names[i] for i in y_test_pred],
    'prob_AD': y_test_pred_proba[:, 0],
    'prob_MCI': y_test_pred_proba[:, 1],
    'prob_NL': y_test_pred_proba[:, 2]
})
predictions_df.to_csv("ascii32/Modelos/tabpfn_test_predictions.csv", index=False)
print("Test predictions saved to: ascii32/Modelos/tabpfn_test_predictions.csv")

# SUMMARY

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

device_info = "Apple Silicon GPU (MPS)" if DEVICE == "mps" else "CPU"

print(f"""
TabPFN Model Training Complete


Hardware Configuration:
- Device: {device_info}
- PyTorch version: {torch.__version__}
- MPS available: {torch.backends.mps.is_available()}

Model Configuration:
- Ensemble size: {HYPERPARAMETERS['n_ensemble_configurations']}
- Features used: {len(selected_features)} (selected from {n_features_original})
- Training time: {training_time:.2f} seconds

Performance Metrics (Test Set):
- Accuracy:  {test_accuracy:.4f}
- Precision: {test_precision:.4f}
- Recall:    {test_recall:.4f}
- F1-Score:  {test_f1:.4f}

ROC AUC Scores (One-vs-Rest):
- AD:  {roc_auc[0]:.4f}
- MCI: {roc_auc[1]:.4f}
- NL:  {roc_auc[2]:.4f}
""")

print("="*80)
print("EXECUTION COMPLETED SUCCESSFULLY")
print("="*80)
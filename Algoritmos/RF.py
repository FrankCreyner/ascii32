# ==========================================
# 1. IMPORT NECESSARY LIBRARIES
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

import os
import sys
import joblib

script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_path)


# ==========================================
# 2. DATA LOADING AND PREPROCESSING
# ==========================================

# Define the dataframes
dfTrain = pd.read_csv('TADPOLE_D1_D2_BL_RF_TRAIN.csv')
dfTest = pd.read_csv('TADPOLE_D1_D2_BL_RF_TEST.csv')
dfVal = pd.read_csv('TADPOLE_D1_D2_BL_RF_VALIDATION.csv')

dfTrain = dfTrain[dfTrain["DX"].isin([0, 1, 2])]
dfTest  = dfTest[dfTest["DX"].isin([0, 1, 2])]
dfVal   = dfVal[dfVal["DX"].isin([0, 1, 2])]

# Define the mapping of diagnosis codes to labels

mapping = {
    2: "NL",  # Normal Cognition
    1: "MCI", # Mild Cognitive Impairment
    0: "AD"   # Alzheimer's Disease
}
dfTrain["DX"] = dfTrain["DX"].map(mapping)
dfTest["DX"]  = dfTest["DX"].map(mapping)
dfVal["DX"]   = dfVal["DX"].map(mapping)

# Separate identification columns (PTID and EXAMDATE)
cols_id_train = dfTrain[["PTID", "EXAMDATE"]].copy()
dfTrain = dfTrain.drop(columns=["PTID", "EXAMDATE"])

cols_id_test = dfTest[["PTID", "EXAMDATE"]].copy()
dfTest = dfTest.drop(columns=["PTID", "EXAMDATE"])

cols_id_val = dfVal[["PTID", "EXAMDATE"]].copy()
dfVal = dfVal.drop(columns=["PTID", "EXAMDATE"])

# Define the predictor variables (X) and target variable (y)
X_train = dfTrain.drop('DX', axis=1)
y_train = dfTrain['DX']

X_test = dfTest.drop('DX', axis=1)
y_test = dfTest['DX']

X_val = dfVal.drop('DX', axis=1)
y_val = dfVal['DX']


# ==========================================
# 3. MODEL TRAINING
# ==========================================

# Model the Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Cross-Validation Setup
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross Validation Accuracy Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")
rf.fit(X_train, y_train)


# ==========================================
# 4. EVALUATION (TEST AND VALIDATION) 
# ==========================================


# Define the evaluation sets to iterate over cleanly
evaluation_sets = {
    "TEST": (X_test, y_test),
    "VALIDATION": (X_val, y_val)
}

print("\n" + "="*60)
print("MODEL EVALUATION STARTED")
print("="*60)

for name, (X_data, y_true) in evaluation_sets.items():
    y_pred = rf.predict(X_data)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "-"*50)
    print(f"METRICS FOR THE {name} SET:")
    print("-" * 50)
    print(f"* Accuracy:     {accuracy:.4f}")
    print(f"* Precision:    {precision:.4f}")
    print(f"* Recall:       {recall:.4f}")
    print(f"* F1 Score:     {f1:.4f}")
    
    print(f"\nClassification Report ({name}):")
    print(classification_report(y_true, y_pred))
    
    # --- Generating Figures (SVG Output) ---
    fig_cm, ax_cm = plt.subplots(figsize=(10, 7)) 
    ConfusionMatrixDisplay.from_estimator(rf, X_data, y_true, cmap='Blues', ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix - {name}', fontsize=16)
    ax_cm.set_xlabel('Predicted Label (DX)', fontsize=12)
    ax_cm.set_ylabel('True Label (DX)', fontsize=12)

    plt.tight_layout()
    
    # Save Confusion Matrix plot
    cm_svg_filename = f'performance_{name}_confusion_matrix.svg'
    fig_cm.savefig(cm_svg_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved successfully: '{cm_svg_filename}'")

    fig_imp, ax_imp = plt.subplots(figsize=(14, 8))
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(30)
    
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax_imp)
    ax_imp.set_title(f'Top 30 Features ({name})', fontsize=16) 
    ax_imp.set_xlabel('Feature Importance Score', fontsize=12)
    ax_imp.set_ylabel('Predictor Feature', fontsize=12)

    plt.tight_layout()
    
    # Save Feature Importance plot
    imp_svg_filename = f'performance_{name}_feature_importance.svg'
    fig_imp.savefig(imp_svg_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved successfully: '{imp_svg_filename}'")

plt.show()

# ==========================================
# 5. VISUALIZING THE BIG TREE (DECISION TREE)
# ==========================================

fig_tree, ax_tree = plt.subplots(figsize=(50, 20))
class_names = [str(cls) for cls in y_train.unique()] 

plot_tree(rf.estimators_[0], 
          feature_names=X_train.columns, 
          class_names=class_names, 
          filled=True, 
          fontsize=8,
          ax=ax_tree)


ax_tree.set_title('Full Decision Tree Structure', fontsize=20)
plt.text(0.5, -0.1, 
         "This visualizes the decision rules used by one of the Random Forest trees.", 
         transform=ax_tree.transAxes, ha='center', fontsize=12, color='gray')

svg_tree_filename = 'full_decision_tree.svg'
fig_tree.savefig(svg_tree_filename, dpi=300, bbox_inches='tight')
print("\n" + "="*50)
print(f"Tree visualization saved successfully as '{svg_tree_filename}'")
plt.show()


# ==========================================
# 6. SAVING THE MODEL
# ==========================================

print("\n" + "="*40)
savemodel = input("Do you want to save the trained model? (y/n): ").strip().lower()
if savemodel == 'y':
    joblib.dump(rf, 'random_forest_model.pkl')
    print("Model saved successfully as 'random_forest_model.pkl'")
else:
    print("Model saving skipped")
print("="*40)
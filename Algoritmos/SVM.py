import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

# ==========================================
# 1. CONFIGURACIÓN Y REPRODUCIBILIDAD
# ==========================================
SEED = 42
OUTPUT_DIR = "imagenes_SVM"

# Crear carpeta para guardar imágenes si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seeds(seed):
    """Fija todas las semillas para garantizar la reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)
    
set_seeds(SEED)

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================

def preprocess_data(df, label_encoder=None, is_train=False):
    """Filtra clases, mapea etiquetas, elimina columnas con data leakage y codifica."""
    df = df[df["DX"].isin([0, 1, 2])].copy()
    mapping = {2: "NL", 1: "MCI", 0: "AD"}
    df["DX"] = df["DX"].map(mapping)
    
    cols_to_drop = ['DX', 'PTID', 'EXAMDATE', 'CDRSB']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y_raw = df["DX"]
    
    if is_train:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)
        
    return X, y, label_encoder

def plot_multiclass_roc(clf, X, y, label_encoder, title, filename):
    """Grafica curvas ROC multiclase (One-vs-Rest) y guarda la imagen."""
    y_score = clf.predict_proba(X)
    classes_encoded = label_encoder.transform(label_encoder.classes_)
    y_bin = label_binarize(y, classes=classes_encoded)
    n_classes = y_bin.shape[1]
    
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        class_name = label_encoder.classes_[i]
        
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC {class_name} (AUC = {roc_auc:0.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Azar (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Guardar y mostrar
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.show()
    print(f"-> Imagen guardada: {filepath}")

# ==========================================
# 3. CARGA Y PREPROCESAMIENTO
# ==========================================
print("Cargando y preprocesando datos...")
df_train = pd.read_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv")
df_val = pd.read_csv("TADPOLE_D1_D2_BL_RF_VALIDATION.csv")
df_test = pd.read_csv("TADPOLE_D1_D2_BL_RF_TEST.csv")

X_train, y_train, le = preprocess_data(df_train, is_train=True)
X_val, y_val, _ = preprocess_data(df_val, label_encoder=le)
X_test, y_test, _ = preprocess_data(df_test, label_encoder=le)

print("\nDistribución de clases TRAIN:")
print(pd.Series(le.inverse_transform(y_train)).value_counts())
print("\nDistribución de clases VALIDATION:")
print(pd.Series(le.inverse_transform(y_val)).value_counts())
print("\nDistribución de clases TEST:")
print(pd.Series(le.inverse_transform(y_test)).value_counts())


# ==========================================
# 4. ENTRENANDO MODELO SVM NORMAL (SIN PCA)
# ==========================================
print("\n" + "="*50)
print("ENTRENANDO MODELO SVM NORMAL (SIN PCA)")
print("="*50)

pipeline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_select", SelectKBest(score_func=mutual_info_classif)),
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED))
])

param_grid_svm = {
    "feature_select__k": [50, 100, 150],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001]
}

scoring = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}

grid_svm = GridSearchCV(
    pipeline_svm, param_grid_svm, cv=5, scoring=scoring, refit="f1", n_jobs=-1, verbose=1
)

grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_

# --- EVALUACIÓN SVM NORMAL ---
y_pred_val_svm = best_svm.predict(X_val)
y_pred_test_svm = best_svm.predict(X_test)

print("\n[Resultados SVM Normal - Set de VALIDACIÓN]")
print(classification_report(y_val, y_pred_val_svm, target_names=le.classes_))

print("\n[Resultados SVM Normal - Set de PRUEBA (FINAL)]")
print(classification_report(y_test, y_pred_test_svm, target_names=le.classes_))

# Guardar Gráfica de Matrices de Confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val_svm, display_labels=le.classes_, cmap="Blues", ax=axes[0])
axes[0].set_title("M. Confusión - Validación (SVM Normal)")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test_svm, display_labels=le.classes_, cmap="Blues", ax=axes[1])
axes[1].set_title("M. Confusión - Prueba (SVM Normal)")
plt.tight_layout()
filepath_cm_svm = os.path.join(OUTPUT_DIR, 'matriz_confusion_svm_normal.svg')
plt.savefig(filepath_cm_svm, format='svg', bbox_inches='tight')
plt.show()
print(f"-> Imagen guardada: {filepath_cm_svm}")

# Guardar Curvas ROC
plot_multiclass_roc(best_svm, X_test, y_test, le, "Curvas ROC Multiclase - SVM Normal (Test)", 'roc_svm_normal.svg')

# Guardar Importancia de características
selector_svm = best_svm.named_steps["feature_select"]
df_importancias = pd.DataFrame({'Caracteristica': X_train.columns, 'Importancia (Mutual Info)': selector_svm.scores_})
top_50_features = df_importancias.sort_values(by='Importancia (Mutual Info)', ascending=False).head(50)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importancia (Mutual Info)', y='Caracteristica', data=top_50_features, hue='Caracteristica', palette='magma', legend=False)
plt.title('Top 50 Biomarcadores más Importantes (SVM Normal)', fontsize=14)
plt.tight_layout()
filepath_imp_svm = os.path.join(OUTPUT_DIR, 'importancia_biomarcadores_svm_normal.svg')
plt.savefig(filepath_imp_svm, format='svg', bbox_inches='tight')
plt.show()
print(f"-> Imagen guardada: {filepath_imp_svm}")

joblib.dump(best_svm, 'SVM.pkl')
print("¡Modelo normal empaquetado y guardado como 'SVM.pkl'!")


# ==========================================
# 5. ENTRENANDO MODELO SVM CON PCA
# ==========================================
print("\n" + "="*50)
print("ENTRENANDO MODELO SVM CON PCA")
print("="*50)

pipeline_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_select", SelectKBest(score_func=mutual_info_classif)),
    ("pca", PCA(random_state=SEED)),
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED))
])

param_grid_pca = {
    "feature_select__k": [50, 100, 150],
    "pca__n_components": [20, 30, 40],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001]
}

grid_pca = GridSearchCV(
    pipeline_pca, param_grid_pca, cv=5, scoring=scoring, refit="f1", n_jobs=-1, verbose=1
)

grid_pca.fit(X_train, y_train)
best_svm_pca = grid_pca.best_estimator_

# --- EVALUACIÓN SVM CON PCA ---
y_pred_val_pca = best_svm_pca.predict(X_val)
y_pred_test_pca = best_svm_pca.predict(X_test)

print("\n[Resultados SVM con PCA - Set de VALIDACIÓN]")
print(classification_report(y_val, y_pred_val_pca, target_names=le.classes_))

print("\n[Resultados SVM con PCA - Set de PRUEBA (FINAL)]")
print(classification_report(y_test, y_pred_test_pca, target_names=le.classes_))

# Guardar Gráfica de Matrices de Confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val_pca, display_labels=le.classes_, cmap="Oranges", ax=axes[0])
axes[0].set_title("M. Confusión - Validación (SVM con PCA)")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test_pca, display_labels=le.classes_, cmap="Oranges", ax=axes[1])
axes[1].set_title("M. Confusión - Prueba (SVM con PCA)")
plt.tight_layout()
filepath_cm_pca = os.path.join(OUTPUT_DIR, 'matriz_confusion_svm_pca.svg')
plt.savefig(filepath_cm_pca, format='svg', bbox_inches='tight')
plt.show()
print(f"-> Imagen guardada: {filepath_cm_pca}")

# Guardar Curvas ROC
plot_multiclass_roc(best_svm_pca, X_test, y_test, le, "Curvas ROC Multiclase - SVM con PCA (Test)", 'roc_svm_pca.svg')

# Guardar Gráfica de Varianza Explicada
pca_step = best_svm_pca.named_steps["pca"]
varianza_acumulada = np.cumsum(pca_step.explained_variance_ratio_)
n_components_pca = pca_step.n_components_

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='-', color='#2ca02c')
plt.axhline(y=0.90, color='red', linestyle='--', label='90% de Varianza Explicada')
plt.axhline(y=0.95, color='orange', linestyle='--', label='95% de Varianza Explicada')
plt.title(f"Varianza Acumulada Explicada por el PCA ({n_components_pca} Componentes)")
plt.xlabel("Número de Componentes Principales")
plt.ylabel("Proporción de Varianza Acumulada")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
filepath_var_pca = os.path.join(OUTPUT_DIR, 'varianza_explicada_pca.svg')
plt.savefig(filepath_var_pca, format='svg', bbox_inches='tight')
plt.show()
print(f"-> Imagen guardada: {filepath_var_pca}")

joblib.dump(best_svm_pca, 'SVM_con_PCA.pkl')
print("¡Modelo con PCA empaquetado y guardado como 'SVM_con_PCA.pkl'!")


# ==========================================
# 6. TABLA COMPARATIVA DE HIPERPARÁMETROS
# ==========================================
print("\n" + "="*60)
print("TABLA DE MEJORES HIPERPARÁMETROS ENCONTRADOS")
print("="*60)

params_normal = grid_svm.best_params_
params_pca = grid_pca.best_params_

df_params = pd.DataFrame({
    "Hiperparámetro": [
        "SelectKBest (k)", 
        "PCA (n_components)", 
        "SVM (C)", 
        "SVM (gamma)"
    ],
    "SVM Normal (Sin PCA)": [
        params_normal.get('feature_select__k', 'N/A'),
        "No Aplica",
        params_normal.get('svm__C', 'N/A'),
        params_normal.get('svm__gamma', 'N/A')
    ],
    "SVM con PCA": [
        params_pca.get('feature_select__k', 'N/A'),
        params_pca.get('pca__n_components', 'N/A'),
        params_pca.get('svm__C', 'N/A'),
        params_pca.get('svm__gamma', 'N/A')
    ]
})

print(df_params.to_markdown(index=False, tablefmt="grid"))
print("="*60 + "\n")
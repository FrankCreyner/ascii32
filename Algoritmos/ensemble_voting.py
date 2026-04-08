import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mode, chi2


# Cargar los datasets para procesar las variables y la codificación correctas
train_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv")
test_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TEST.csv")

# Dropeamos IDs que no sirven para el modelo en los datos de entrada
train_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
test_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")

target = 'DX'

# MAPEO DE CLASES A NOMBRES PARA PLOTS Y ESTANDARIZACIÓN
def mapear_dx(val):
    val_str = str(val).strip()
    if val_str in ["4", "4.0", "NL"]: return "NL"
    if val_str in ["1", "1.0", "MCI"]: return "MCI"
    if val_str in ["0", "0.0", "AD", "Dementia"]: return "AD"
    return "OTHER"

train_df[target] = train_df[target].apply(mapear_dx)
test_df[target] = test_df[target].apply(mapear_dx)

train_df = train_df[train_df[target] != "OTHER"].reset_index(drop=True)
test_df = test_df[test_df[target] != "OTHER"].reset_index(drop=True)

le = LabelEncoder()
y_train = le.fit_transform(train_df[target])
y_test = le.transform(test_df[target]) 
etiquetas_clases = [str(c) for c in le.classes_]
n_clases = len(le.classes_)

X_test = test_df.drop(columns=[target])

def estandarizar_predicciones(preds):
    if len(preds) == 0:
        return preds
    if isinstance(preds[0], str):
        mapped = pd.Series(preds).apply(mapear_dx)
        return le.transform(mapped)
    else:
        return np.array(preds, dtype=int)

xgb_model = joblib.load("xgboost_model.pkl")
rf_model = joblib.load("modelo_random_forest.pkl")
svm_model = joblib.load("SVM.pkl")

# Predicciones discretas
pred_xgb = estandarizar_predicciones(xgb_model.predict(X_test))
pred_rf = estandarizar_predicciones(rf_model.predict(X_test))
pred_svm = estandarizar_predicciones(svm_model.predict(X_test))

# Scores para ROC
# XGB y RF arrojan probabilidades, SVM usa distances al plano si no se entrenó con probability=True
scores_xgb = xgb_model.predict_proba(X_test)
scores_rf = rf_model.predict_proba(X_test)
try:
    scores_svm = svm_model.predict_proba(X_test)
except (AttributeError, NotImplementedError):
    # Si no tiene predict_proba, usamos decision_function
    scores_svm = svm_model.decision_function(X_test)

# Hard voting
todas_las_preds = np.vstack((pred_xgb, pred_rf, pred_svm))
ensemble_pred, count = mode(todas_las_preds, axis=0, keepdims=False)

# SECCIÓN 1: EVALUACIÓN Y MÉTRICAS GLOBALES
metricas = {}
nombres = ['XGBoost', 'Random_Forest', 'SVM', 'Ensamble']
preds_list = [pred_xgb, pred_rf, pred_svm, ensemble_pred]

for nombre, preds in zip(nombres, preds_list):
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    metricas[nombre] = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    
    print(f"\nResultados para {nombre}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    if nombre == 'Ensamble':
        print(classification_report(y_test, preds, target_names=etiquetas_clases, zero_division=0))

# Plot comparativo de barras de métricas
df_metricas = pd.DataFrame(metricas).T
df_metricas.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Comparación Directa de Modelos (Métricas Globales)", fontsize=14, fontweight='bold')
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.05)
plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig("figures/08_Comparacion_Metricas.png", dpi=300)
plt.close()
print("Guardado: figures/08_Comparacion_Metricas.png")


# SECCIÓN 2: CURVAS ROC MULTICLASES INDIVIDUALES
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

def plot_roc_multiclass(y_true_bin, y_scores, title, filename):
    plt.figure(figsize=(8, 6))
    colores = ['blue', 'orange', 'green']
    
    for i, color in zip(range(n_clases), colores):
        clase_nombre = etiquetas_clases[i]
        
        # Para evitar problemas con decision_function si devolviera shapes diferentes en binario
        if len(y_scores.shape) == 1: 
            # Caso anómalo 
            score = y_scores
        else:
            score = y_scores[:, i]
            
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'Clase {clase_nombre} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Guardado ROC: {filename}")

plot_roc_multiclass(y_test_bin, scores_xgb, "Curva ROC - XGBoost", "figures/09_ROC_XGBoost.png")
plot_roc_multiclass(y_test_bin, scores_rf, "Curva ROC - Random Forest", "figures/09_ROC_Random_Forest.png")

# SVM: validar scores de decision function
# Scikit-learn OVR decision function retorna array (n_samples, 3). 
# Si tuviéramos un error lo atrapamos de forma segura para garantizar la corrida.
try:
    plot_roc_multiclass(y_test_bin, scores_svm, "Curva ROC - SVM", "figures/09_ROC_SVM.png")
except Exception as e:
    print(f"No se pudo graficar ROC para SVM: {e}")


# SECCIÓN 3: PRUEBA ESTADÍSTICA DE MCNEMAR 
print("\n" + "="*60)
print("TEST DE MCNEMAR: SIGNIFICANCIA ESTADÍSTICA DEL ENSAMBLE")

def test_mcnemar(y_true, pred_A, pred_B, nombre_A, nombre_B):
    # a: A correct, B correct
    # b: A correct, B wrong
    # c: A wrong, B correct
    # d: A wrong, B wrong
    b = np.sum((pred_A == y_true) & (pred_B != y_true))
    c = np.sum((pred_A != y_true) & (pred_B == y_true))
    
    # Continuos correction McNemar
    estadistico = ((abs(b - c) - 1.0) ** 2) / (b + c + 1e-10) # 1e-10 para evitar division por cero
    
    # p-value desde distribucion Chi2 con 1 grado de libertad
    p_value = chi2.sf(estadistico, 1)
    
    print(f"--- Comparando {nombre_A} vs {nombre_B} ---")
    print(f"Casos donde {nombre_A} acertó y {nombre_B} falló: {b}")
    print(f"Casos donde {nombre_A} falló y {nombre_B} acertó: {c}")
    print(f"P-Valor Obtenido: {p_value:.6f}")
    
    if p_value < 0.05:
        if b > c:
            print(f"ONCLUSIÓN: {nombre_A} es ESTADÍSTICAMENTE superior a {nombre_B}.")
        else:
            print(f"CONCLUSIÓN: {nombre_B} es ESTADÍSTICAMENTE superior a {nombre_A}.")
    else:
         print(f"CONCLUSIÓN: No hay diferencia estadísticamente significativa.")
    print("")

test_mcnemar(y_test, ensemble_pred, pred_xgb, "Ensamble (Voting)", "XGBoost")
test_mcnemar(y_test, ensemble_pred, pred_rf, "Ensamble (Voting)", "Random Forest")
test_mcnemar(y_test, ensemble_pred, pred_svm, "Ensamble (Voting)", "SVM")

print("\n" + "="*60)
print("Comparativas Individuales")
test_mcnemar(y_test, pred_xgb, pred_rf, "XGBoost", "Random Forest")
test_mcnemar(y_test, pred_xgb, pred_svm, "XGBoost", "SVM")
test_mcnemar(y_test, pred_rf, pred_svm, "Random Forest", "SVM")


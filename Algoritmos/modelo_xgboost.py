import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Crear bucket para figuras si no existe
os.makedirs("figures", exist_ok=True)

train_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv")
validation_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_VALIDATION.csv")
test_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TEST.csv")

#Dropeamos IDs que no sirven para el modelo
train_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
validation_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
test_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")

target = 'DX'

#MAPEO DE CLASES A NOMBRES PARA PLOTS Y ATENCIÓN A DESBALANCEO
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

#Filtramos cualquier fila que no sea estrictamente AD, MCI, NL
train_df = train_df[train_df[target] != "OTHER"].reset_index(drop=True)
validation_df = validation_df[validation_df[target] != "OTHER"].reset_index(drop=True)
test_df = test_df[test_df[target] != "OTHER"].reset_index(drop=True)


# Imprimir advertencia sobre el balanceo de clases
print("\n--- DISTRIBUCIÓN DE CLASES (TRAIN) ---")
conteo = train_df[target].value_counts(normalize=True) * 100
for clase, pct in conteo.items():
    print(f"{clase}: {pct:.2f}%")
    
print("\nATENCIÓN: Se observa un fuerte desbalance de clases natural en estos datos clínicos ")
print("Tener una mayoría de MCI significa que intrínsecamente y probabilísticamente es mucho más")
print("posible que el clasificador prediga la clase MCI de forma más confiable que las minoritarias.")


# LABEL ENCODING
# LabelEncoder ordenará alfabéticamente: AD=0, MCI=1, NL=2
le = LabelEncoder()

# FIT estrictamente en TRAIN, aplicamos en validation y test
y_train = le.fit_transform(train_df[target])
y_validation = le.transform(validation_df[target])
y_test = le.transform(test_df[target])

# Quitamos el target de X
X_train_all = train_df.drop(columns=[target])
X_validation_all = validation_df.drop(columns=[target])
X_test_all = test_df.drop(columns=[target])

class_names = [str(c) for c in le.classes_]

# Print class encoding mapping (consistent with TabPFN model)
print("\n--- MAPEO DE CLASES ---")
print(f"Class encoding: {dict(zip(class_names, range(len(class_names))))}")
print("This mapping is used by LabelEncoder for all predictions and evaluations.\n")

def encontrar_optimo_n_estimators(X_train, y_train, X_test, y_test, nombre_modelo, filename_prefix, max_estimators=500):
    print("\n" + "-" * 60)
    print(f"Buscando el número óptimo de iteraciones (Elbow) para:")
    print(f"{nombre_modelo}")
    
    #Modelo para evaluación (entrena hasta max_estimators)
    modelo_eval = xgb.XGBClassifier(
        n_estimators=max_estimators,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric=['mlogloss', 'merror'],
        use_label_encoder=False
    )
    
    #Entrenar evaluando en cada iteración
    eval_set = [(X_train, y_train), (X_test, y_test)]
    modelo_eval.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    #Obtener historial de métricas
    resultados = modelo_eval.evals_result()
    epochs = len(resultados['validation_0']['mlogloss'])
    x_axis = range(0, epochs)
    
    #Log Loss
    test_logloss = resultados['validation_1']['mlogloss']
    best_iter_logloss = np.argmin(test_logloss)
    min_logloss = test_logloss[best_iter_logloss]
    
    #Error de Clasificación (merror = 1 - accuracy)
    test_error = resultados['validation_1']['merror']
    best_iter_error = np.argmin(test_error)
    min_error = test_error[best_iter_error]
    
    print(f"Mejor Log Loss en Test: {min_logloss:.4f} (Iteración: {best_iter_logloss+1})")
    print(f"Mejor Error en Test:    {min_error:.4f} (Iteración: {best_iter_error+1})")
    
    #PLOT 1: LOG LOSS
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, resultados['validation_0']['mlogloss'], label='Train Log Loss', alpha=0.8)
    plt.plot(x_axis, resultados['validation_1']['mlogloss'], label='Test Log Loss', alpha=0.9, linewidth=2)
    plt.axvline(best_iter_logloss, color='red', linestyle='--', label=f'Óptimo Test (Iter={best_iter_logloss+1})')
    plt.title(f'XGBoost Log Loss - Método Elbow\n{nombre_modelo}', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Iteraciones (Árboles)', fontsize=11)
    plt.ylabel('Log Loss', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{filename_prefix}_Elbow_LogLoss.svg", format='svg')
    plt.close()
    
    #PLOT 2: ERROR
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, resultados['validation_0']['merror'], label='Train Error (1-Acc)', alpha=0.8)
    plt.plot(x_axis, resultados['validation_1']['merror'], label='Test Error (1-Acc)', alpha=0.9, linewidth=2)
    plt.axvline(best_iter_error, color='red', linestyle='--', label=f'Óptimo Test (Iter={best_iter_error+1})')
    plt.title(f'XGBoost Classification Error - Método Elbow\n{nombre_modelo}', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Iteraciones (Árboles)', fontsize=11)
    plt.ylabel('Classification Error', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{filename_prefix}_Elbow_Error.svg", format='svg')
    plt.close()
    
    print(f"Curvas guardadas en: figures/{filename_prefix}_Elbow_LogLoss.svg y Error.svg")
    
    # Retornamos la iteración donde el logloss fue mínimo (+1 porque n_estimators es 1-indexed)
    mejor_n_estimators = max(1, int(best_iter_logloss) + 1)
    
    print(f"Se seleccionarán {mejor_n_estimators} árboles para el entrenamiento final.\n")
    return mejor_n_estimators

def entrenar_modelo(X_train, y_train, X_test, y_test, nombre_modelo, filename_prefix, n_estimators_optimo):
    print("\n")
    print("-" * 50)
    print(f"Entrenamiento Final: {nombre_modelo} (con {n_estimators_optimo} iteraciones)")
    print("-" * 50)
    
    # Entrenamos el modelo con el x_train (TADPOLE_D1_D2_BL_RF_TRAIN.csv)
    modelo = xgb.XGBClassifier(
        n_estimators=n_estimators_optimo,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    modelo.fit(X_train, y_train)

    # Validamos con x_test (TADPOLE_D1_D2_BL_RF_TEST.csv)
    y_pred = modelo.predict(X_test)

    # Métricas Globales
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"* Exactitud (Accuracy):  {acc:.4f}")
    print(f"* Precisión (Precision): {precision:.4f}")
    print(f"* Exhaustividad(Recall): {recall:.4f}")
    print(f"* Puntuación F1 (F1):    {f1:.4f}\n")

    print(f"Reporte de Clasificación para {nombre_modelo}:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    #PLOT 1: Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False)
    plt.title(f"Matriz de Confusión\n{nombre_modelo}", fontsize=13, fontweight='bold')
    plt.xlabel('Predicción del Modelo', fontsize=11)
    plt.ylabel('Diagnóstico Clínico Real', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"figures/{filename_prefix}_Confusion_Matrix.svg", format='svg')
    plt.close()
    print(f"Guardado: figures/{filename_prefix}_Confusion_Matrix.svg")

    #PLOT 2: Importancia de Variables
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1][:30] # Mostramos solo el TOP 30 variables
    nombres_features = X_train.columns[indices]
    
    plt.figure(figsize=(30, 20))
    sns.barplot(x=importances[indices], y=nombres_features, palette="mako")
    plt.title(f"Predictibilidad - Top 30 Características\n{nombre_modelo}", fontsize=13, fontweight='bold')
    plt.xlabel("Puntuación de Importancia de XGBoost (F-score/Gini)", fontsize=11)
    plt.ylabel("Características Biomédicas", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"figures/{filename_prefix}_Feature_Importance.svg", format='svg')
    plt.close()
    print(f"Guardado: figures/{filename_prefix}_Feature_Importance.svg")

    return modelo

# ENTRENAMIENTO 1: Todas las variables 
# 1. Encontrar el óptimo usando el elbow method
optimo_all = encontrar_optimo_n_estimators(
    X_train_all, y_train, X_test_all, y_test,
    nombre_modelo="XGBoost", 
    filename_prefix="05_XGBoost_ALL"
)

# 2. Correr el modelo definitivo con ese óptimo
modelo_all = entrenar_modelo(
    X_train_all, y_train, X_test_all, y_test,
    nombre_modelo="Modelo XGBoost",
    filename_prefix="05_XGBoost_ALL",
    n_estimators_optimo=optimo_all
)

# Save the model
print("\n" + "="*60)
joblib.dump(modelo_all, "../Modelos/xgboost_model_all_features.pkl")
print("El modelo con todas las features se guardó como '../Modelos/xgboost_model_all_features.pkl'")

# ENTRENAMIENTO 2: Con las 100 features seleccionadas por TabPFN
print("\n" + "="*60)
print("ENTRENAMIENTO CON LAS 100 FEATURES SELECCIONADAS POR TABPFN")
print("="*60)

# Cargar las features seleccionadas por TabPFN
with open("../Modelos/tabpfn_selected_features.json", "r") as f:
    selected_features = json.load(f)

print(f"\nCargadas {len(selected_features)} features seleccionadas por TabPFN")
print("Estas features fueron seleccionadas usando ANOVA F-statistic para máxima discriminación")

# Filtrar los datasets para usar solo las features seleccionadas
X_train_selected = X_train_all[selected_features]
X_validation_selected = X_validation_all[selected_features]
X_test_selected = X_test_all[selected_features]

print(f"Shape del conjunto de entrenamiento: {X_train_selected.shape}")
print(f"Shape del conjunto de validación: {X_validation_selected.shape}")
print(f"Shape del conjunto de test: {X_test_selected.shape}")

# 1. Encontrar el óptimo usando el elbow method
optimo_selected = encontrar_optimo_n_estimators(
    X_train_selected, y_train, X_test_selected, y_test,
    nombre_modelo="XGBoost con 100 Features TabPFN",
    filename_prefix="05_XGBoost_TabPFN_100"
)

# 2. Correr el modelo definitivo con ese óptimo
modelo_selected = entrenar_modelo(
    X_train_selected, y_train, X_test_selected, y_test,
    nombre_modelo="Modelo XGBoost (100 Features TabPFN)",
    filename_prefix="05_XGBoost_TabPFN_100",
    n_estimators_optimo=optimo_selected
)

# Save the model with selected features
print("\n" + "="*60)
joblib.dump(modelo_selected, "../Modelos/xgboost_model_tabpfn_features.pkl")
print("El modelo con features TabPFN se guardó como '../Modelos/xgboost_model_tabpfn_features.pkl'")

# Save feature list used for this model
with open("../Modelos/xgboost_tabpfn_features_used.json", "w") as f:
    json.dump(selected_features, f, indent=4)
print("Lista de features guardada en '../Modelos/xgboost_tabpfn_features_used.json'")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print("\nSe entrenaron 2 modelos XGBoost:")
print("1. Modelo con todas las features disponibles")
print("2. Modelo con las 100 features seleccionadas por TabPFN")
print("\nAmbos modelos están guardados en ../Modelos/")
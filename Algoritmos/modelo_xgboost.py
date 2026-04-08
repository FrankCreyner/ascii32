import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib

# Crear bucket para figuras si no existe
os.makedirs("figures", exist_ok=True)

# Los datos de las variables que nos pasó Octavio
filtradas = ['FDG', 'AV45', 'CDRSB', 'ADAS13', 'MMSE', 'RAVLT_immediate',
       'RAVLT_learning', 'RAVLT_perc_forgetting', 'FAQ', 'MOCA', 'EcogPtMem',
       'EcogPtPlan', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
       'EcogSPOrgan', 'EcogSPDivatt', 'Hippocampus', 'Entorhinal',
       'EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
       'ST123CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
       'ST123TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
       'ST64TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
       'ST89SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
       'update_stamp_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
       'ST12SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST24TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST26TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST30SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST32CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST32TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST40TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST58TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST71SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST83TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
       'ST99TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 'HCI_BAIPETNMRC_09_12_16',
       'SROI_BAIPETNMRC_09_12_16', 'MCSUVRWM3_BAIPETNMRC_09_12_16',
       'CTX_LH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16',
       'CTX_RH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16',
       'IsthmusCingulate2_TS_MEAN', 'Amygdala_SV_MEAN', 'Bankssts_CV_MEAN',
       'Entorhinal_CV_MEAN', 'Fusiform_CV_MEAN', 'Hippocampus_SV_MEAN',
       'InferiorTemporal_CV_MEAN', 'MiddleTemporal_CV_MEAN'] 



train_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv")
test_df = pd.read_csv("TADPOLE_D1_D2_BL_RF_TEST.csv")


#Dropeamos IDs que no sirven para el modelo
train_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")
test_df.drop(columns=['EXAMDATE', 'PTID'], inplace=True, errors="ignore")

target = 'DX'

#MAPEO DE CLASES A NOMBRES PARA PLOTS Y ATENCIÓN A DESBALANCEO
def mapear_dx(val):
    val_str = str(val).strip()
    # Estandarizamos tanto los codigos numericos como los nombres
    if val_str in ["4", "4.0", "NL"]: return "NL"
    if val_str in ["1", "1.0", "MCI"]: return "MCI"
    if val_str in ["0", "0.0", "AD", "Dementia"]: return "AD"
    
    #Si hay clases raras residuales (ej. el grupo '2' o '3' que vimos en el error)
    return "OTHER"

train_df[target] = train_df[target].apply(mapear_dx)
test_df[target] = test_df[target].apply(mapear_dx)

#Filtramos cualquier fila que no sea estrictamente AD, MCI, NL (como vimos, había un pequeño 0.3% de ruido)
train_df = train_df[train_df[target] != "OTHER"].reset_index(drop=True)
test_df = test_df[test_df[target] != "OTHER"].reset_index(drop=True)


# Imprimir advertencia sobre el balanceo de clases
print("\n--- DISTRIBUCIÓN DE CLASES (TRAIN) ---")
conteo = train_df[target].value_counts(normalize=True) * 100
for clase, pct in conteo.items():
    print(f"{clase}: {pct:.2f}%")
    
print("\nATENCIÓN: Se observa un fuerte desbalance de clases natural en estos datos clínicos ")
print("Tener una mayoría de MCI significa que intrínsecamente y probabilísticamente es mucho más")
print("posible que el clasificador prediga la clase MCI de forma más confiable que las minoritarias.")


# Extraemos etiquetas para los plots (LabelEncoder las ordenará alfabéticamente: AD=0, MCI=1, NL=2)
le = LabelEncoder()

# FIT estrictamente en TRAIN, aplicamos en ambos
y_train = le.fit_transform(train_df[target])
y_test = le.transform(test_df[target]) 

# Quitamos el target de X
X_train_all = train_df.drop(columns=[target])
X_test_all = test_df.drop(columns=[target])

etiquetas_clases = [str(c) for c in le.classes_]

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
    plt.savefig(f"figures/{filename_prefix}_Elbow_LogLoss.png", dpi=300)
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
    plt.savefig(f"figures/{filename_prefix}_Elbow_Error.png", dpi=300)
    plt.close()
    
    print(f"Curvas guardadas en: figures/{filename_prefix}_Elbow_LogLoss.png y Error.png")
    
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
    print(classification_report(y_test, y_pred, target_names=etiquetas_clases, zero_division=0))

    #PLOT 1: Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=etiquetas_clases, 
                yticklabels=etiquetas_clases,
                cbar=False)
    plt.title(f"Matriz de Confusión\n{nombre_modelo}", fontsize=13, fontweight='bold')
    plt.xlabel('Predicción del Modelo', fontsize=11)
    plt.ylabel('Diagnóstico Clínico Real', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"figures/{filename_prefix}_Confusion_Matrix.png", dpi=300)
    plt.close()
    print(f"Guardado: figures/{filename_prefix}_Confusion_Matrix.png")

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
    plt.savefig(f"figures/{filename_prefix}_Feature_Importance.png", dpi=300)
    plt.close()
    print(f"Guardado: figures/{filename_prefix}_Feature_Importance.png")

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

# ENTRENAMIENTO 2: PCA (Análisis de Componentes Principales)
print("\n" + "="*60)
print("Añadiendo PCA (Análisis de Componentes Principales)")
print("="*60)

# 1. Estandarización de las características para PCA (promedio 0, varianza 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test_all)

# 3. Ajustar PCA inicial para evaluar la varianza explicada
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)

# Calcular varianza explicada acumulada
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Encontrar el número de componentes que explican el 95% de la varianza
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Número de componentes requeridos para explicar >95% de varianza: {n_components_95}")

# PLOT: Varianza Explicada Acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', ms=3)
plt.axhline(y=0.95, color='r', linestyle='-', label='95% de Varianza Explicada')
plt.axvline(x=n_components_95, color='r', linestyle='--')
plt.title('PCA: Varianza Explicada Acumulada', fontsize=14, fontweight='bold')
plt.xlabel('Número de Componentes Principales', fontsize=12)
plt.ylabel('Varianza Explicada Acumulada', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/06_PCA_Explained_Variance.png", dpi=300)
plt.close()
print("Guardado: figures/06_PCA_Explained_Variance.png")

# 4. Encontrar las "mejores características" analizando los 'loadings' (cargas)
loadings = pd.DataFrame(
    pca_full.components_.T[:, :3], 
    index=X_train_all.columns, 
    columns=['PC1', 'PC2', 'PC3']
)

# PLOT: Mejores características en PC1 (Componente Principal 1)
pc1_top = loadings['PC1'].abs().sort_values(ascending=False).head(20)
plt.figure(figsize=(12, 8))
sns.barplot(x=pc1_top.values, y=pc1_top.index, palette='viridis')
plt.title('PCA: Top 20 Características en el 1er Componente Principal', fontsize=14, fontweight='bold')
plt.xlabel('Carga Absoluta (Importancia)', fontsize=12)
plt.ylabel('Característica Clínica/Biológica', fontsize=12)
plt.tight_layout()
plt.savefig("figures/06_PCA_Top_Features_PC1.png", dpi=300)
plt.close()
print("Guardado: figures/06_PCA_Top_Features_PC1.png")

# 5. Transformar los datos para el entrenamiento reteniendo el 95% de varianza
pca_95 = PCA(n_components=n_components_95, random_state=42)
X_train_pca_np = pca_95.fit_transform(X_train_scaled)
X_test_pca_np = pca_95.transform(X_test_scaled)

# Convertir a DataFrames
component_names = [f"PC{i}" for i in range(1, n_components_95 + 1)]
X_train_pca = pd.DataFrame(X_train_pca_np, columns=component_names)
X_test_pca = pd.DataFrame(X_test_pca_np, columns=component_names)

# 6. Encontrar el óptimo de iteraciones para PCA
optimo_pca = encontrar_optimo_n_estimators(
    X_train_pca, y_train, X_test_pca, y_test,
    nombre_modelo=f"XGBoost con PCA ({n_components_95} Componentes)", 
    filename_prefix="06_XGBoost_PCA"
)

# 7. Modelo Definitivo PCA
modelo_pca = entrenar_modelo(
    X_train_pca, y_train, X_test_pca, y_test,
    nombre_modelo=f"Modelo XGBoost PCA (>95% Varianza)",
    filename_prefix="06_XGBoost_PCA",
    n_estimators_optimo=optimo_pca
)

#Hacemos el empaquetado del modelo normal
print("\n" + "="*60)


joblib.dump(modelo_all, "xgboost_model.pkl")
print("El modelo se guardó como 'xgboost_model.pkl'")
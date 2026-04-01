#Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

#Definir el dataframe
dfTrain = pd.read_csv('TADPOLE_D1_D2_BL_RF_TRAIN.csv')
dfTest = pd.read_csv('TADPOLE_D1_D2_BL_RF_TEST.csv')
dfTrain = dfTrain[dfTrain["DX"].isin([0, 1, 4])]
dfTest  = dfTest[dfTest["DX"].isin([0, 1, 4])]

# Mapear los valores de la columna "DX" a sus respectivas etiquetas
mapping = {
    4: "NL",
    1: "MCI",
    0: "AD"
}
dfTrain["DX"] = dfTrain["DX"].map(mapping)
dfTest["DX"]  = dfTest["DX"].map(mapping)
dfTest["DX"]

# Separar las columnas de identificación (PTID y EXAMDATE) del resto de los datos
columnas_id_train = dfTrain[["PTID", "EXAMDATE"]].copy()
dfTrain = dfTrain.drop(columns=["PTID", "EXAMDATE"])

columnas_id_test = dfTest[["PTID", "EXAMDATE"]].copy()
dfTest = dfTest.drop(columns=["PTID", "EXAMDATE"])

#Definir las variables predictoras (X) y la variable objetivo (y)
X_train = dfTrain.drop('DX', axis=1)
y_train = dfTrain['DX']

X_test = dfTest.drop('DX', axis=1)
y_test = dfTest['DX']

#Modelar el modelo de Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

#Entrenar el modelo
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#Métricas
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print(f"* Exactitud (Accuracy):   {accuracy:.4f}")
print(f"* Precisión (Precision):  {precision:.4f}")
print(f"* Exhaustividad (Recall): {recall:.4f}")
print(f"* Puntuación F1 (F1):     {f1:.4f}")

print("\nReporte de Clasificación para Modelo Random Forest con TODAS las Características:")
print(classification_report(y_test, y_pred))

#Gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test,
                                      cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusión')

feature_imp = pd.DataFrame({
    'Característica': X_train.columns,
    'Importancia':    rf.feature_importances_
}).sort_values('Importancia', ascending=False).head(15)

sns.barplot(x='Importancia', y='Característica',
            data=feature_imp, palette='viridis', ax=axes[1])
axes[1].set_title('Top 15 Características')

plt.tight_layout()
plt.show()

#Definir arbol de decisión gigante
plt.figure(figsize=(50, 20))

nombres_clases = [str(clase) for clase in y_train.unique()]

plot_tree(rf.estimators_[0], 
          feature_names=X_train.columns, 
          class_names=nombres_clases, 
          filled=True, 
          fontsize=8) # Bajamos un poco el tamaño de letra para que quepa más texto

plt.title('Árbol de Decisión Completo')

plt.savefig('arbol_gigante.png', dpi=300, bbox_inches='tight')

plt.show()
print("guardado como 'arbol_gigante.png'")
#Fin del código :D
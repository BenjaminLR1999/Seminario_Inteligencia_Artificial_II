#Practica 1. Ejercicio 4

import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar los datos desde el archivo CSV
datos = pd.read_csv('irisbin.csv')

# Separar características (X) y etiquetas (y)
X = datos.iloc[:, :-3].values
y = datos.iloc[:, -3:].values

# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para generalización)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un perceptrón multicapa (ajusta los parámetros según sea necesario)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Entrenar el modelo
mlp.fit(X_entrenamiento, y_entrenamiento)

# Realizar predicciones en el conjunto de prueba
y_pred = mlp.predict(X_prueba)

# Calcular la precisión del modelo
precision = accuracy_score(y_prueba, y_pred)
print(f'Precisión del modelo en el conjunto de prueba: {precision * 100:.2f}%')

# Validación cruzada con leave-one-out
loo = LeaveOneOut()
puntuaciones_loo = []

for indices_entrenamiento, indice_prueba in loo.split(X):
    X_entrenamiento, X_prueba = X[indices_entrenamiento], X[indice_prueba]
    y_entrenamiento, y_prueba = y[indices_entrenamiento], y[indice_prueba]

    mlp.fit(X_entrenamiento, y_entrenamiento)
    y_pred_loo = mlp.predict(X_prueba)

    precision_loo = accuracy_score(y_prueba, y_pred_loo)
    puntuaciones_loo.append(precision_loo)

precision_promedio_loo = np.mean(puntuaciones_loo)
desviacion_estandar_loo = np.std(puntuaciones_loo)

print(f'Leave-one-out - Precisión promedio: {precision_promedio_loo * 100:.2f}%, Desviación estándar: {desviacion_estandar_loo * 100:.2f}%')

# Validación cruzada con leave-p-out (puedes ajustar el número de p según sea necesario)
lpout = LeavePOut(p=5)
puntuaciones_lpout = []

for indices_entrenamiento, indices_prueba in lpout.split(X):
    X_entrenamiento, X_prueba = X[indices_entrenamiento], X[indices_prueba]
    y_entrenamiento, y_prueba = y[indices_entrenamiento], y[indices_prueba]

    mlp.fit(X_entrenamiento, y_entrenamiento)
    y_pred_lpout = mlp.predict(X_prueba)

    precision_lpout = accuracy_score(y_prueba, y_pred_lpout)
    puntuaciones_lpout.append(precision_lpout)

precision_promedio_lpout = np.mean(puntuaciones_lpout)
desviacion_estandar_lpout = np.std(puntuaciones_lpout)

print(f'Leave-p-out - Precisión promedio: {precision_promedio_lpout * 100:.2f}%, Desviación estándar: {desviacion_estandar_lpout * 100:.2f}%')

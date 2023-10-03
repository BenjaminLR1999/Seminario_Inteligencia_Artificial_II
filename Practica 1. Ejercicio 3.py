import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Funciones de activación y sus derivadas
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 2. Inicialización de pesos de la red
def inicializar_pesos(capas):
    pesos = {}
    for i in range(1, len(capas)):
        pesos[i] = np.random.randn(capas[i], capas[i-1]) * 0.01
    return pesos

# 3. Feedforward
def propagacion_hacia_adelante(X, pesos):
    activaciones = {0: X}
    zs = {}
    for i in range(1, len(pesos) + 1):
        z = np.dot(activaciones[i-1], pesos[i].T)
        zs[i] = z
        activacion = sigmoid(z)
        activaciones[i] = activacion
    return activaciones, zs

# 4. Retropropagación
def retropropagar(activaciones, zs, pesos, y):
    m = y.shape[0]
    gradientes = {}
    L = len(activaciones) - 1
    dz = activaciones[L] - y
    dw = np.dot(dz.T, activaciones[L-1]) / m
    gradientes[L] = dw
    for i in range(L-1, 0, -1):
        dz = np.dot(dz, pesos[i+1]) * sigmoid_prime(zs[i])
        dw = np.dot(dz.T, activaciones[i-1]) / m
        gradientes[i] = dw
    return gradientes

# Entrenamiento
def entrenar(X, y, capas, epocas, tasa_aprendizaje):
    pesos = inicializar_pesos(capas)
    for epoca in range(epocas):
        activaciones, zs = propagacion_hacia_adelante(X, pesos)
        gradientes = retropropagar(activaciones, zs, pesos, y)
        for i in range(1, len(pesos) + 1):
            pesos[i] -= tasa_aprendizaje * gradientes[i]
    return pesos

# Leemos los datos
datos = pd.read_csv("concentlite.csv")
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values.reshape(-1, 1)

# Entrenar
capas = [X.shape[1], 10, 1]  # por ejemplo, una capa oculta de 10 neuronas
pesos = entrenar(X, y, capas, 10000, 0.01)

# Predicción
def predecir(X, pesos):
    activaciones, _ = propagacion_hacia_adelante(X, pesos)
    return activaciones[len(capas) - 1]

y_pred = predecir(X, pesos)
plt.scatter(X[:, 0], X[:, 1], c=y_pred.reshape(-1), cmap='viridis')
plt.show()

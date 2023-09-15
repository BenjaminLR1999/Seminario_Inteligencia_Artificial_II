#Practica 1. Ejercicio 2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, TimeSeriesSplit

# Carga tu dataset (reemplaza 'dataset.csv' con tu archivo)
dataset = pd.read_csv('dataset.csv')

# Parámetros de entrada
num_partitions = int(input("Ingrese la cantidad de particiones: "))
train_ratio = float(input("Ingrese el porcentaje de patrones de entrenamiento (0-1): "))

# Algoritmo 1: Muestreo aleatorio simple
def random_sampling(dataset, num_partitions, train_ratio):
    partitions = []
    for _ in range(num_partitions):
        train_set, test_set = train_test_split(dataset, test_size=(1 - train_ratio), random_state=np.random.randint(1, 100))
        partitions.append((train_set, test_set))
    return partitions

# Algoritmo 2: Validación cruzada k-fold
def k_fold_cross_validation(dataset, num_partitions):
    partitions = []
    kf = KFold(n_splits=num_partitions)
    for train_index, test_index in kf.split(dataset):
        train_set, test_set = dataset.iloc[train_index], dataset.iloc[test_index]
        partitions.append((train_set, test_set))
    return partitions

# Algoritmo 3: Muestreo estratificado
def stratified_sampling(dataset, num_partitions, train_ratio):
    partitions = []
    sss = StratifiedShuffleSplit(n_splits=num_partitions, test_size=(1 - train_ratio), random_state=np.random.randint(1, 100))
    for train_index, test_index in sss.split(dataset, dataset['target_column']):  # Reemplaza 'target_column'
        train_set, test_set = dataset.iloc[train_index], dataset.iloc[test_index]
        partitions.append((train_set, test_set))
    return partitions

# Algoritmo 4: División por porcentaje fijo
def fixed_percentage_split(dataset, num_partitions, train_ratio):
    partitions = []
    for _ in range(num_partitions):
        train_set, test_set = train_test_split(dataset, test_size=(1 - train_ratio), shuffle=False)
        partitions.append((train_set, test_set))
    return partitions

# Algoritmo 5: Partición temporal
def temporal_split(dataset, num_partitions, train_ratio):
    partitions = []
    split_point = int(len(dataset) * train_ratio)
    for i in range(num_partitions):
        train_set = dataset.iloc[:split_point]
        test_set = dataset.iloc[split_point:]
        partitions.append((train_set, test_set))
        split_point += int(len(dataset) * train_ratio)
    return partitions

# Selecciona el algoritmo
print("Selecciona un algoritmo:")
print("1. Muestreo aleatorio simple")
print("2. Validación cruzada k-fold")
print("3. Muestreo estratificado")
print("4. División por porcentaje fijo")
print("5. Partición temporal")
algorithm_choice = int(input())

if algorithm_choice == 1:
    partitions = random_sampling(dataset, num_partitions, train_ratio)
elif algorithm_choice == 2:
    partitions = k_fold_cross_validation(dataset, num_partitions)
elif algorithm_choice == 3:
    partitions = stratified_sampling(dataset, num_partitions, train_ratio)
elif algorithm_choice == 4:
    partitions = fixed_percentage_split(dataset, num_partitions, train_ratio)
elif algorithm_choice == 5:
    partitions = temporal_split(dataset, num_partitions, train_ratio)
else:
    print("Algoritmo no válido")

# Imprimir las particiones generadas
for i, (train_set, test_set) in enumerate(partitions):
    print(f"Partición {i + 1}:")
    print(f"Tamaño del conjunto de entrenamiento: {len(train_set)}")
    print(f"Tamaño del conjunto de prueba: {len(test_set)}")

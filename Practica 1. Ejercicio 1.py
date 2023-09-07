#Practica 1. Ejercicio 1

#LIBRERIAS
import random

#VARIABLES
X1  = 0     #ENTRADA
X2  = 0     #ENTRADA
B   = 1     #BIAS
WX1 = 0     #PESO
WX2 = 0     #PESO
WXB = 0     #PESO
Y   = 0     #SALIDA
Sumatoria = 0

print("Practica 1. Ejercicio 1")
print("-- Perceptron Simple --", "\n")

X1 = float(input("Ingresa X1: "))
X2 = float(input("Ingresa X2: "))

#PESOS ENTRE -1 Y 1
WX1 = random.uniform(-1, 1)
WX2 = random.uniform(-1, 1)
WXB = random.uniform(-1, 1)

#BIAS
B = B * WXB

#SUMATORIA
Sumatoria = (X1 * WX1) + (X2 * WX2) + B

#FUNCION DE ACTIVACION
if (Sumatoria >= 0):

    Y = 1
    
else:

    Y = 0

print("Resultado Sumatoria:  ", Sumatoria)
print("Resultado Salida (Y): ", Y)

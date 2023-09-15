#Practica 1. Ejercicio 1

#DATOS ENTRADA (PRUEBA)
#X1 = 1.0145
#X2 = 1.045
#RESULTADO = -1

# LIBRERIAS
import random

# VARIABLES
Factor_Aprendizaje = 0.4;     # FACTOR DE APRENDIZAJE
Salida_Deseada     = -1;      # SALIDA DESEADA
X1                 = 0;       # ENTRADA 1
X2                 = 0;       # ENTRADA 2
B                  = 1;       # ENTRADA BIAS
W1                 = 0;       # PESO 1
W2                 = 0;       # PESO 2
WB                 = 0;       # PESO BIAS
Y                  = 0;       # SALIDA
Error              = 0;       # ERROR

print("Practica 1. Ejercicio 1");
print("-- Perceptron Simple --", "\n");

# ENTRADAS
X1 = float(input("Ingresa X1: "));
x2 = float(input("Ingresa X2: "));

# PESOS ALEATORIOS (-1, 1)
W1 = random.uniform(-1, 1);
W2 = random.uniform(-1, 1);
WB = random.uniform(-1, 1);

# BIAS
B = B * WB;

# CICLO WHILE 
while (Error != 0):

    # SUMATORIA 
    Y = (X1 * W1) + (X2 * W2) + B;

    # FUNCION DE ACTIVACION
    # Y >= 0 --> Y =  1
    # Y <  0 --> Y = -1
    if (Y >= 0):

        Y = 1

    elif (Y < 0):
    
        Y = -1

    # ERROR 
    Error = Salida_Deseada - Y;

    if (Error != 0):
        # RECALCULAR PESOS
        WB = WB + (Factor_Aprendizaje * Error * B);
        W1 = W1 + (Factor_Aprendizaje * Error * B);
        W2 = W2 + (Factor_Aprendizaje * Error * B);  

# RESULTADO: PESOS IDEALES
print("Peso W1: ", W1);
print("Peso W2: ", W2);
print("Peso WB: ", WB);

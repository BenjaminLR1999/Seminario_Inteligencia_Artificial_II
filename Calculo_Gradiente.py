#Calculo Gradiente

import sympy as sp

# Define las variables y la función
x, y = sp.symbols('x y')
f = x**2 + 2*y**2

# Calcula las derivadas parciales
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

# Crea el vector gradiente
gradiente = [df_dx, df_dy]

# Punto en el que deseas evaluar el gradiente
punto = (1, 2)

# Evalúa el gradiente en el punto
gradiente_punto = [g.evalf(subs={x: punto[0], y: punto[1]}) for g in gradiente]

# Imprime el resultado
print("Gradiente en el punto {}: {}".format(punto, gradiente_punto))

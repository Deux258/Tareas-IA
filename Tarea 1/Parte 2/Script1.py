import numpy as np

# Definimos saltos por serpientes y escaleras según el enunciado
escaleras = {6:24, 19:30, 16:37, 13:43, 40:50}
serpientes = {18:10, 49:17, 36:15, 46:25, 41:22, 23:11}
saltos = {**escaleras, **serpientes}

# Inicializamos matriz 50x50 con ceros
P = np.zeros((50,50))

for i in range(1,51):
    # Si i es casilla base de escalera o cabeza de serpiente, salto instantáneo
    if i in saltos:
        destino = saltos[i]
        # Si destino es 50, la partida acaba y el siguiente estado es 1
        j = 1 if destino==50 else destino
        P[i-1, j-1] = 1.0
        continue
    # Caso normal: lanzar dado(s) hasta fin de turno
    # Definimos contribuciones iniciales (caras 1-5 y cara 6 con posibles continuación)
    contrib = np.zeros(50)
    extra6 = 0.0
    # Caras 1 a 5
    for d in range(1,6):
        if i + d > 50:
            # No se mueve
            contrib[i-1] += 1/6
        else:
            newpos = i + d
            if newpos == 50:
                contrib[0] += 1/6  # reinicio a 1
            elif newpos in saltos:
                dest = saltos[newpos]
                j = 1 if dest==50 else dest
                contrib[j-1] += 1/6
            else:
                contrib[newpos-1] += 1/6
    # Cara 6
    if i + 6 > 50:
        # si rebasa con 6: se queda en i y turnos extra
        extra6 = 1/6
    else:
        newpos = i + 6
        if newpos == 50:
            contrib[0] += 1/6
        elif newpos in saltos:
            dest = saltos[newpos]
            j = 1 if dest==50 else dest
            contrib[j-1] += 1/6
        else:
            # Caso cara 6 normal: se va a newpos y luego turno extra
            target6 = newpos
    # Resolver recursividad de overshoot con 6:
    if extra6 > 0:
        P[i-1, :] = contrib / (1 - extra6)
    else:
        if 'target6' in locals():
            P[i-1, :] = contrib + (1/6)*P[target6-1, :]
        else:
            P[i-1, :] = contrib

# Control: normalizar filas (por redondeo)
for i in range(50):
    P[i,:] /= P[i,:].sum()

# Guardamos matriz de transición en CSV
np.savetxt("matriz_transicion.csv", P, delimiter=",")

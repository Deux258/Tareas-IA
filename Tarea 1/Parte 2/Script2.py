# %%
"""
Notebook: Tarea 1 - Punto 2
Modelo de la cadena de Markov para "Serpientes y Escaleras" (casillas 1..50)
Entrega: Jupyter-ready Python script (celdas marcadas con # %%)

Contenido:
- Construcción de la matriz de transición 50x50 (reglas explícitas)
- Cálculo de π por 3 métodos (exacto, iterativo, simulación)
- Cálculo de duración esperada de partida (simulación) y vector de visitas por partida
- Guardado de CSVs: matriz_transicion.csv, pi_vector.csv, pi_vector_simulation.csv, visitas_por_partida.csv
- Gráficas: convergencia π, histograma duración partidas, visitas por casilla
- Salidas impresas (progreso) para seguir la ejecución

Todas las librerías usadas son estándar (numpy, pandas, matplotlib).
"""

# %%
# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from math import isclose

# Para reproducibilidad en simulaciones
RANDOM_SEED = 42

# %%
# PARÁMETROS Y MAPA DE SALTOS (según tu entrada)
escaleras = {6:24, 19:30, 16:37, 13:43, 40:50}
serpientes = {18:10, 49:17, 36:15, 46:25, 41:22, 23:11}
saltos = {**escaleras, **serpientes}

N = 50  # casillas 1..50

# Convertir saltos a indices 0-based para uso interno
jump_idx = {k-1: v-1 for k,v in saltos.items()}

print("Saltos (0-based):", jump_idx)

# %%
# FUNCION: construir matriz de transición P (50x50)
# Reglas implementadas:
# - Dado justo 1..6
# - Si sale 6 y tras el movimiento no se activa escalera/serpiente, se repite el turno
# - Si sale 6 y tras movimiento activa escalera/serpiente, NO se repite
# - Si el dado hace que se pase (>50) no te mueves; si fue 6 y te pasas, repetís el turno
# - Saltos inmediatos: si la casilla final es base de escalera o cabeza de serpiente -> ir inmediatamente al destino
# - Al llegar exactamente a 50, la partida se reinicia a 1 (50->1) -> modelado como estado 50 con prob 1 a 1

# %%
def build_transition_matrix(restart_on_50=True, verbose=True):
    """
    Construye la matriz de transición P (N x N).
    Parámetros:
      restart_on_50: bool
         - True: si el destino es casilla 50 se remapea inmediatamente a casilla 1 (estado 0).
           Esto modela la cadena ergódica con reinicio y es el que usas para calcular π.
         - False: si el destino es casilla 50 se mantiene la casilla 50 (idx 49) como destino.
           Esto permite construir P_absorbente para cálculo analítico de tiempos hasta la casilla 50.
    """
    P = np.zeros((N,N), dtype=float)
    if verbose:
        print(f"Construyendo matriz de transición (50x50), restart_on_50={restart_on_50} ...")
    for i in range(N):
        if verbose and i % 10 == 0:
            print(f"  Procesando fila {i+1}/50...")
        # Si la casilla i es inicio de salto, el turno termina en la casilla destino inmediatamente
        if i in jump_idx:
            dest = jump_idx[i]
            # si destino es 49 (casilla 50) => decidir si remapear a 0 o dejar 49
            if dest == 49 and restart_on_50:
                P[i, 0] = 1.0
            else:
                P[i, dest] = 1.0
            continue
        # si la casilla es la meta (49 index)
        if i == 49:
            # si reinicio, ir a 0; si no, estado absorbente en 49
            P[i, 0 if restart_on_50 else 49] = 1.0
            continue

        contrib = np.zeros(N, dtype=float)
        prob_continue_same = 0.0
        target6 = None

        # caras 1..5
        for d in range(1,6):
            if i + d > 49:
                contrib[i] += 1/6
            else:
                newpos = i + d
                if newpos == 49:
                    if restart_on_50:
                        contrib[0] += 1/6
                    else:
                        contrib[49] += 1/6
                elif newpos in jump_idx:
                    dest = jump_idx[newpos]
                    if dest == 49 and restart_on_50:
                        contrib[0] += 1/6
                    else:
                        contrib[dest] += 1/6
                else:
                    contrib[newpos] += 1/6

        # cara 6
        if i + 6 > 49:
            prob_continue_same = 1/6
        else:
            newpos = i + 6
            if newpos == 49:
                if restart_on_50:
                    contrib[0] += 1/6
                else:
                    contrib[49] += 1/6
            elif newpos in jump_idx:
                dest = jump_idx[newpos]
                if dest == 49 and restart_on_50:
                    contrib[0] += 1/6
                else:
                    contrib[dest] += 1/6
            else:
                target6 = newpos

        if prob_continue_same > 0:
            P[i, :] = contrib / (1 - prob_continue_same)
        else:
            if target6 is None:
                P[i, :] = contrib
            else:
                P[i, :] = contrib
                # la dependencia será resuelta en una segunda pasada

    # Segunda pasada: resolver dependencias P[i] = contrib + (1/6)*P[target6]
    if verbose:
        print("Resolviendo dependencias lineales internas para filas con extra-turn proveniente de cara 6...")
    deps = []
    contribs = {}
    for i in range(N):
        if i in jump_idx or i == 49:
            continue
        contrib = np.zeros(N, dtype=float)
        prob_continue_same = 0.0
        target6 = None
        for d in range(1,6):
            if i + d > 49:
                contrib[i] += 1/6
            else:
                newpos = i + d
                if newpos == 49:
                    if restart_on_50:
                        contrib[0] += 1/6
                    else:
                        contrib[49] += 1/6
                elif newpos in jump_idx:
                    dest = jump_idx[newpos]
                    if dest == 49 and restart_on_50:
                        contrib[0] += 1/6
                    else:
                        contrib[dest] += 1/6
                else:
                    contrib[newpos] += 1/6
        if i + 6 > 49:
            prob_continue_same = 1/6
        else:
            newpos = i + 6
            if newpos == 49:
                if restart_on_50:
                    contrib[0] += 1/6
                else:
                    contrib[49] += 1/6
            elif newpos in jump_idx:
                dest = jump_idx[newpos]
                if dest == 49 and restart_on_50:
                    contrib[0] += 1/6
                else:
                    contrib[dest] += 1/6
            else:
                target6 = newpos
        if prob_continue_same > 0:
            continue
        if target6 is not None:
            deps.append((i, target6))
            contribs[i] = contrib

    # Inicializar filas dependientes con contrib y iterar sustituciones
    for i, t in deps:
        P[i, :] = contribs[i]
    max_iter = 5000
    tol = 1e-12
    for it in range(max_iter):
        maxdiff = 0.0
        for i, t in deps:
            newrow = contribs[i].copy()
            newrow += (1/6) * P[t, :]
            diff = np.max(np.abs(newrow - P[i, :]))
            if diff > 0:
                P[i, :] = newrow
            if diff > maxdiff:
                maxdiff = diff
        if maxdiff < tol:
            if verbose:
                print(f"    convergencia alcanzada en {it} iteraciones (maxdiff={maxdiff:.3e})")
            break
    else:
        print("    advertencia: iteración local no convergió completamente")

    # Normalizar filas por si hay residuos numéricos
    for i in range(N):
        s = P[i,:].sum()
        if s == 0:
            P[i,i] = 1.0
        else:
            P[i,:] = P[i,:] / s

    if verbose:
        print("Matriz de transición construida.")
    return P


# %%
# Ejecutar la construcción (verbose True para imprimir progreso)
P = build_transition_matrix(verbose=True)
# Guardar CSV
out_dir = 'csv'
os.makedirs(out_dir, exist_ok=True)
np.savetxt(os.path.join(out_dir, 'matriz_transicion.csv'), P, delimiter=',')
print('Matriz guardada en', os.path.join(out_dir, 'matriz_transicion.csv'))

# %%
# Sanity checks automáticos sobre la matriz guardada
import numpy as np
P_check = np.loadtxt(os.path.join(out_dir, 'matriz_transicion.csv'), delimiter=',')
rowsum = P_check.sum(axis=1)
print("Rows sum min/max:", rowsum.min(), rowsum.max())
if not np.allclose(rowsum, 1.0, atol=1e-9):
    raise ValueError("Alguna fila de P no suma 1.")
# Comprobar que filas de inicio-de-salto sean determinísticas al destino
for i,d in jump_idx.items():
    nz = np.where(P_check[i] > 1e-12)[0]
    if not (len(nz) == 1 and (nz[0] == (0 if d==49 and True else d) or nz[0] == d)):
        print(f"Advertencia: fila {i} no determinística en salto (nzs={nz})")
print("Sanity checks OK.")



# %%
# FUNCIONES PARA CALCULAR π

def pi_exact(P, verbose=True):
    n = P.shape[0]
    if verbose: print('Calculando π por método exacto (resolviendo sistema)...')
    A = P.T - np.eye(n)
    # sustituir última ecuación por la normalización
    A[-1,:] = 1.0
    b = np.zeros(n); b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print('LinAlgError en resolución directa:', e)
        # fallback por autovector
        vals, vecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        v = np.real(vecs[:, idx])
        pi = v / v.sum()
    # limpia pequeños negativos
    pi[pi < 0] = 0.0
    pi = pi / pi.sum()
    if verbose:
        print('π exacto calculado.')
    return pi


def pi_iterative(P, tol=1e-10, max_iters=20000, verbose=True, print_every=100):
    n = P.shape[0]
    if verbose: print('Calculando π por multiplicación iterativa (potencias)...')
    pi = np.ones(n) / n
    diffs = []
    for k in range(1, max_iters+1):
        pi_next = pi @ P
        diff = np.max(np.abs(pi_next - pi))
        diffs.append(diff)
        if verbose and (k % print_every == 0 or diff < tol):
            print(f'  iter {k:6d}, diff = {diff:.3e}')
        if diff < tol:
            if verbose: print(f'Convergió en {k} iteraciones (tol={tol}).')
            return pi_next, diffs
        pi = pi_next
    if verbose: print('No convergió en max_iters. Devolviendo último vector. diff final=', diffs[-1])
    return pi, diffs

# %%
# FUNCIONES DE SIMULACIÓN: random walk sobre P y simulación de partidas

def pi_by_random_walk(P, n_steps=200_000, verbose=True, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    current = 0
    t0 = time.time()
    report_every = max(1, n_steps // 10)
    if verbose: print(f'Simulación random walk: {n_steps} pasos...')
    for step in range(n_steps):
        counts[current] += 1
        probs = P[current]
        # protección por errores numéricos
        if not isclose(probs.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            probs = probs / probs.sum()
        current = rng.choice(n, p=probs)
        if verbose and ((step+1) % report_every == 0):
            print(f'  {step+1}/{n_steps} pasos ({(step+1)/n_steps:.0%})')
    t1 = time.time()
    if verbose: print('Simulación finalizada en', t1-t0, 's')
    return counts / counts.sum()


def play_one_game(rng):
    # Simula una partida completa bajo las reglas (retorna turnos y vector de visitas por turno iniciado)
    pos = 0  # index 0 => casilla 1
    turns = 0
    visits = np.zeros(N, dtype=np.int64)
    while True:
        visits[pos] += 1
        turns += 1
        extra = True
        while extra:
            extra = False
            d = int(rng.integers(1,7))
            if pos + d > 49:
                # se pasa: no se mueve
                if d == 6:
                    extra = True
            else:
                newpos = pos + d
                if newpos in jump_idx:
                    dest = jump_idx[newpos]
                    # si llega a 50 -> fin de partida
                    if dest == 49:
                        return turns, visits
                    pos = dest
                    # si cayó en salto, no repites (extra stays False)
                else:
                    if newpos == 49:
                        return turns, visits
                    pos = newpos
                    if d == 6:
                        extra = True
        # siguiente turno comienza en 'pos'


def simulate_games(N_games=20000, verbose=True, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    total_turns = 0
    acc_visits = np.zeros(N, dtype=np.int64)
    t0 = time.time()
    report_every = max(1, N_games // 10)
    if verbose: print(f'Simulación de {N_games} partidas (esto puede tardar)...')
    for g in range(N_games):
        t, v = play_one_game(rng)
        total_turns += t
        acc_visits += v
        if verbose and ((g+1) % report_every == 0):
            print(f'  partidas simuladas: {g+1}/{N_games} ({(g+1)/N_games:.0%}), duración promedio hasta ahora = {total_turns/(g+1):.4f} turnos')
    t1 = time.time()
    if verbose: print('Simulación de partidas finalizada en', round(t1-t0,2), 's')
    return total_turns / N_games, acc_visits / N_games

# %%
# EJECUCIÓN DE CÁLCULOS PRINCIPALES (con impresiones de proceso)
print('\n=== CÁLCULOS PRINCIPALES ===')

# 1) π exacto
pi_e = pi_exact(P, verbose=True)
np.savetxt(os.path.join(out_dir, 'pi_vector.csv'), pi_e, delimiter=',')
print('π exacto guardado en', os.path.join(out_dir, 'pi_vector.csv'))

# 2) π por potencias iterativas (imprime progreso cada 100 iter)
pi_it, diffs = pi_iterative(P, tol=1e-10, max_iters=20000, verbose=True, print_every=100)
np.savetxt(os.path.join(out_dir, 'pi_iter.csv'), pi_it, delimiter=',')
print('π iterativo guardado en', os.path.join(out_dir, 'pi_iter.csv'))

# 3) π por simulación random walk
pi_sim = pi_by_random_walk(P, n_steps=200000, verbose=True, seed=RANDOM_SEED)
np.savetxt(os.path.join(out_dir, 'pi_vector_simulation.csv'), pi_sim, delimiter=',')
print('π por simulación (random walk) guardado en', os.path.join(out_dir, 'pi_vector_simulation.csv'))

# Convergencia del random walk (checkpoints)
def pi_random_walk_convergence(P, n_steps=200_000, checkpoints=200, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    current = 0
    counts = np.zeros(n, dtype=np.int64)
    ks = np.linspace(1, n_steps, checkpoints, dtype=int)
    errors = []
    step = 0
    for k_target in ks:
        while step < k_target:
            counts[current] += 1
            probs = P[current]
            current = rng.choice(n, p=probs)
            step += 1
        est = counts / counts.sum()
        errors.append(np.max(np.abs(est - pi_e)))  # norma infinito respecto a pi_exact
    return ks, errors

ks, errors = pi_random_walk_convergence(P, n_steps=200_000, checkpoints=200)
plt.figure(figsize=(8,4))
plt.plot(ks, errors, marker='.')
plt.yscale('log')
plt.xlabel('Pasos en random walk')
plt.ylabel('max|π_est - π_exact| (L∞)')
plt.title('Convergencia: random walk hacia π_exact')
plt.grid(True)
plt.tight_layout()
plt.show()

# Comparación de las primeras 12 componentes
df_cmp = pd.DataFrame({'pi_exact': pi_e, 'pi_iter': pi_it, 'pi_sim': pi_sim})
print('\nComparación (primeras 12 componentes):')
print(df_cmp.head(12).to_string(index=True))


# %%
# GRAFICA: convergencia del método iterativo (error por iteración)
plt.figure(figsize=(8,4))
plt.plot(diffs, label='max|π(k+1)-π(k)|')
plt.yscale('log')
plt.xlabel('Iteración')
plt.ylabel('Diferencia (escala log)')
plt.title('Convergencia: multiplicación matriz-vector')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# SIMULACIÓN DE PARTIDAS: duración esperada y vector de visitas por partida
avg_turns, visitas_por_partida = simulate_games(N_games=20000, verbose=True, seed=RANDOM_SEED)
print('\nDuración media por partida (turnos):', avg_turns)
np.savetxt(os.path.join(out_dir, 'visitas_por_partida.csv'), visitas_por_partida, delimiter=',')
print('Vector visitas por partida guardado en', os.path.join(out_dir, 'visitas_por_partida.csv'))


# %%
# VERIFICACIÓN: normalizar visitas por partida y comparar con pi_exact
visitas_norm = visitas_por_partida / visitas_por_partida.sum()

# Guardar versión normalizada
pd.DataFrame(visitas_norm, columns=['visitas_norm']).to_csv(os.path.join(out_dir, 'visitas_por_partida_normalizada.csv'), index=False)

# Métricas de comparación con pi_exact
max_abs_diff = np.max(np.abs(pi_e - visitas_norm))
rmse = np.sqrt(np.mean((pi_e - visitas_norm)**2))
corr = np.corrcoef(pi_e, visitas_norm)[0,1]

print('\n=== VERIFICACIÓN: visitas_por_partida vs π_exact ===')
print(f'  Máxima diferencia absoluta: {max_abs_diff:.6e}')
print(f'  RMSE: {rmse:.6e}')
print(f'  Correlación (Pearson): {corr:.6f}')

# Mostrar primeras 12 filas
df_check = pd.DataFrame({
    'pi_exact': pi_e,
    'visitas_norm': visitas_norm,
    'diff': pi_e - visitas_norm
})
print('\nPrimeras 12 entradas de la comparación:')
print(df_check.head(12).to_string(index=True))


# %%
# CÁLCULO ANALÍTICO: construir P_abs (50 como estado absorbente) y calcular tiempo esperado con matriz fundamental
P_abs = build_transition_matrix(restart_on_50=False, verbose=False)  # 50 será absorbente ahora
# Q = submatriz 0..48
Q = P_abs[:49, :49]
I = np.eye(Q.shape[0])
try:
    Nmat = np.linalg.inv(I - Q)
    t_vec = Nmat.sum(axis=1)
    print("Tiempo esperado (analítico) hasta alcanzar casilla 50 empezando en casilla 1:", t_vec[0])
    print("Duración simulada (avg_turns):", avg_turns)
    print("Diferencia simulación - analítico:", avg_turns - t_vec[0])
except np.linalg.LinAlgError as e:
    print("No se pudo invertir I-Q (singular). Error:", e)



# %%
# GRAFICA: histograma aproximado de duraciones (opcional: simular y recolectar duraciones individuales)
# Para mostrar distribución, volvemos a simular pero guardando duraciones (con menos partidas para rapidez)
print('\nGenerando histograma de duración de partidas (simulación corta de 2000 partidas)...')
rng = np.random.default_rng(RANDOM_SEED+1)
durations = []
for _ in range(2000):
    t, _ = play_one_game(rng)
    durations.append(t)

plt.figure(figsize=(7,4))
plt.hist(durations, bins=range(1, max(durations)+2), align='left')
plt.xlabel('Turnos por partida')
plt.ylabel('Frecuencia')
plt.title('Histograma: duración de partidas (2000 sim)')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# GRAFICA: visitas por casilla (promedio por partida)
plt.figure(figsize=(10,4))
plt.bar(np.arange(1,N+1), visitas_por_partida)
plt.xlabel('Casilla')
plt.ylabel('Visitas promedio por partida')
plt.title('Visitas promedio por partida - casilla 1..50')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# %%
# RESUMEN FINAL E IMPRESIÓN DE RESULTADOS IMPORTANTES
print('\n=== RESUMEN DE RESULTADOS ===')
print('Duración media por partida (estimada):', avg_turns)
print('\nTop 10 casillas por π (probabilidad estacionaria):')
order = np.argsort(-pi_e)
for idx in order[:10]:
    print(f'  casilla {idx+1:2d}: π = {pi_e[idx]:.6f}, visitas_por_partida = {visitas_por_partida[idx]:.4f}')

print('\nLos archivos generados están en:', out_dir)
print(' - matriz_transicion.csv')
print(' - pi_vector.csv (exacto)')
print(' - pi_iter.csv (iterativo)')
print(' - pi_vector_simulation.csv (random walk)')
print(' - visitas_por_partida.csv')

# %%
# FIN
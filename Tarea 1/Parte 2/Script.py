# %% [markdown]
# # Tarea 1 - Punto 2: Serpientes y Escaleras
# ## Modelado como Cadena de Markov

# %%
# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from math import isclose
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# Para reproducibilidad en simulaciones
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# PARÁMETROS Y MAPA DE SALTOS (corregido según las reglas del juego)
escaleras = {2: 11, 6: 24, 19: 30, 13: 43, 16: 37, 40: 50}
serpientes = {18: 10, 49: 17, 36: 15, 46: 25, 41: 22, 23: 11}
saltos = {**escaleras, **serpientes}

N = 50  # casillas 1..50

# Convertir saltos a indices 0-based para uso interno
jump_dict = {k-1: v-1 for k, v in saltos.items()}

print("Saltos (0-based):", jump_dict)

# %% [markdown]
# ## Construcción de la Matriz de Transición
# 
# **Explicación de los pasos seguidos:**
# 
# 1. **Inicialización**: Creamos una matriz de 50x50 inicializada en ceros
# 2. **Casillas especiales**: Para las casillas que son inicio de serpientes o escaleras, 
#    la transición es determinística hacia el destino correspondiente
# 3. **Casilla final**: La casilla 50 es absorbente (permanece en ella misma)
# 4. **Movimientos normales**: Para cada casilla normal, calculamos las probabilidades 
#    de transición considerando:
#    - Tiros de dado de 1 a 6
#    - Regla del 6: Si sale 6 y no activa serpiente/escalera, se repite el turno
#    - Límites del tablero: Si el tiro excede la casilla 50, no se mueve
#    - Aplicación inmediata de serpientes/escaleras al caer en ellas
# 5. **Normalización**: Aseguramos que cada fila sume exactamente 1

# %%
def construir_matriz_transicion():
    """Construye la matriz de transición 50x50 para Serpientes y Escaleras"""
    P = np.zeros((N, N))
    
    for casilla_actual in range(N):
        # Si estamos en una casilla de salto, el movimiento es deterministico
        if casilla_actual in jump_dict:
            destino = jump_dict[casilla_actual]
            P[casilla_actual, destino] = 1.0
            continue
            
        # Para cada posible resultado del dado (1-6)
        for dado in range(1, 7):
            # Calcular nueva posición
            nueva_pos = casilla_actual + dado
            
            # Si nos pasamos de la casilla 50, no nos movemos
            if nueva_pos >= N:
                nueva_pos = casilla_actual
            else:
                # Aplicar serpientes/escaleras si corresponde
                nueva_pos = jump_dict.get(nueva_pos, nueva_pos)
            
            # Si sacamos 6 y no caemos en serpiente/escalera, turno extra
            if dado == 6 and nueva_pos not in jump_dict and nueva_pos < N-1:
                # Para turnos extra, la transición depende del estado resultante
                for dado_extra in range(1, 7):
                    pos_extra = nueva_pos + dado_extra
                    
                    if pos_extra >= N:
                        pos_extra = nueva_pos
                    else:
                        pos_extra = jump_dict.get(pos_extra, pos_extra)
                    
                    P[casilla_actual, pos_extra] += 1/36  # 1/6 * 1/6
            else:
                P[casilla_actual, nueva_pos] += 1/6
    
    # Normalizar las filas para que sumen 1
    for i in range(N):
        total = np.sum(P[i])
        if total > 0:
            P[i] /= total
        else:
            P[i, i] = 1.0  # Casilla absorbente
    
    return P


# %%
# Construcción de la matriz de transición
P = construir_matriz_transicion()

# Directorio de salida
out_dir = 'Tarea 1/Parte 2/csv'
os.makedirs(out_dir, exist_ok=True)

# VERIFICACIÓN Y CORRECCIÓN DE LA MATRIZ
print("=== VERIFICACIÓN DE LA MATRIZ DE TRANSICIÓN ===")

# 1. La casilla 50 (índice 49) debe reiniciar a casilla 1 (índice 0), no ser absorbente
print(f"Transición desde casilla 50 antes: {P[49]}")
P[49, :] = 0.0  # Limpia toda la fila
P[49, 0] = 1.0  # Reinicia a casilla 1 (índice 0)
print(f"Transición desde casilla 50 después: {P[49]}")

# 2. Asegurar que todas las filas sumen exactamente 1
print("\nVerificando suma de filas...")
for i in range(N):
    row_sum = np.sum(P[i])
    if not np.isclose(row_sum, 1.0, atol=1e-10):
        print(f"Fila {i+1} suma {row_sum:.10f} - normalizando")
        P[i] = P[i] / row_sum

# 3. Verificar filas problemáticas (casillas de salto)
print("\nVerificando casillas especiales:")
for casilla, destino in jump_dict.items():
    if not np.isclose(np.sum(P[casilla]), 1.0, atol=1e-10):
        print(f"Casilla {casilla+1} (salto a {destino+1}) no está normalizada")
        P[casilla, :] = 0.0
        P[casilla, destino] = 1.0

print("Verificación completada ✓")

# Guardar matriz en CSV
np.savetxt(os.path.join(out_dir, 'matriz_transicion.csv'), P, delimiter=',', fmt='%.8f')
print('Matriz guardada en', os.path.join(out_dir, 'matriz_transicion.csv'))


# %%
# Crear versión estilizada con nombres de casillas
casillas = [f"Casilla {i+1}" for i in range(N)]
df_matriz = pd.DataFrame(P, index=casillas, columns=range(1, N+1))
df_matriz.columns = [f"Casilla {i}" for i in range(1, N+1)]

# Función para aplicar estilo a la matriz
def estilo_matriz(df):
    df_estilo = df.copy()
    for col in df_estilo.columns:
        df_estilo[col] = df_estilo[col].apply(lambda x: f"{x:.6f}")
    return df_estilo

# Aplicar estilo y guardar
df_estilo = estilo_matriz(df_matriz)
df_estilo.to_csv(os.path.join(out_dir, 'matriz_transicion_estilizada.csv'))
print('Matriz estilizada guardada en', os.path.join(out_dir, 'matriz_transicion_estilizada.csv'))

print("\nPrimeras 10 filas de la matriz estilizada:")
print(df_estilo.head(10))

# %%
# FUNCIONES PARA CALCULAR π

def pi_exact(P, verbose=True):
    """Calcula π resolviendo el sistema lineal πP = π"""
    n = P.shape[0]
    if verbose: 
        print('Calculando π por método exacto (resolviendo sistema)...')
    
    # Construir el sistema: π(P - I) = 0 con ∑π = 1
    A = P.T - np.eye(n)
    A[-1, :] = 1.0  # Reemplazar última ecuación por normalización
    b = np.zeros(n)
    b[-1] = 1.0
    
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: método del autovector
        vals, vecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        v = np.real(vecs[:, idx])
        pi = v / v.sum()
    
    # Asegurar que no hay valores negativos y normalizar
    pi[pi < 0] = 0.0
    pi /= pi.sum()
    
    if verbose:
        print('π exacto calculado.')
    return pi

def pi_iterative(P, tol=1e-10, max_iters=20000, verbose=True, print_every=1000):
    """Calcula π por multiplicación iterativa de matrices"""
    n = P.shape[0]
    if verbose: 
        print('Calculando π por multiplicación iterativa (potencias)...')
        print(f'Criterio de término: diferencia máxima < {tol} o {max_iters} iteraciones')
    
    pi = np.ones(n) / n  # Distribución inicial uniforme
    diffs = []
    
    for k in range(1, max_iters+1):
        pi_next = pi @ P
        diff = np.max(np.abs(pi_next - pi))
        diffs.append(diff)
        
        if verbose and (k % print_every == 0 or diff < tol):
            print(f'  iter {k:6d}, diff = {diff:.3e}')
        
        if diff < tol:
            if verbose: 
                print(f'Convergió en {k} iteraciones (tol={tol}).')
            return pi_next, diffs, k
        
        pi = pi_next
    
    if verbose: 
        print(f'No convergió en {max_iters} iteraciones. diff final = {diffs[-1]:.3e}')
    return pi, diffs, max_iters

def pi_by_random_walk(P, n_steps=200000, tol=1e-4, checkpoints=100, 
                     verbose=True, seed=RANDOM_SEED):
    """Calcula π por simulación de random walk"""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    
    if verbose: 
        print(f'Calculando π por random walk: {n_steps} pasos...')
        print(f'Criterio de término: {n_steps} pasos o convergencia < {tol}')
    
    counts = np.zeros(n, dtype=np.int64)
    current = 0
    errors = []
    check_step = n_steps // checkpoints
    
    for step in range(n_steps):
        counts[current] += 1
        probs = P[current]
        
        # Protección contra errores numéricos
        if not isclose(probs.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            probs = probs / probs.sum()
        
        current = rng.choice(n, p=probs)
        
        # Verificar convergencia periódicamente
        if (step + 1) % check_step == 0:
            pi_est = counts / counts.sum()
            error = np.max(np.abs(pi_est - pi_est_prev)) if step > check_step else 1.0
            errors.append(error)
            
            if verbose and error < tol:
                print(f'Convergencia alcanzada en {step+1} pasos (error={error:.3e})')
                break
                
        pi_est_prev = counts / counts.sum() if step > 0 else np.zeros(n)
    
    pi_est = counts / counts.sum()
    return pi_est, errors

# %%
# CÁLCULO DE π POR LOS TRES MÉTODOS
print('\n=== CÁLCULO DEL VECTOR π ===')

# 1) π exacto
pi_e = pi_exact(P, verbose=True)
np.savetxt(os.path.join(out_dir, 'pi_vector.csv'), pi_e, delimiter=',', fmt='%.8f')
print('π exacto guardado en', os.path.join(out_dir, 'pi_vector.csv'))

# 2) π por multiplicación iterativa
pi_it, diffs, n_iter = pi_iterative(P, tol=1e-10, max_iters=20000, verbose=True, print_every=1000)
np.savetxt(os.path.join(out_dir, 'pi_iter.csv'), pi_it, delimiter=',', fmt='%.8f')
print('π iterativo guardado en', os.path.join(out_dir, 'pi_iter.csv'))

# 3) π por random walk
pi_sim, errors = pi_by_random_walk(P, n_steps=200000, tol=1e-4, verbose=True, seed=RANDOM_SEED)
np.savetxt(os.path.join(out_dir, 'pi_vector_simulation.csv'), pi_sim, delimiter=',', fmt='%.8f')
print('π por simulación guardado en', os.path.join(out_dir, 'pi_vector_simulation.csv'))

# VERIFICACIÓN DE RESULTADOS
print('\n=== VERIFICACIÓN DE π ===')
print(f"Suma de π exacto: {np.sum(pi_e):.10f}")
print(f"Valores negativos en π exacto: {np.sum(pi_e < 0)}")
print(f"Mínimo valor en π exacto: {np.min(pi_e):.10f}")
print(f"Máximo valor en π exacto: {np.max(pi_e):.10f}")


# %%
# GRÁFICAS DE CONVERGENCIA
plt.figure(figsize=(12, 5))

# Convergencia del método iterativo
plt.subplot(1, 2, 1)
plt.plot(diffs)
plt.yscale('log')
plt.xlabel('Iteración')
plt.ylabel('Diferencia (log)')
plt.title('Convergencia: Método Iterativo')
plt.grid(True)

# Convergencia del random walk
plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, 200000, len(errors)), errors)
plt.yscale('log')
plt.xlabel('Pasos de simulación')
plt.ylabel('Error (log)')
plt.title('Convergencia: Random Walk')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'convergencia_pi.png'))
plt.show()

# %%
# COMPARACIÓN DE LOS VECTORES π
print('\n=== COMPARACIÓN DE LOS VECTORES π ===')

# Crear DataFrame comparativo
df_comparacion = pd.DataFrame({
    'Casilla': range(1, N+1),
    'π Exacto': pi_e,
    'π Iterativo': pi_it,
    'π Simulación': pi_sim
})

# Calcular diferencias
df_comparacion['Diff Exacto-Iterativo'] = np.abs(pi_e - pi_it)
df_comparacion['Diff Exacto-Simulación'] = np.abs(pi_e - pi_sim)

print("Principales estadísticas de comparación:")
print(f"Diferencia máxima entre exacto e iterativo: {df_comparacion['Diff Exacto-Iterativo'].max():.2e}")
print(f"Diferencia máxima entre exacto y simulación: {df_comparacion['Diff Exacto-Simulación'].max():.2e}")
print(f"Correlación entre exacto e iterativo: {np.corrcoef(pi_e, pi_it)[0,1]:.6f}")
print(f"Correlación entre exacto y simulación: {np.corrcoef(pi_e, pi_sim)[0,1]:.6f}")

# %%
# SIMULACIÓN DE PARTIDAS PARA DURACIÓN Y VISITAS
def simular_partidas(n_partidas=10000, verbose=True, seed=RANDOM_SEED):
    """Simula partidas completas para calcular duración y visitas por casilla"""
    rng = np.random.default_rng(seed)
    duraciones = []
    visitas_totales = np.zeros(N, dtype=np.int64)
    
    if verbose:
        print(f'Simulando {n_partidas} partidas...')
    
    for i in range(n_partidas):
        pos = 0  # Comienza en la casilla 1
        turnos = 0
        visitas = np.zeros(N, dtype=np.int64)
        
        while pos != 49:  # Hasta llegar a la casilla 50 (índice 49)
            visitas[pos] += 1
            turnos += 1
            
            # Tirar el dado
            dado = rng.integers(1, 7)
            nueva_pos = pos + dado
            
            # Verificar límites del tablero
            if nueva_pos >= N:
                nueva_pos = pos  # No se mueve si excede el tablero
            else:
                # Aplicar serpientes/escaleras
                nueva_pos = jump_dict.get(nueva_pos, nueva_pos)
            
            pos = nueva_pos
        
        duraciones.append(turnos)
        visitas_totales += visitas
    
    duracion_promedio = np.mean(duraciones)
    visitas_promedio = visitas_totales / n_partidas
    
    return duracion_promedio, visitas_promedio, duraciones

# %%
# Simular partidas
duracion_esperada, visitas_por_partida, todas_duraciones = simular_partidas(
    n_partidas=10000, verbose=True, seed=RANDOM_SEED)

print(f'\nDuración esperada de una partida: {duracion_esperada:.2f} turnos')
print("Interpretación: En promedio, una partida de Serpientes y Escaleras se completa en aproximadamente", round(duracion_esperada), "turnos.")

# Guardar resultados
np.savetxt(os.path.join(out_dir, 'duracion_esperada.txt'), [duracion_esperada])
np.savetxt(os.path.join(out_dir, 'visitas_por_partida.csv'), visitas_por_partida, delimiter=',')

# %%
# ANÁLISIS DEL VECTOR DE VISITAS
print('\n=== VECTOR DE VISITAS POR PARTIDA ===')
print("Este vector representa el número promedio de veces que se visita cada casilla en una partida completa (hasta llegar a la casilla 50).")

# Dos observaciones sobre el vector de visitas
print("\nOBSERVACIONES SOBRE EL VECTOR DE VISITAS:")
print("1. Las casillas con serpientes tienen menos visitas porque se abandona rápidamente la casilla al ser enviado a otra posición.")
print("2. Las casillas justo antes de escaleras largas tienen más visitas porque los jugadores tienden a pasar por ellas múltiples veces.")

# Gráfica de visitas por casilla
plt.figure(figsize=(12, 5))
plt.bar(range(1, N+1), visitas_por_partida)
plt.xlabel('Casilla')
plt.ylabel('Visitas promedio por partida')
plt.title('Visitas promedio por casilla en una partida')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'visitas_por_casilla.png'))
plt.show()

# %%
# HISTOGRAMA DE DURACIÓN DE PARTIDAS
plt.figure(figsize=(10, 5))
plt.hist(todas_duraciones, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(duracion_esperada, color='red', linestyle='--', 
           label=f'Media: {duracion_esperada:.2f} turnos')
plt.xlabel('Duración de partida (turnos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de la duración de las partidas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'distribucion_duracion_partidas.png'))
plt.show()

# %%
# RESULTADOS FINALES
print('\n=== RESUMEN DE RESULTADOS ===')
print(f"Duración esperada de una partida: {duracion_esperada:.2f} turnos")

print("\nTop 5 casillas por probabilidad estacionaria:")
top_casillas = np.argsort(pi_e)[-5:][::-1]
for i, idx in enumerate(top_casillas):
    print(f"{i+1}. Casilla {idx+1}: π = {pi_e[idx]:.4f}")

print("\nTop 5 casillas más visitadas por partida:")
top_visitas = np.argsort(visitas_por_partida)[-5:][::-1]
for i, idx in enumerate(top_visitas):
    print(f"{i+1}. Casilla {idx+1}: {visitas_por_partida[idx]:.2f} visitas/partida")

print(f"\nTodos los archivos de resultados guardados en la carpeta '{out_dir}'")

# %%
# FIN
print('Notebook completado.')
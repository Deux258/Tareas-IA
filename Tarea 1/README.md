# Tarea 1 - Inteligencia Artificial

**Integrantes:** Diego Muñoz Barra y Aarón Pozas Oyarce

## Descripción General

Este repositorio contiene la solución a la Tarea 1 de IA. La tarea está dividida en dos partes. El objetivo es realizar simulaciones de cadenas de Markov y análisis de resultados relacionados.

1. **Parte 1:** Aprendizaje y validación de modelos de redes bayesianas sobre el dataset "Adult" de UCI, incluyendo el análisis del impacto del aumento de datos.
2. **Parte 2:** Modelado y análisis probabilístico del juego "Serpientes y Escaleras" como una cadena de Markov, para calcular simulaciones y visualizaciones.


## Estructura 

```
>>>>>>> f81b976beab3265e4221bacf273fac964489494c
Tareas-IA/
└── Tarea 1/
    ├── Parte 1/
    │   └── T1_IA_Punto2.ipynb
    ├── Parte 2/
    │   └── Script.py
    ├── csv/
    │   ├── matriz_transicion.csv
    │   ├── pi_iter.csv
    │   ├── pi_vector_simulation.csv
    │   └── pi_vector.csv
    │   ├── convergencia_pi.png
    │   ├── distribucion_duracion_partidas.png
    │   └── visitas_por_casilla.png
    │   └── visitas_por_partida.csv
    └── README.md
```

# Requerimientos

Para ejecutar el código, asegúrate de tener las siguientes bibliotecas instaladas:

```
pip install -r requirements.txt
```

# Parte 1: Redes Bayesianas y Aumento de Datos

Archivo principal: Parte 1/Tarea_IA.ipynb

### Objetivos

Aprender a construir una red bayesiana usando el dataset "Adult" con dos algoritmos: GES (Greedy Equivalence Search) y Hill Climb Search, entrenar modelos bayesianos y hacer inferencias probabilísticas y validar las inferencias con datos de prueba. Por último pero no menos importante, ver cómo afecta aumentar el dataset en un 50% usando resampling.

###Flujo de trabajo

1. Primero, instalamos las librerías necesarias: pandas, sklearn, pgmpy, matplotlib, etc.
2. Bajamos el dataset "Adult" desde UCI y lo limpiamos para quedarnos con 32,561 filas y 15 columnas, que es lo que necesitamos.
3. Dividimos el dataset en entrenamiento (70%) y prueba (30%) usando train_test_split, para poder hacer validaciones después.
4. Aprendizaje de la Estructura
    - GES: Usamos el algoritmo GES para aprender la estructura de la red bayesiana, basándonos en la métrica BIC.
    - Hill Climb Search: Hacemos lo mismo con el algoritmo Hill Climb Search, también utilizando la métrica BIC.

5. Entrenamiento de Modelos Bayesianos: Para esto, filtramos las aristas y nodos que no aportan nada, para evitar ciclos y nodos aislados en la red. Luego entrenamos los modelos bayesianos discretos para cada una de las estructuras que aprendimos antes.
6. Hacemos inferencias sobre variables clave como income, sex y occupation, probando diferentes evidencias y luego comparamos los resultados entre ambos modelos (GES y Hill Climb).
7. Validamos la precisión de las inferencias comparando las probabilidades que el modelo predijo con las observadas en el conjunto de prueba.
8. Aumentamos el dataset en un 50% mediante resampling. Aumentado el dataset, repetimos todo el proceso de aprendizaje, entrenamiento, inferencia y validación con el nuevo dataset aumentado.

Comparación de Modelos

Finalmente, comparamos los resultados de los modelos entrenados con los datos originales y los aumentados. Analizamos cómo impactó el aumento de datos en la precisión y robustez de los modelos.

# Parte 2: Cadena de Markov con juego de Serpientes y Escaleras

Archivo principal: Parte 2/T1_IA_Punto2.ipynb

### Objetivos

Modelar el juego de "Serpientes y Escaleras" como una cadena de Markov con 50 estados. Para esto, se constuye la matriz de transición considerando las reglas especiales del juego (como las escaleras, serpientes, turnos extra, y rebases) .
Calcular la distribución estacionaria π usando tres métodos: exacto, iterativo y simulación. Simular partidas para analizar la duración esperada y las visitas promedio por casilla y por último, viisualizar los resultados y guardar los archivos con los análisis realizados.

### Flujo de trabajo

1. Se definen las posiciones de las escaleras y las serpientes, además de las reglas del juego como los turnos extra, los rebases y si el juego tiene un estado absorbente o si reinicia al llegar a la casilla 50.

2. Se crea una función que arma la matriz de transición de 50x50, aplicando todas las reglas del juego como escaleras y serpientes.

3. Verificamos que la matriz cumple con las propiedades de una cadena de Markov (es decir, la suma de las filas es igual a 1) y luego se guarda en un archivo CSV llamado csv/matriz_transicion.csv.

4. Se calcula la distribución estacionaria π usando tres métodos diferentes:
    - Exacto: Resolviendo el sistema lineal.
    - Iterativo: Aplicando la multiplicación iterativa (potencias de la matriz).
    - Simulación: A través de un random walk.

Los resultados se guardan en archivos CSV (pi_vector.csv, pi_iter.csv, pi_vector_simulation.csv).

5. Se simulan 10,000 partidas para calcular la duración esperada de las partidas y obtener el vector de visitas promedio por casilla.
Los resultados se guardan en un archivo CSV llamado csv/visitas_por_partida.csv.

6. Se generan gráficos para analizar los resultados, como:  
    - Gráfico de convergencia de π (convergencia_pi.png).
    - Gráfico de distribución de duración de partidas (distribucion_duracion_partidas.png).
    - Gráfico de visitas por casilla (visitas_por_casilla.png).

7. Comparamos los resultados obtenidos con los tres métodos de cálculo de la distribución estacionaria π.
    - Para esto, se analizan las casillas más visitadas durante las simulaciones y se comparan con las casillas de mayor probabilidad estacionaria.
    - También se analiza la frecuencia de las caras del dado durante las simulaciones.
  
### Archivos de resultados 

- Todos los archivos generados por la simulación y el análisis se encuentran en la carpeta `Parte 2/csv/`:
    - matriz_transicion.csv: Matriz de transición del juego de Serpientes y Escaleras.
    - pi_vector.csv, pi_iter.csv, pi_vector_simulation.csv: Vectores de π calculados por los diferentes métodos.
    - visitas_por_partida.csv: Vector con la cantidad promedio de visitas por casilla.
    - convergencia_pi.png, distribucion_duracion_partidas.png, visitas_por_casilla.png: Gráficas con el análisis visual de los resultados.



## Referencias

- [pgmpy Documentation](https://pgmpy.org/index.html)
- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

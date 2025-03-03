import heapq
import time
import math
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np

# -------------------------------------------------------
# 1. Funciones para cargar y procesar el laberinto
# -------------------------------------------------------
def cargar_laberinto(nombre_archivo):
    with open(nombre_archivo, 'r') as file:
        laberinto = [list(line.strip()) for line in file if line.strip()]
    # Rellenar filas que tengan menos columnas (con '1' = pared)
    if laberinto:
        max_len = max(len(fila) for fila in laberinto)
        for i in range(len(laberinto)):
            if len(laberinto[i]) < max_len:
                laberinto[i].extend(['1'] * (max_len - len(laberinto[i])))
    return laberinto

def encontrar_posiciones(laberinto, valor):
    posiciones = []
    for fila in range(len(laberinto)):
        for col in range(len(laberinto[0])):
            if laberinto[fila][col] == valor:
                posiciones.append((fila, col))
    return posiciones

def distancia_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def distancia_euclidiana(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def movimientos_validos(laberinto, pos):
    filas, columnas = len(laberinto), len(laberinto[0])
    # Orden: Arriba, Derecha, Abajo, Izquierda
    movs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    vecinos = []
    for dx, dy in movs:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < filas and 0 <= ny < columnas and laberinto[nx][ny] != '1':
            vecinos.append((nx, ny))
    return vecinos

def reconstruir_camino(padres, inicio, fin):
    camino = []
    actual = fin
    while actual != inicio:
        camino.append(actual)
        actual = padres[actual]
    camino.append(inicio)
    return camino[::-1]

# -------------------------------------------------------
# 2. Generadores de animación para cada algoritmo
# -------------------------------------------------------
def bfs_anim(laberinto, inicio, fin):
    cola = deque([inicio])
    padres = {inicio: None}
    explorados = set()
    yield None, list(cola), explorados  # Estado inicial
    while cola:
        nodo = cola.popleft()
        explorados.add(nodo)
        yield nodo, list(cola), explorados
        if nodo == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            yield nodo, list(cola), explorados, camino
            return camino, len(explorados)
        for vecino in movimientos_validos(laberinto, nodo):
            if vecino not in padres:
                padres[vecino] = nodo
                cola.append(vecino)
    yield None, list(cola), explorados
    return None, len(explorados)

def dfs_anim(laberinto, inicio, fin):
    pila = [inicio]
    padres = {inicio: None}
    explorados = set()
    yield None, list(pila), explorados
    while pila:
        nodo = pila.pop()
        explorados.add(nodo)
        yield nodo, list(pila), explorados
        if nodo == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            yield nodo, list(pila), explorados, camino
            return camino, len(explorados)
        for vecino in movimientos_validos(laberinto, nodo):
            if vecino not in padres:
                padres[vecino] = nodo
                pila.append(vecino)
    yield None, list(pila), explorados
    return None, len(explorados)

def greedy_anim(laberinto, inicio, fin, heuristica):
    heap = [(heuristica(inicio, fin), inicio)]
    padres = {inicio: None}
    explorados = set()
    yield None, [item[1] for item in heap], explorados
    while heap:
        _, nodo = heapq.heappop(heap)
        explorados.add(nodo)
        yield nodo, [item[1] for item in heap], explorados
        if nodo == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            yield nodo, [item[1] for item in heap], explorados, camino
            return camino, len(explorados)
        for vecino in movimientos_validos(laberinto, nodo):
            if vecino not in padres:
                padres[vecino] = nodo
                heapq.heappush(heap, (heuristica(vecino, fin), vecino))
    yield None, [item[1] for item in heap], explorados
    return None, len(explorados)

def a_star_anim(laberinto, inicio, fin, heuristica):
    heap = [(0, inicio)]
    g_score = {inicio: 0}
    padres = {inicio: None}
    explorados = set()
    yield None, [item[1] for item in heap], explorados
    while heap:
        _, nodo = heapq.heappop(heap)
        explorados.add(nodo)
        yield nodo, [item[1] for item in heap], explorados
        if nodo == fin:
            camino = reconstruir_camino(padres, inicio, fin)
            yield nodo, [item[1] for item in heap], explorados, camino
            return camino, len(explorados)
        for vecino in movimientos_validos(laberinto, nodo):
            temp_g = g_score[nodo] + 1
            if vecino not in g_score or temp_g < g_score[vecino]:
                g_score[vecino] = temp_g
                f_score = temp_g + heuristica(vecino, fin)
                padres[vecino] = nodo
                heapq.heappush(heap, (f_score, vecino))
    yield None, [item[1] for item in heap], explorados
    return None, len(explorados)

# -------------------------------------------------------
# 3. Función para crear la imagen (frame) para la animación
# -------------------------------------------------------
def crear_frame(laberinto, explorados, frontera, current, camino=None):
    """
    Crea una matriz numérica para visualizar el laberinto.
    Se asignan los siguientes valores:
      0: pared ('1')           → negro
      1: camino libre ('0')     → blanco
      2: nodo en frontera       → azul
      3: nodo explorado         → gris
      4: camino solución        → rojo
      5: punto de partida ('2') → verde
      6: punto de salida ('3')  → amarillo
      7: nodo actual            → naranja
    """
    filas = len(laberinto)
    cols = len(laberinto[0])
    frame = np.zeros((filas, cols))
    
    # Base: asignar paredes y caminos
    for i in range(filas):
        for j in range(cols):
            if laberinto[i][j] == '1':
                frame[i, j] = 0
            else:
                frame[i, j] = 1
            if laberinto[i][j] == '2':
                frame[i, j] = 5
            if laberinto[i][j] == '3':
                frame[i, j] = 6
    
    # Marcar nodos explorados (sin sobreescribir inicio/fin)
    for (i, j) in explorados:
        if laberinto[i][j] in ['0', '2', '3']:
            frame[i, j] = 3
    # Marcar frontera
    for (i, j) in frontera:
        if laberinto[i][j] in ['0', '2', '3']:
            frame[i, j] = 2
    # Marcar nodo actual
    if current is not None:
        frame[current[0], current[1]] = 7
    # Marcar camino solución, si existe (no sobreescribe inicio/fin)
    if camino:
        for (i, j) in camino:
            if (i, j) not in [current]:
                frame[i, j] = 4
    return frame

# Definir colormap: índices
# 0 -> negro, 1 -> blanco, 2 -> azul, 3 -> gris, 4 -> rojo, 5 -> verde, 6 -> amarillo, 7 -> naranja
cmap = mcolors.ListedColormap(['black', 'white', 'blue', 'gray', 'red', 'green', 'yellow', 'orange'])

# -------------------------------------------------------
# 4. Programa principal con selección y animación en vivo
# -------------------------------------------------------
def main():
    archivo = input("Ingrese el nombre (ruta) del laberinto (e.g., 'Laberinto1.txt'): ").strip()
    laberinto = cargar_laberinto(archivo)
    if not laberinto:
        print("Error: Laberinto vacío o no se pudo cargar.")
        return
    
    # Mostrar dimensiones y, opcionalmente, la visualización estática
    print(f"Laberinto cargado: {len(laberinto)} filas x {len(laberinto[0])} columnas.")
    
    inicio_pos = encontrar_posiciones(laberinto, '2')
    salida_pos = encontrar_posiciones(laberinto, '3')
    if not inicio_pos or not salida_pos:
        print("No se encontraron punto de inicio ('2') o salida ('3').")
        return
    inicio = inicio_pos[0]
    fin = salida_pos[0]
    
    # Seleccionar algoritmo
    print("\nSeleccione el algoritmo a animar:")
    print("1. BFS")
    print("2. DFS")
    print("3. Greedy (Manhattan)")
    print("4. A* (Manhattan)")
    opcion = input("Ingrese el número de la opción: ").strip()
    
    if opcion == "1":
        generador = bfs_anim(laberinto, inicio, fin)
        alg = "BFS"
    elif opcion == "2":
        generador = dfs_anim(laberinto, inicio, fin)
        alg = "DFS"
    elif opcion == "3":
        generador = greedy_anim(laberinto, inicio, fin, distancia_manhattan)
        alg = "Greedy (Manhattan)"
    elif opcion == "4":
        generador = a_star_anim(laberinto, inicio, fin, distancia_manhattan)
        alg = "A* (Manhattan)"
    else:
        print("Opción inválida, usando BFS por defecto.")
        generador = bfs_anim(laberinto, inicio, fin)
        alg = "BFS"
    
    # Configurar animación
    fig, ax = plt.subplots(figsize=(12, 6))
    frame_inicial = crear_frame(laberinto, set(), [], None)
    im = ax.imshow(frame_inicial, cmap=cmap, vmin=0, vmax=7, origin='upper')
    ax.set_title(f"Animación de {alg}")
    ax.axis('off')
    
    def update(frame):
        try:
            resultado = next(generador)
        except StopIteration:
            return im,
        # El generador yield 3 o 4 valores
        if len(resultado) == 3:
            current, frontier, explorados = resultado
            camino = None
        else:
            current, frontier, explorados, camino = resultado
        imagen = crear_frame(laberinto, explorados, frontier, current, camino)
        im.set_data(imagen)
        return im,
    
    ani = animation.FuncAnimation(fig, update, interval=200, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
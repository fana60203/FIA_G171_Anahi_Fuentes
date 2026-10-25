"""
Actividad 2.4 - Algoritmos de búsqueda ciega (BFS / DFS)
Fundamentos de Inteligencia Artificial - Politécnica de Atlautla

Objetivo:
    Resolver un laberinto en cuadrícula (movimientos Arriba, Derecha, Abajo,
    Izquierda) con Búsqueda en Anchura (BFS) y Búsqueda en Profundidad (DFS),
    y comparar: tiempo, nodos expandidos, longitud del camino y memoria
    (tamaño máximo de la frontera).

Uso:
    python busqueda_ciega.py

Salida:
    - Resultados impresos en consola.
    - Imágenes y animación (animacion_bfs_dfs.gif) en la carpeta ../img/
    - Archivo resultados.json en esta misma carpeta.
"""

# ----------------------------- Importaciones -------------------------------
import json                     # Guardar los resultados en un archivo .json
import random                   # Generar el laberinto de forma pseudoaleatoria
import statistics               # Calcular la mediana de los tiempos medidos
import time                     # Medir el tiempo de ejecución (perf_counter)
from collections import deque   # Cola eficiente (FIFO) para BFS
from pathlib import Path        # Manejo de rutas de archivos

import matplotlib
matplotlib.use("Agg")           # Backend sin ventana: solo guarda imágenes
import matplotlib.pyplot as plt  # Gráficas
import networkx as nx           # Grafo auxiliar para validar los resultados
from matplotlib.animation import FuncAnimation, PillowWriter  # Animación (GIF)
from matplotlib.colors import ListedColormap  # Colores pared/libre

# Carpeta donde se guardan las imágenes (../img respecto a este archivo)
IMG = Path(__file__).resolve().parent.parent / "img"
IMG.mkdir(exist_ok=True)        # Se crea si no existe

# Acciones posibles: (cambio en fila, cambio en columna, nombre).
# El orden es fijo para que los resultados sean reproducibles.
ACCIONES = [(-1, 0, "Arriba"), (0, 1, "Derecha"),
            (1, 0, "Abajo"), (0, -1, "Izquierda")]


# ---------------------------------------------------------------------------
# 1. Generación del laberinto (datos de entrada)
# ---------------------------------------------------------------------------
def generar_laberinto(n_celdas=10, semilla=7, bucles=0.15):
    """Crea un laberinto de (2n+1) x (2n+1). 1 = pared, 0 = celda libre.

    Paso 1: laberinto "perfecto" (sin ciclos) con el algoritmo backtracker.
    Paso 2: se abre una fracción `bucles` de paredes internas para crear
            ciclos; así hay varios caminos posibles y BFS y DFS pueden
            diferir en la calidad de la solución.
    Devuelve: (cuadrícula, celda_inicial, celda_meta)
    """
    rnd = random.Random(semilla)        # Generador con semilla => reproducible
    N = 2 * n_celdas + 1                # Tamaño total de la cuadrícula
    g = [[1] * N for _ in range(N)]     # Se empieza con todo pared

    pila = [(1, 1)]                     # Pila del backtracker (celda actual arriba)
    vis = {(1, 1)}                      # Celdas ya visitadas
    g[1][1] = 0                         # La celda inicial queda libre

    while pila:
        r, c = pila[-1]                 # Celda en la que estamos
        # Vecinos a 2 pasos (saltando la pared intermedia) aún no visitados
        vec = [(r + dr * 2, c + dc * 2, dr, dc) for dr, dc, _ in ACCIONES
               if 0 < r + dr * 2 < N - 1 and 0 < c + dc * 2 < N - 1
               and (r + dr * 2, c + dc * 2) not in vis]
        if not vec:                     # Sin vecinos nuevos: retroceder
            pila.pop()
            continue
        nr, nc, dr, dc = rnd.choice(vec)  # Elegir un vecino al azar
        g[r + dr][c + dc] = 0           # Abrir la pared intermedia
        g[nr][nc] = 0                   # Abrir la celda vecina
        vis.add((nr, nc))               # Marcarla como visitada
        pila.append((nr, nc))           # Avanzar hacia ella

    # Paredes internas que separan dos celdas libres: candidatas a abrirse
    cand = [(r, c) for r in range(1, N - 1) for c in range(1, N - 1)
            if g[r][c] == 1 and ((g[r - 1][c] == 0 and g[r + 1][c] == 0) or
                                 (g[r][c - 1] == 0 and g[r][c + 1] == 0))]
    rnd.shuffle(cand)                   # Orden aleatorio reproducible
    for r, c in cand[: int(len(cand) * bucles)]:
        g[r][c] = 0                     # Se abre la pared => se crea un ciclo
    # Inicio en la esquina superior izquierda, meta en la inferior derecha
    return g, (1, 1), (N - 2, N - 2)


# ---------------------------------------------------------------------------
# 2. Representación del grafo
# ---------------------------------------------------------------------------
def construir_grafo(grid):
    """Construye la lista de adyacencia {(fila, col): [vecinos...]}.

    Nodo   = celda libre.
    Arista = dos celdas libres adyacentes (costo 1).
    """
    filas, cols = len(grid), len(grid[0])
    ady = {}
    for r in range(filas):
        for c in range(cols):
            if grid[r][c] == 0:         # Solo las celdas libres son nodos
                # Vecinos: celdas dentro de la cuadrícula y libres
                ady[(r, c)] = [(r + dr, c + dc) for dr, dc, _ in ACCIONES
                               if 0 <= r + dr < filas and 0 <= c + dc < cols
                               and grid[r + dr][c + dc] == 0]
    return ady


def a_networkx(ady):
    """Convierte la lista de adyacencia en un grafo de NetworkX (para validar)."""
    G = nx.Graph()
    for u, vs in ady.items():
        for v in vs:
            G.add_edge(u, v)            # Cada arista se agrega una sola vez
    return G


# ---------------------------------------------------------------------------
# 3. Algoritmos de búsqueda ciega
# ---------------------------------------------------------------------------
def reconstruir(padre, meta):
    """Reconstruye el camino inicio->meta siguiendo los padres desde la meta."""
    camino, n = [], meta
    while n is not None:                # El inicio tiene padre None
        camino.append(n)
        n = padre[n]
    return camino[::-1]                 # Se invierte: de inicio a meta


def bfs(ady, inicio, meta):
    """Búsqueda en anchura (BFS): la frontera es una COLA (FIFO).

    Explora por niveles: primero todos los nodos a distancia 1, luego 2, etc.
    Con costo uniforme encuentra el camino más corto (es óptimo).
    Devuelve: (camino, orden_de_expansión, tamaño_máximo_de_frontera)
    """
    frontera = deque([inicio])          # Cola con el nodo inicial
    padre = {inicio: None}              # Padres; también sirve como "visitados"
    orden, max_front = [], 1            # Nodos expandidos y memoria máxima usada
    while frontera:
        max_front = max(max_front, len(frontera))  # Registrar tamaño de frontera
        nodo = frontera.popleft()       # Sacar el nodo MÁS ANTIGUO (FIFO)
        orden.append(nodo)              # Contarlo como nodo expandido
        if nodo == meta:                # Prueba de meta
            return reconstruir(padre, meta), orden, max_front
        for v in ady[nodo]:             # Generar sucesores
            if v not in padre:          # Solo si nunca se había alcanzado
                padre[v] = nodo         # Recordar de dónde vino
                frontera.append(v)      # Entra al final de la cola
    return None, orden, max_front       # Frontera vacía: no hay solución


def dfs(ady, inicio, meta):
    """Búsqueda en profundidad (DFS): la frontera es una PILA (LIFO).

    Avanza por una rama hasta que no puede continuar y entonces retrocede.
    Es más barata en memoria pero NO garantiza el camino más corto.
    Devuelve: (camino, orden_de_expansión, tamaño_máximo_de_frontera)
    """
    frontera = [inicio]                 # Pila (lista de Python) con el inicio
    padre = {inicio: None}              # Padres para reconstruir el camino
    visitado, orden, max_front = set(), [], 1
    while frontera:
        max_front = max(max_front, len(frontera))  # Registrar tamaño de frontera
        nodo = frontera.pop()           # Sacar el nodo MÁS RECIENTE (LIFO)
        if nodo in visitado:            # Puede haber duplicados en la pila
            continue
        visitado.add(nodo)              # Marcar como expandido
        orden.append(nodo)              # Contarlo como nodo expandido
        if nodo == meta:                # Prueba de meta
            return reconstruir(padre, meta), orden, max_front
        # Se apilan en reversa para que "Arriba" sea lo primero en explorarse
        for v in reversed(ady[nodo]):
            if v not in visitado:
                padre[v] = nodo         # Padre más reciente (el que usará la pila)
                frontera.append(v)
    return None, orden, max_front       # No hay solución


def acciones_de(camino):
    """Traduce un camino de celdas a la lista de acciones (Arriba, Derecha...)."""
    nombres = {(dr, dc): n for dr, dc, n in ACCIONES}
    # Para cada par de celdas consecutivas se busca qué movimiento las une
    return [nombres[(b[0] - a[0], b[1] - a[1])] for a, b in zip(camino, camino[1:])]


def medir(algoritmo, ady, ini, meta, repeticiones=200):
    """Ejecuta un algoritmo varias veces y devuelve sus métricas.

    El tiempo reportado es la MEDIANA de las repeticiones (en milisegundos),
    lo que reduce el ruido causado por el sistema operativo.
    """
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()        # Reloj de alta resolución
        res = algoritmo(ady, ini, meta)  # Se ejecuta la búsqueda
        tiempos.append((time.perf_counter() - t0) * 1000)  # s -> ms
    camino, orden, mf = res             # Resultado de la última ejecución
    return {"camino": camino, "orden": orden,
            "tiempo_ms": statistics.median(tiempos),
            "expandidos": len(orden),                       # Nodos expandidos
            "longitud": len(camino) - 1 if camino else None,  # Pasos del camino
            "max_frontera": mf}                             # Memoria máxima


# ---------------------------------------------------------------------------
# 4. Gráficas
# ---------------------------------------------------------------------------
CMAP = ListedColormap(["#ffffff", "#2b2d42"])   # 0 = blanco (libre), 1 = oscuro (pared)


def dibujar(grid, ini, meta, ax, camino=None, orden=None, titulo=""):
    """Dibuja el laberinto; opcionalmente el orden de expansión y el camino."""
    ax.imshow(grid, cmap=CMAP, interpolation="nearest")   # Paredes y pasillos
    if orden:                                             # Nodos expandidos
        ys, xs = zip(*orden)
        # El color va de claro a oscuro según el orden en que se expandieron
        ax.scatter(xs, ys, c=range(len(orden)), cmap="YlGnBu", s=26,
                   marker="s", alpha=0.85)
    if camino:                                            # Camino solución
        ys, xs = zip(*camino)
        ax.plot(xs, ys, color="#e63946", lw=2.4)
    ax.scatter(ini[1], ini[0], c="#2a9d8f", s=110, marker="o", zorder=5, edgecolors="k")   # Inicio
    ax.scatter(meta[1], meta[0], c="#f4a261", s=140, marker="*", zorder=5, edgecolors="k")  # Meta
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])                  # Sin ejes numéricos


def figuras(grid, ini, meta, rb, rd):
    """Genera y guarda las figuras del reporte en ../img/."""
    # Figura 1: laberinto original
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    dibujar(grid, ini, meta, ax,
            titulo=f"Laberinto {len(grid)}x{len(grid[0])}  (S=verde, G=estrella)")
    fig.tight_layout(); fig.savefig(IMG / "laberinto.png", dpi=170); plt.close(fig)

    # Figura 3: soluciones y orden de exploración de BFS y DFS lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.7))
    for ax, r, n in zip(axs, (rb, rd), ("BFS", "DFS")):
        dibujar(grid, ini, meta, ax, r["camino"], r["orden"],
                f"{n}: {r['expandidos']} expandidos, camino = {r['longitud']} pasos")
    fig.suptitle("Orden de expansión (claro->oscuro) y camino solución (rojo)", fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / "soluciones.png", dpi=170); plt.close(fig)

    # Figura 4: barras comparando las cuatro métricas
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 3.1))
    claves = [("tiempo_ms", "Tiempo (ms)"), ("expandidos", "Nodos expandidos"),
              ("longitud", "Longitud camino"), ("max_frontera", "Frontera máx.")]
    for ax, (k, t) in zip(axs, claves):
        b = ax.bar(["BFS", "DFS"], [rb[k], rd[k]], color=["#2a9d8f", "#e76f51"])
        ax.bar_label(b, fmt="%.3g", fontsize=8)           # Valor sobre cada barra
        ax.set_title(t, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "comparativa.png", dpi=170); plt.close(fig)

    # Figura 2: ejemplo pequeño que muestra cuadrícula -> grafo
    peq = [[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0]]
    G = a_networkx(construir_grafo(peq))
    pos = {n: (n[1], -n[0]) for n in G}                   # Posición = (col, -fila)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    axs[0].imshow(peq, cmap=CMAP); axs[0].set_title("Cuadrícula (negro = pared)", fontsize=9)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    nx.draw(G, pos, ax=axs[1], with_labels=True, node_size=520, font_size=6,
            node_color="#a8dadc", edge_color="#457b9d")
    axs[1].set_title("Grafo G=(V,E) equivalente", fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "grafo_ejemplo.png", dpi=170); plt.close(fig)


def animar(grid, ini, meta, rb, rd, paso=3):
    """Crea un GIF con el MOVIMIENTO de BFS y DFS explorando el laberinto.

    Cada fotograma avanza `paso` nodos en el orden real de expansión de cada
    algoritmo (punto rojo = nodo que se está expandiendo). Al terminar la
    exploración se dibuja el camino solución y se deja unos fotogramas fijos.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.5))
    datos = [(rb, "BFS"), (rd, "DFS")]
    espera = 15                                           # Fotogramas finales fijos
    total = max(len(rb["orden"]), len(rd["orden"])) // paso + 1 + espera

    def actualizar(f):
        k = f * paso                                      # Nodos expandidos hasta ahora
        for ax, (r, nombre) in zip(axs, datos):
            ax.clear()                                    # Se redibuja cada cuadro
            fin = k >= len(r["orden"])                    # ¿Ya terminó este algoritmo?
            vistos = r["orden"][:k]
            dibujar(grid, ini, meta, ax,
                    camino=r["camino"] if fin else None,  # Camino solo al final
                    orden=vistos or None,
                    titulo=f"{nombre}: {min(k, len(r['orden']))} expandidos"
                           + (f" | camino = {r['longitud']} pasos" if fin else ""))
            if vistos and not fin:                        # Nodo que se expande ahora
                ax.scatter(vistos[-1][1], vistos[-1][0], c="#e63946", s=60, zorder=6)

    anim = FuncAnimation(fig, actualizar, frames=total)
    anim.save(IMG / "animacion_bfs_dfs.gif", writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)


def fotogramas(grid, ini, meta, rb, rd):
    """Figura estática con 4 'fotos' del movimiento (25 %, 50 %, 75 %, 100 %)."""
    fracs = [0.25, 0.5, 0.75, 1.0]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.8))
    for fila, (r, nombre) in enumerate(((rb, "BFS"), (rd, "DFS"))):
        for col, fr in enumerate(fracs):
            k = int(len(r["orden"]) * fr)                 # Nodos expandidos en esa foto
            dibujar(grid, ini, meta, axs[fila][col],
                    camino=r["camino"] if fr == 1.0 else None,
                    orden=r["orden"][:k],
                    titulo=f"{nombre} - {int(fr * 100)} % ({k} nodos)")
    fig.tight_layout(); fig.savefig(IMG / "movimiento.png", dpi=150); plt.close(fig)


def escalamiento():
    """Compara BFS y DFS en laberintos de tamaños crecientes.

    Para cada tamaño se promedian 10 laberintos distintos (semillas 0..9).
    Sirve para comprobar empíricamente el crecimiento ~ O(|V| + |E|).
    """
    tams = [5, 10, 15, 20, 30, 40]                        # Celdas por lado
    datos = {"BFS": {"exp": [], "t": [], "lon": []},
             "DFS": {"exp": [], "t": [], "lon": []}}
    for n in tams:
        acum = {a: {"exp": [], "t": [], "lon": []} for a in datos}
        for s in range(10):                               # 10 semillas por tamaño
            g, i, m = generar_laberinto(n, semilla=s, bucles=0.15)
            ady = construir_grafo(g)
            for nombre, f in (("BFS", bfs), ("DFS", dfs)):
                r = medir(f, ady, i, m, repeticiones=15)
                acum[nombre]["exp"].append(r["expandidos"])
                acum[nombre]["t"].append(r["tiempo_ms"])
                acum[nombre]["lon"].append(r["longitud"])
        for a in datos:                                   # Promedio por tamaño
            for k in ("exp", "t", "lon"):
                datos[a][k].append(statistics.mean(acum[a][k]))

    # Figura 5: tres gráficas de líneas (expandidos, tiempo y longitud)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, k, t in zip(axs, ("exp", "t", "lon"),
                        ("Nodos expandidos", "Tiempo (ms)", "Longitud del camino")):
        for a, col in (("BFS", "#2a9d8f"), ("DFS", "#e76f51")):
            ax.plot(tams, datos[a][k], marker="o", label=a, color=col)
        ax.set_title(t, fontsize=9); ax.set_xlabel("celdas por lado (n)")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend()
    fig.tight_layout(); fig.savefig(IMG / "escalamiento.png", dpi=170); plt.close(fig)
    return {"tamanos": tams, **datos}


# ---------------------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------------------
def main():
    # a) Datos de entrada: laberinto, grafo y sus tamaños
    grid, ini, meta = generar_laberinto(n_celdas=10, semilla=7, bucles=0.15)
    ady = construir_grafo(grid)
    V = len(ady)                                          # Número de nodos
    E = sum(len(v) for v in ady.values()) // 2            # Aristas (cada una cuenta 2 veces)
    G = a_networkx(ady)
    # Validación: nuestro grafo coincide con el de NetworkX
    assert G.number_of_nodes() == V and G.number_of_edges() == E
    print(f"Laberinto {len(grid)}x{len(grid[0])} | V={V} | E={E} | ciclos={E - V + 1}")
    print(f"Inicio={ini}  Meta={meta}  conexo={nx.is_connected(G)}")

    # b) Ejecutar y medir ambos algoritmos
    rb = medir(bfs, ady, ini, meta)
    rd = medir(dfs, ady, ini, meta)

    # c) Validación: BFS debe igualar el camino óptimo calculado por NetworkX
    optimo = nx.shortest_path_length(G, ini, meta)
    assert rb["longitud"] == optimo, "BFS debe ser óptimo"

    # d) Mostrar resultados en consola
    for n, r in (("BFS", rb), ("DFS", rd)):
        print(f"{n}: pasos={r['longitud']}  expandidos={r['expandidos']}  "
              f"frontera_max={r['max_frontera']}  tiempo={r['tiempo_ms']:.4f} ms")
    print("Camino BFS (acciones):", ", ".join(acciones_de(rb["camino"])))
    print("Camino DFS (acciones):", ", ".join(acciones_de(rd["camino"])))

    # e) Gráficas y experimento de escalamiento
    figuras(grid, ini, meta, rb, rd)
    fotogramas(grid, ini, meta, rb, rd)   # Fotos del movimiento (para el PDF)
    animar(grid, ini, meta, rb, rd)       # GIF animado del movimiento
    esc = escalamiento()

    # f) Guardar todo en resultados.json (el campo "orden" se omite por tamaño)
    out = {"V": V, "E": E, "ciclos": E - V + 1, "inicio": ini, "meta": meta,
           "optimo_networkx": optimo, "grid": grid,
           "BFS": {k: v for k, v in rb.items() if k != "orden"},
           "DFS": {k: v for k, v in rd.items() if k != "orden"},
           "acciones_BFS": acciones_de(rb["camino"]),
           "acciones_DFS": acciones_de(rd["camino"]),
           "escalamiento": esc}
    Path(__file__).with_name("resultados.json").write_text(json.dumps(out, indent=1))
    print("Listo: imágenes en", IMG)


# Se ejecuta main() solo si el archivo se corre directamente
if __name__ == "__main__":
    main()

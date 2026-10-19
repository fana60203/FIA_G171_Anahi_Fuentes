"""
problema2.py -- Actividad 2 (2.1 Problemas de búsqueda + 2.2 Espacios de búsqueda)
PROBLEMA 2: robot de almacén que recoge un paquete en P y lo entrega en D.

"""
import os
import math
import heapq
from collections import deque
import matplotlib
matplotlib.use("Agg")            # guarda imágenes en archivo (sin abrir ventanas)
import matplotlib.pyplot as plt
import networkx as nx


# ===========================================================================
# PARTE 1: MARCO GENÉRICO DE BÚSQUEDA
# ---------------------------------------------------------------------------
# Un problema de búsqueda se define con 5 componentes (Russell y Norvig, 2020):
#   1) Estado inicial          2) Acciones(s)         3) Resultado(s, a)
#   4) Costo de paso c(s,a,s') 5) Prueba de meta
# Si el problema se define así, el ESPACIO DE ESTADOS es un grafo dirigido y
# ponderado: nodo = estado, arista dirigida = acción, peso = costo del paso.
# ===========================================================================
class ProblemaBusqueda:
    """Interfaz abstracta: cada problema concreto hereda de aquí y define sus
    5 componentes. Los algoritmos de abajo solo usan estos métodos, por eso
    sirven para cualquier problema."""

    estado_inicial = None            # (1) estado en el que empieza la búsqueda

    def acciones(self, s):           # (2) lista de acciones aplicables en el estado s
        raise NotImplementedError

    def resultado(self, s, a):       # (3) modelo de transición: estado que resulta de aplicar a en s
        raise NotImplementedError

    def costo(self, s, a, s2):       # (4) costo de ir de s a s2 con la acción a (debe ser > 0)
        return 1

    def es_meta(self, s):            # (5) ¿el estado s cumple el objetivo?
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2.2 ESPACIO DE BÚSQUEDA: generador de estados
# ---------------------------------------------------------------------------
def generar_espacio_estados(problema):
    """Recorre TODOS los estados alcanzables desde el inicial y construye el
    grafo dirigido y ponderado (nx.DiGraph) del espacio de estados."""
    G = nx.DiGraph()                       # grafo dirigido (las aristas tienen sentido)
    inicio = problema.estado_inicial
    G.add_node(inicio)                     # el estado inicial es el primer nodo
    visitados = {inicio}                   # estados ya descubiertos (evita repetirlos)
    cola = deque([inicio])                 # cola FIFO: exploración por niveles
    while cola:
        s = cola.popleft()                 # saca el siguiente estado por expandir
        if problema.es_meta(s):            # un estado meta ya no se expande
            continue
        for a in problema.acciones(s):     # prueba cada acción aplicable
            s2 = problema.resultado(s, a)  # estado al que llega con esa acción
            # arista s -> s2 con el nombre de la acción y su costo como peso
            G.add_edge(s, s2, accion=a, weight=problema.costo(s, a, s2))
            if s2 not in visitados:        # si es un estado nuevo, hay que explorarlo
                visitados.add(s2)
                cola.append(s2)
    return G


# ---------------------------------------------------------------------------
# 2.1 ALGORITMOS DE BÚSQUEDA CIEGA (búsqueda en grafo con lista de explorados)
# Cada uno devuelve un diccionario con:
#   camino (lista de estados), acciones, costo, expandidos, generados
# "expandidos" = estados cuyos sucesores se calcularon; "generados" = estados
# descubiertos. Ambos miden el esfuerzo (tiempo) del algoritmo.
# ---------------------------------------------------------------------------
def _reconstruir(padres, s):
    """Sigue los punteros 'padre' desde la meta hasta el inicio y devuelve el
    camino y las acciones en orden de inicio -> meta."""
    camino, acciones = [s], []
    while padres[s][0] is not None:        # el inicial tiene padre None
        s, a = padres[s]                   # retrocede al padre y recuerda la acción
        camino.append(s)
        acciones.append(a)
    camino.reverse()                       # estaban de meta a inicio: se invierten
    acciones.reverse()
    return camino, acciones


def _costo_camino(problema, camino, acciones):
    """Suma el costo de cada paso del camino."""
    return sum(problema.costo(camino[i], acciones[i], camino[i + 1])
               for i in range(len(acciones)))


def bfs(problema):
    """BÚSQUEDA EN ANCHURA. Usa cola FIFO. Encuentra la solución con MENOS
    PASOS; solo es óptima en costo si todos los pasos cuestan lo mismo."""
    s0 = problema.estado_inicial
    padres = {s0: (None, None)}            # estado -> (padre, acción); también marca los descubiertos
    frontera = deque([s0])                 # cola de estados por expandir
    expandidos, generados = 0, 1
    if problema.es_meta(s0):
        return dict(camino=[s0], acciones=[], costo=0, expandidos=0, generados=1)
    while frontera:
        s = frontera.popleft()             # el más antiguo primero (por niveles)
        expandidos += 1
        for a in problema.acciones(s):
            s2 = problema.resultado(s, a)
            if s2 not in padres:           # solo estados nuevos
                padres[s2] = (s, a)
                generados += 1
                if problema.es_meta(s2):   # BFS puede probar la meta al generar
                    c, ac = _reconstruir(padres, s2)
                    return dict(camino=c, acciones=ac, costo=_costo_camino(problema, c, ac),
                                expandidos=expandidos, generados=generados)
                frontera.append(s2)
    return None                            # no hay solución


def dfs(problema):
    """BÚSQUEDA EN PROFUNDIDAD. Usa pila LIFO. Con lista de explorados es
    completa en espacios finitos, pero NO garantiza solución óptima."""
    s0 = problema.estado_inicial
    padres = {s0: (None, None)}
    pila = [s0]                            # pila de estados por explorar
    explorados = set()                     # estados ya sacados de la pila
    expandidos, generados = 0, 1
    while pila:
        s = pila.pop()                     # el último en entrar sale primero (profundiza)
        if s in explorados:
            continue
        explorados.add(s)
        if problema.es_meta(s):
            c, ac = _reconstruir(padres, s)
            return dict(camino=c, acciones=ac, costo=_costo_camino(problema, c, ac),
                        expandidos=expandidos, generados=generados)
        expandidos += 1
        for a in reversed(problema.acciones(s)):   # reversed: explora en el orden natural
            s2 = problema.resultado(s, a)
            if s2 not in explorados:
                if s2 not in padres:
                    generados += 1
                padres[s2] = (s, a)
                pila.append(s2)
    return None


def ucs(problema):
    """BÚSQUEDA DE COSTO UNIFORME (Dijkstra). Usa cola de prioridad ordenada
    por el costo acumulado g(n). Es COMPLETA y ÓPTIMA si todos los costos > 0."""
    s0 = problema.estado_inicial
    contador = 0                           # desempate para que el heap nunca compare estados
    frontera = [(0, contador, s0)]         # montículo de (costo acumulado, desempate, estado)
    padres = {s0: (None, None)}
    g = {s0: 0}                            # mejor costo conocido hasta cada estado
    explorados = set()
    expandidos, generados = 0, 1
    while frontera:
        costo, _, s = heapq.heappop(frontera)   # saca el estado de MENOR costo acumulado
        if s in explorados:                # entrada vieja (ya hay una mejor): se ignora
            continue
        if problema.es_meta(s):            # la meta se prueba al EXTRAER: así el costo es mínimo
            c, ac = _reconstruir(padres, s)
            return dict(camino=c, acciones=ac, costo=costo,
                        expandidos=expandidos, generados=generados)
        explorados.add(s)
        expandidos += 1
        for a in problema.acciones(s):
            s2 = problema.resultado(s, a)
            nuevo = costo + problema.costo(s, a, s2)   # costo de llegar a s2 pasando por s
            if s2 not in g or nuevo < g[s2]:           # ¿encontramos un camino mejor?
                g[s2] = nuevo
                padres[s2] = (s, a)
                contador += 1
                generados += 1
                heapq.heappush(frontera, (nuevo, contador, s2))
    return None


def factor_ramificacion(G):
    """Factor de ramificación promedio b = (nº de aristas) / (nº de nodos con salida)."""
    con_salida = [n for n in G if G.out_degree(n) > 0]
    return G.number_of_edges() / max(1, len(con_salida))


# ---------------------------------------------------------------------------
# ANÁLISIS DE COMPLEJIDAD (teórico + medido). Lo imprime cada programa.
# Notación (Russell y Norvig, 2020):
#   b  = factor de ramificación      d  = profundidad de la solución (nº de pasos)
#   m  = profundidad máxima del espacio   C* = costo de la solución óptima
#   eps = costo mínimo de un paso    |S|, |A| = nº de estados y de transiciones
# ---------------------------------------------------------------------------
def analisis_complejidad(nombre, G, res, unidad):
    n, m_aristas = G.number_of_nodes(), G.number_of_edges()
    b = factor_ramificacion(G)
    u = res["UCS"]
    d, C = len(u["acciones"]), u["costo"]
    eps = min(w for _, _, w in G.edges(data="weight"))
    k = 1 + int(C // eps)                  # exponente de UCS: 1 + piso(C*/eps)
    print(f"\n===== ANÁLISIS DE COMPLEJIDAD — {nombre} =====")
    print(f"Datos medidos: |S|={n}  |A|={m_aristas}  b={b:.2f}  d={d}  "
          f"C*={C} {unidad}  eps={eps} {unidad}")
    print("\nCotas teóricas SIN lista de explorados (árbol de búsqueda):")
    print(f"  BFS: tiempo O(b^d)         = {b:.2f}^{d}      ≈ {b ** d:,.0f} nodos | espacio O(b^d)")
    print(f"  DFS: tiempo O(b^m)         (m = prof. máxima) | espacio O(b·m)   | no óptimo")
    print(f"  UCS: tiempo O(b^(1+C*/eps)) = {b:.2f}^{k}     ≈ {b ** k:,.0f} nodos | espacio igual | óptimo")
    print("\nCotas CON lista de explorados (lo que implementa este programa):")
    print(f"  BFS y DFS: O(|S|+|A|)               = O({n}+{m_aristas}) = {n + m_aristas} operaciones")
    print(f"  UCS con montículo: O((|S|+|A|)log|S|) ≈ {(n + m_aristas) * math.log2(n):,.0f} operaciones")
    print("  Espacio: O(|S|) en los tres (guardan los estados explorados).")
    print("\nResultados medidos (esfuerzo real de cada algoritmo):")
    print(f"  {'Algoritmo':<10}{'Costo':>8}{'Pasos':>8}{'Expandidos':>12}{'Generados':>11}")
    for nom in ("BFS", "DFS", "UCS"):
        r = res[nom]
        print(f"  {nom:<10}{r['costo']:>8}{len(r['acciones']):>8}{r['expandidos']:>12}{r['generados']:>11}")
    print(f"\nConclusión: la cota b^d (≈{b ** d:,.0f}) sería inviable, pero la lista de explorados "
          f"limita el trabajo a los {n} estados del espacio; UCS expandió {u['expandidos']} de {n} "
          f"({100 * u['expandidos'] / n:.0f}%) y es el único que garantiza el costo mínimo.")


# ===========================================================================
# PARTE 2: PROBLEMA 2 — ROBOT DE ALMACÉN (recoger un paquete y entregarlo)
# ===========================================================================

# ---- Datos del problema -----------------------------------------------------
# Plano del almacén (5 filas x 7 columnas):
#   R = base del robot (estado inicial)   P = anaquel donde recoge el paquete
#   D = muelle de entrega                 # = estante (obstáculo)
#   ~ = pasillo congestionado (avanzar cuesta más)   . = pasillo libre
MAPA = [
    "R...#..",
    ".#..#.D",
    ".#.~~..",
    "...#.#.",
    ".P.....",
]
C_NORMAL, C_CONGEST, C_ACCION = 1, 3, 2   # costos en segundos: mover / mover a zona ~ / recoger o entregar


class RobotAlmacen(ProblemaBusqueda):
    """FORMULACIÓN DEL PROBLEMA (los 5 componentes).
    Estado   : (fila, columna, fase)  fase 0 = sin paquete, 1 = con paquete, 2 = entregado
    Acciones : N, S, E, O (mover una celda), RECOGER (solo en P, fase 0),
               ENTREGAR (solo en D, fase 1)
    Meta     : fase == 2.
    La fase hace el grafo DIRIGIDO: RECOGER y ENTREGAR no se pueden deshacer."""

    MOV = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "O": (0, -1)}   # desplazamiento (fila, col)

    def __init__(self, mapa=MAPA):
        self.mapa = mapa
        self.nf, self.nc = len(mapa), len(mapa[0])
        for r, fila in enumerate(mapa):                 # localiza R, P y D en el plano
            for c, ch in enumerate(fila):
                if ch == "R":
                    self.estado_inicial = (r, c, 0)     # empieza en R, sin paquete
                elif ch == "P":
                    self.P = (r, c)
                elif ch == "D":
                    self.D = (r, c)

    def _libre(self, r, c):
        """¿La celda existe dentro del plano y no es un estante?"""
        return 0 <= r < self.nf and 0 <= c < self.nc and self.mapa[r][c] != "#"

    def acciones(self, s):
        r, c, f = s
        if f == 2:                                      # ya entregó: no hay más acciones
            return []
        acc = []
        for a, (dr, dc) in self.MOV.items():            # movimientos posibles
            if self._libre(r + dr, c + dc):
                acc.append(a)
        if f == 0 and (r, c) == self.P:                 # solo puede recoger estando en P sin paquete
            acc.append("RECOGER")
        if f == 1 and (r, c) == self.D:                 # solo puede entregar en D con paquete
            acc.append("ENTREGAR")
        return acc

    def resultado(self, s, a):
        r, c, f = s
        if a == "RECOGER":
            return (r, c, 1)                            # misma celda, ahora con paquete
        if a == "ENTREGAR":
            return (r, c, 2)                            # misma celda, paquete entregado (meta)
        dr, dc = self.MOV[a]
        return (r + dr, c + dc, f)                      # se mueve; la fase no cambia

    def costo(self, s, a, s2):
        if a in ("RECOGER", "ENTREGAR"):
            return C_ACCION                             # 2 s
        # moverse a una celda congestionada cuesta 3 s; a una libre, 1 s
        return C_CONGEST if self.mapa[s2[0]][s2[1]] == "~" else C_NORMAL

    def es_meta(self, s):
        return s[2] == 2                                # fase 2 = paquete entregado


def dibujar_mapa(p, res, archivo):
    """Dibuja el plano del almacén y la ruta óptima encima."""
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    col = {"#": "#444444", "~": "#f4b183", ".": "#ffffff", "R": "#c6e0b4", "P": "#9dc3e6",
           "D": "#ffe699"}
    for r in range(p.nf):
        for c in range(p.nc):
            ch = p.mapa[r][c]
            ax.add_patch(plt.Rectangle((c, p.nf - r - 1), 1, 1, fc=col[ch], ec="#888888"))
            if ch in "RPD":                              # rotula R, P y D
                ax.text(c + .5, p.nf - r - .5, ch, ha="center", va="center", fontsize=15,
                        fontweight="bold")
            if ch == "~":                                # marca el costo triple
                ax.text(c + .5, p.nf - r - .5, "×3", ha="center", va="center", fontsize=10,
                        color="#843c0c")
    xs = [s[1] + .5 for s in res["camino"]]              # centro de cada celda de la ruta
    ys = [p.nf - s[0] - .5 for s in res["camino"]]
    ax.plot(xs, ys, "-o", color="#d6001c", lw=2.4, ms=4)
    ax.set_xlim(0, p.nc); ax.set_ylim(0, p.nf); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Problema 2 — Plano del almacén y ruta óptima (costo = {res['costo']} s)",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(archivo, dpi=170); plt.close()


def dibujar_grafo(p, G, res, archivo):
    """Dibuja el espacio de estados en tres capas (fase 0, fase 1 y meta):
    grafo dirigido y ponderado, con la ruta óptima en rojo."""
    desplaz = {0: 0, 1: p.nc + 2, 2: 2 * (p.nc + 2)}     # desplazamiento horizontal de cada capa
    pos = {s: (s[1] + desplaz[s[2]], -s[0]) for s in G}  # posición (x, y) de cada estado
    plt.figure(figsize=(15, 4.8))
    ruta = res["camino"]
    ar_ruta = set(zip(ruta[:-1], ruta[1:]))
    no_ruta = [e for e in G.edges if e not in ar_ruta]
    tipo = {0: "#9dc3e6", 1: "#f4b183", 2: "#a9d18e"}    # color por fase
    nx.draw_networkx_nodes(G, pos, node_size=55, node_color=[tipo[s[2]] for s in G])
    nx.draw_networkx_edges(G, pos, edgelist=no_ruta, edge_color="#c8c8c8", width=0.5,
                           arrows=True, arrowsize=5, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=list(ar_ruta), edge_color="#d6001c", width=2.4,
                           arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, nodelist=ruta, node_size=70, node_color="#d6001c")
    for f, txt in ((0, "Capa 0: sin paquete"), (1, "Capa 1: con paquete"), (2, "Meta")):
        plt.text(desplaz[f] + (p.nc - 1) / 2, 0.7, txt, ha="center", fontsize=10,
                 fontweight="bold")
    plt.ylim(-4.6, 1.3)
    plt.title("Problema 2 — Espacio de estados (fila, col, fase): grafo dirigido y ponderado.\n"
              "Las aristas RECOGER y ENTREGAR son de un solo sentido (capa 0→1→meta)",
              fontsize=11, pad=18)
    plt.axis("off"); plt.tight_layout(); plt.savefig(archivo, dpi=170); plt.close()


def main():
    # carpeta "figuras" junto a este archivo (se crea sola si no existe)
    FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
    os.makedirs(FIG, exist_ok=True)

    p = RobotAlmacen()                                   # 1) se crea el problema
    G = generar_espacio_estados(p)                       # 2) se genera el espacio de estados (grafo)
    print(f"[P2] |S| = {G.number_of_nodes()} estados, |A| = {G.number_of_edges()} transiciones, "
          f"b promedio = {factor_ramificacion(G):.2f}, "
          f"dirigido = {G.is_directed()}, ponderado = {nx.is_weighted(G)}")
    res = {"BFS": bfs(p), "DFS": dfs(p), "UCS": ucs(p)}  # 3) se resuelve con 3 algoritmos
    for nom, r in res.items():
        print(f"[P2] {nom}: costo = {r['costo']} s, pasos = {len(r['acciones'])}, "
              f"expandidos = {r['expandidos']}, generados = {r['generados']}")
    print("[P2] Acciones UCS:", " ".join(res["UCS"]["acciones"]))

    # 4) verificación independiente con Dijkstra de NetworkX
    metas = [n for n in G if p.es_meta(n)]
    mejor = min(nx.dijkstra_path_length(G, p.estado_inicial, m) for m in metas)
    print(f"[P2] Verificación nx.dijkstra: costo óptimo = {mejor} s "
          f"(UCS = {res['UCS']['costo']} s) ->", "OK" if mejor == res["UCS"]["costo"] else "ERROR")
    assert mejor == res["UCS"]["costo"]

    dibujar_mapa(p, res["UCS"], os.path.join(FIG, "p2_mapa_ruta.png"))          # 5) plano con la ruta
    dibujar_grafo(p, G, res["UCS"], os.path.join(FIG, "p2_espacio_estados.png"))  # 5) grafo del espacio
    analisis_complejidad("Problema 2 (Robot de almacén)", G, res, "s")            # 6) complejidad
    return p, G, res


if __name__ == "__main__":
    main()
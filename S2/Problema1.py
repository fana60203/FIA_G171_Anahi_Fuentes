"""
problema1.py -- Actividad 2 (2.1 Problemas de búsqueda + 2.2 Espacios de búsqueda)
PROBLEMA 1: ruta de menor tiempo en el Metro de la CDMX (Universidad -> Zócalo).

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
# PARTE 2: PROBLEMA 1 — RUTA DE MENOR TIEMPO EN EL METRO DE LA CDMX
# ===========================================================================

# ---- Datos del problema (supuestos del modelo) ----------------------------
T_TRAMO = 2        # minutos entre dos estaciones consecutivas (valor medio aproximado)
T_TRANSBORDO = 5   # minutos por hacer un transbordo (caminar + esperar, aproximado)

# Estaciones de las líneas 1, 2 y 3 en orden de recorrido (datos de entrada)
LINEAS = {
    1: ["Observatorio", "Tacubaya", "Juanacatlán", "Chapultepec", "Sevilla", "Insurgentes",
        "Cuauhtémoc", "Balderas", "Salto del Agua", "Isabel la Católica", "Pino Suárez",
        "Merced", "Candelaria", "San Lázaro", "Moctezuma", "Balbuena",
        "Boulevard Puerto Aéreo", "Gómez Farías", "Zaragoza", "Pantitlán"],
    2: ["Cuatro Caminos", "Panteones", "Tacuba", "Cuitláhuac", "Popotla", "Colegio Militar",
        "Normal", "San Cosme", "Revolución", "Hidalgo", "Bellas Artes", "Allende",
        "Zócalo", "Pino Suárez", "San Antonio Abad", "Chabacano", "Viaducto", "Xola",
        "Villa de Cortés", "Nativitas", "Portales", "Ermita", "General Anaya", "Tasqueña"],
    3: ["Indios Verdes", "Deportivo 18 de Marzo", "Potrero", "La Raza", "Tlatelolco",
        "Guerrero", "Hidalgo", "Juárez", "Balderas", "Niños Héroes", "Hospital General",
        "Centro Médico", "Etiopía", "Eugenia", "División del Norte", "Zapata", "Coyoacán",
        "Viveros", "Miguel Ángel de Quevedo", "Copilco", "Universidad"],
}
ORIGEN = ("Universidad", 3)   # estado inicial: en Universidad, viajando en la Línea 3
DESTINO = "Zócalo"            # meta: llegar a Zócalo (en cualquier línea)


class RutaMetro(ProblemaBusqueda):
    """FORMULACIÓN DEL PROBLEMA (los 5 componentes).
    Estado   : (estación, línea en la que va el usuario)
    Acciones : IR_A_<estación vecina> en la misma línea (2 min)
               TRANSBORDO_L<n> si la estación pertenece también a otra línea (5 min)
    Meta     : estar en la estación Zócalo."""

    def __init__(self):
        self.estado_inicial = ORIGEN
        # índice auxiliar: estación -> {línea: posición dentro de esa línea}
        self.idx = {}
        for l, est in LINEAS.items():
            for i, e in enumerate(est):
                self.idx.setdefault(e, {})[l] = i

    def acciones(self, s):
        est, l = s
        pos = self.idx[est][l]             # posición de la estación en su línea
        lista = LINEAS[l]
        acc = []
        if pos > 0:                        # hay estación anterior
            acc.append(f"IR_A_{lista[pos - 1]}")
        if pos < len(lista) - 1:           # hay estación siguiente
            acc.append(f"IR_A_{lista[pos + 1]}")
        for otra in self.idx[est]:         # líneas que pasan por esta estación
            if otra != l:
                acc.append(f"TRANSBORDO_L{otra}")
        return acc

    def resultado(self, s, a):
        est, l = s
        if a.startswith("TRANSBORDO_L"):   # mismo lugar, otra línea
            return (est, int(a[len("TRANSBORDO_L"):]))
        return (a[len("IR_A_"):], l)       # otra estación, misma línea

    def costo(self, s, a, s2):
        # transbordar cuesta 5 min; avanzar un tramo cuesta 2 min (siempre > 0)
        return T_TRANSBORDO if a.startswith("TRANSBORDO") else T_TRAMO

    def es_meta(self, s):
        return s[0] == DESTINO             # no importa en qué línea llegue


def resumen_ruta(res):
    """Agrupa el camino en tramos por línea (se corta en cada transbordo)."""
    tramos, ini = [], res["camino"][0]
    for i, a in enumerate(res["acciones"]):
        if a.startswith("TRANSBORDO"):
            tramos.append((ini, res["camino"][i]))   # termina el tramo antes del transbordo
            ini = res["camino"][i + 1]               # el siguiente tramo empieza ya en la otra línea
    tramos.append((ini, res["camino"][-1]))
    return tramos


def dibujar(G, res, archivo):
    """Dibuja el espacio de estados (grafo dirigido ponderado) y resalta en rojo
    la ruta óptima. Cada línea del Metro se dibuja en un renglón."""
    pos = {}                                         # posición (x, y) de cada nodo en el dibujo
    for l, est in LINEAS.items():
        for i, e in enumerate(est):
            pos[(e, l)] = (i, -3 * l)
    colores = {1: "#e8742a", 2: "#1f5fbf", 3: "#7aa13c"}
    plt.figure(figsize=(15, 6.2))
    ruta = set(res["camino"])
    aristas_ruta = set(zip(res["camino"][:-1], res["camino"][1:]))
    for l in LINEAS:                                 # nodos de cada línea con su color
        nodos = [n for n in G if n[1] == l]
        nx.draw_networkx_nodes(G, pos, nodelist=nodos, node_size=70,
                               node_color=colores[l], alpha=0.85)
    ar_normales = [e for e in G.edges if e not in aristas_ruta]
    nx.draw_networkx_edges(G, pos, edgelist=ar_normales, edge_color="#bbbbbb", width=0.6,
                           arrows=True, arrowsize=6, connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_edges(G, pos, edgelist=list(aristas_ruta), edge_color="#d6001c",
                           width=2.6, arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_nodes(G, pos, nodelist=list(ruta), node_size=110,
                           node_color="#d6001c", edgecolors="black")
    for n in ruta:                                   # nombres de la ruta, alternados arriba/abajo
        x, y = pos[n]
        arriba = (x % 2 == 0)
        plt.text(x, y + (0.5 if arriba else -0.5), n[0], rotation=40, fontsize=6.5,
                 ha="left" if arriba else "right", va="bottom" if arriba else "top",
                 rotation_mode="anchor")
    plt.ylim(-12.5, -1.5)
    for l in LINEAS:
        plt.text(-1.3, -3 * l, f"Línea {l}", color=colores[l], fontsize=11, fontweight="bold",
                 va="center")
    plt.title("Problema 1 — Espacio de estados (estación, línea): grafo dirigido y ponderado\n"
              "En rojo: ruta de costo uniforme (óptima). Cada tramo = 2 min; transbordo = 5 min",
              fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(archivo, dpi=170)                    # guarda la imagen (no abre ventana)
    plt.close()


def main():
    # carpeta "figuras" junto a este archivo (se crea sola si no existe)
    FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
    os.makedirs(FIG, exist_ok=True)

    p = RutaMetro()                                  # 1) se crea el problema
    G = generar_espacio_estados(p)                   # 2) se genera el espacio de estados (grafo)
    print(f"[P1] |S| = {G.number_of_nodes()} estados, |A| = {G.number_of_edges()} transiciones, "
          f"b promedio = {factor_ramificacion(G):.2f}, "
          f"dirigido = {G.is_directed()}, ponderado = {nx.is_weighted(G)}")

    resultados = {"BFS": bfs(p), "DFS": dfs(p), "UCS": ucs(p)}   # 3) se resuelve con 3 algoritmos
    for nom, r in resultados.items():
        print(f"[P1] {nom}: costo = {r['costo']} min, pasos = {len(r['acciones'])}, "
              f"expandidos = {r['expandidos']}, generados = {r['generados']}")
        print("      tramos:", resumen_ruta(r))

    # 4) verificación independiente: Dijkstra de NetworkX debe dar el mismo costo óptimo
    metas = [n for n in G if p.es_meta(n)]
    mejor = min(nx.dijkstra_path_length(G, p.estado_inicial, m) for m in metas)
    print(f"[P1] Verificación nx.dijkstra: costo óptimo = {mejor} min "
          f"(UCS = {resultados['UCS']['costo']} min) ->",
          "OK" if mejor == resultados["UCS"]["costo"] else "ERROR")
    assert mejor == resultados["UCS"]["costo"]

    dibujar(G, resultados["UCS"], os.path.join(FIG, "p1_espacio_estados.png"))   # 5) diagrama del grafo
    analisis_complejidad("Problema 1 (Metro CDMX)", G, resultados, "min")        # 6) complejidad
    return p, G, resultados


if __name__ == "__main__":
    main()
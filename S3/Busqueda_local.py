"""
Actividad Parcial 2 - Seccion 2.3 Busqueda Local
Fundamentos de Inteligencia Artificial - Politecnica de Atlautla

Problema: Ruta minima (TSP) para una gira de distribucion que visita
25 ciudades de Mexico y regresa al punto de partida.

Algoritmos: Hill Climbing (con reinicios aleatorios) y Simulated Annealing.
"""
import math
import random
import json
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DATOS DEL PROBLEMA: ciudades (latitud, longitud) aproximadas
# ---------------------------------------------------------------------------
CIUDADES = {
    "Ciudad de Mexico": (19.4326, -99.1332),
    "Guadalajara": (20.6597, -103.3496),
    "Monterrey": (25.6866, -100.3161),
    "Puebla": (19.0414, -98.2063),
    "Queretaro": (20.5888, -100.3899),
    "Merida": (20.9674, -89.5926),
    "Tijuana": (32.5149, -117.0382),
    "Chihuahua": (28.6330, -106.0691),
    "Oaxaca": (17.0732, -96.7266),
    "Veracruz": (19.1738, -96.1342),
    "Leon": (21.1250, -101.6860),
    "Hermosillo": (29.0729, -110.9559),
    "Cancun": (21.1619, -86.8515),
    "Villahermosa": (17.9892, -92.9475),
    "Tuxtla Gutierrez": (16.7528, -93.1152),
    "Acapulco": (16.8531, -99.8237),
    "Morelia": (19.7060, -101.1950),
    "San Luis Potosi": (22.1565, -100.9855),
    "Zacatecas": (22.7709, -102.5832),
    "Durango": (24.0277, -104.6532),
    "Culiacan": (24.8091, -107.3940),
    "La Paz": (24.1426, -110.3128),
    "Saltillo": (25.4232, -101.0053),
    "Tampico": (22.2331, -97.8610),
    "Torreon": (25.5428, -103.4068),
}
NOMBRES = list(CIUDADES.keys())
N = len(NOMBRES)


def haversine(a, b):
    """Distancia en km entre dos coordenadas (lat, lon)."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    d = math.sin((la2 - la1) / 2) ** 2 + \
        math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(d))


DIST = [[haversine(CIUDADES[i], CIUDADES[j]) for j in NOMBRES] for i in NOMBRES]


# ---------------------------------------------------------------------------
# 2. FUNCION OBJETIVO, VECINDARIO
# ---------------------------------------------------------------------------
def costo(ruta):
    """Funcion objetivo: longitud total del circuito cerrado (km)."""
    return sum(DIST[ruta[i]][ruta[(i + 1) % N]] for i in range(N))


def vecino_2opt(ruta, i, j):
    """Vecino 2-opt: invierte el segmento ruta[i..j]."""
    return ruta[:i] + ruta[i:j + 1][::-1] + ruta[j + 1:]


def vecino_aleatorio(ruta):
    i, j = sorted(random.sample(range(1, N), 2))
    return vecino_2opt(ruta, i, j)


# ---------------------------------------------------------------------------
# 3. HILL CLIMBING (maximo ascenso / descenso por la mejor mejora)
# ---------------------------------------------------------------------------
def hill_climbing(ruta_ini):
    """Hill Climbing de mejor vecino. Devuelve (ruta, costo, historial)."""
    actual = ruta_ini[:]
    c_act = costo(actual)
    historial = [c_act]
    mejora = True
    while mejora:
        mejora = False
        mejor_v, mejor_c = None, c_act
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                v = vecino_2opt(actual, i, j)
                c = costo(v)
                if c < mejor_c - 1e-9:
                    mejor_v, mejor_c = v, c
        if mejor_v is not None:
            actual, c_act = mejor_v, mejor_c
            historial.append(c_act)
            mejora = True
    return actual, c_act, historial


def hill_climbing_reinicios(n_reinicios, semilla):
    """HC con reinicios aleatorios para escapar de optimos locales."""
    random.seed(semilla)
    mejor, mejor_c, mejor_h = None, float("inf"), []
    hist_global = []
    mejor_global = float("inf")
    for _ in range(n_reinicios):
        ini = list(range(N))
        random.shuffle(ini)
        r, c, h = hill_climbing(ini)
        for x in h:
            mejor_global = min(mejor_global, x)
            hist_global.append(mejor_global)
        if c < mejor_c:
            mejor, mejor_c = r, c
    return mejor, mejor_c, hist_global


def hill_climbing_simple(semilla):
    """HC con un solo arranque aleatorio (puede quedar atrapado en optimo local)."""
    random.seed(semilla)
    ini = list(range(N))
    random.shuffle(ini)
    r, c, h = hill_climbing(ini)
    return r, c, h


# ---------------------------------------------------------------------------
# 4. SIMULATED ANNEALING
# ---------------------------------------------------------------------------
def simulated_annealing(T0=2000.0, alpha=0.995, iteraciones=20000,
                        T_min=1e-3, semilla=0):
    """
    T0: temperatura inicial, alpha: factor de enfriamiento geometrico (η),
    iteraciones: maximo de iteraciones, T_min: temperatura minima.
    Criterio de Metropolis: acepta peor con prob exp(-delta/T).
    """
    random.seed(semilla)
    actual = list(range(N))
    random.shuffle(actual)
    c_act = costo(actual)
    mejor, c_mejor = actual[:], c_act
    T = T0
    hist_mejor, hist_actual = [c_mejor], [c_act]
    for _ in range(iteraciones):
        if T < T_min:
            break
        v = vecino_aleatorio(actual)
        c_v = costo(v)
        delta = c_v - c_act
        if delta < 0 or random.random() < math.exp(-delta / T):
            actual, c_act = v, c_v
            if c_act < c_mejor:
                mejor, c_mejor = actual[:], c_act
        T *= alpha
        hist_mejor.append(c_mejor)
        hist_actual.append(c_act)
    return mejor, c_mejor, hist_mejor, hist_actual


# ---------------------------------------------------------------------------
# 5. EXPERIMENTOS
# ---------------------------------------------------------------------------
def graficar_ruta(ax, ruta, titulo):
    xs = [CIUDADES[NOMBRES[i]][1] for i in ruta + [ruta[0]]]
    ys = [CIUDADES[NOMBRES[i]][0] for i in ruta + [ruta[0]]]
    ax.plot(xs, ys, "-o", lw=1.6, ms=5)
    for i in ruta:
        ax.annotate(NOMBRES[i], (CIUDADES[NOMBRES[i]][1], CIUDADES[NOMBRES[i]][0]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(alpha=0.3)


def main():
    SEMILLAS = list(range(30))
    N_REINICIOS = 5
    PARAM_SA = dict(T0=2000.0, alpha=0.9995, iteraciones=40000)

    res = {"HC1": [], "HC": [], "SA": []}
    tiempos = {"HC1": [], "HC": [], "SA": []}
    best_hc1 = (None, float("inf"), None)
    best_hc = (None, float("inf"), None)
    best_sa = (None, float("inf"), None, None)

    for s in SEMILLAS:
        t = time.time()
        r, c, h = hill_climbing_simple(s)
        tiempos["HC1"].append(time.time() - t)
        res["HC1"].append(c)
        if c < best_hc1[1]:
            best_hc1 = (r, c, h)

        t = time.time()
        r, c, h = hill_climbing_reinicios(N_REINICIOS, s)
        tiempos["HC"].append(time.time() - t)
        res["HC"].append(c)
        if c < best_hc[1]:
            best_hc = (r, c, h)

        t = time.time()
        r, c, hm, ha = simulated_annealing(semilla=s, **PARAM_SA)
        tiempos["SA"].append(time.time() - t)
        res["SA"].append(c)
        if c < best_sa[1]:
            best_sa = (r, c, hm, ha)

    # Tabla de resultados
    resumen = {}
    for k in ("HC1", "HC", "SA"):
        v = res[k]
        resumen[k] = {
            "mejor": min(v), "peor": max(v), "media": statistics.mean(v),
            "desv": statistics.stdev(v), "tiempo_medio_s": statistics.mean(tiempos[k]),
            "veces_mejor_global": None,
        }
    optimo_ref = min(min(res["HC1"]), min(res["HC"]), min(res["SA"]))
    for k in ("HC1", "HC", "SA"):
        resumen[k]["veces_mejor_global"] = sum(1 for x in res[k] if abs(x - optimo_ref) < 1e-6)
    resumen["referencia_mejor_global"] = optimo_ref
    resumen["n_semillas"] = len(SEMILLAS)

    # Ruta inicial aleatoria para comparar (semilla 0)
    random.seed(0)
    ini = list(range(N))
    random.shuffle(ini)
    resumen["costo_ruta_aleatoria"] = costo(ini)
    resumen["ruta_HC1"] = [NOMBRES[i] for i in best_hc1[0]]
    resumen["ruta_HC"] = [NOMBRES[i] for i in best_hc[0]]
    resumen["ruta_SA"] = [NOMBRES[i] for i in best_sa[0]]
    resumen["params_SA"] = PARAM_SA
    resumen["reinicios_HC"] = N_REINICIOS

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # ---- Grafica 1: convergencia ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(best_hc1[2], color="tab:green", marker="o", ms=3, label="HC simple")
    axs[0].plot(best_hc[2], color="tab:blue", label="HC con reinicios")
    axs[0].legend()
    axs[0].set_title("Convergencia Hill Climbing")
    axs[0].set_xlabel("Paso de mejora acumulado")
    axs[0].set_ylabel("Mejor costo (km)")
    axs[0].grid(alpha=0.3)
    axs[1].plot(best_sa[3], color="tab:orange", alpha=0.4, lw=0.6, label="Costo actual")
    axs[1].plot(best_sa[2], color="tab:red", lw=1.6, label="Mejor costo")
    axs[1].set_title("Convergencia Simulated Annealing")
    axs[1].set_xlabel("Iteracion")
    axs[1].set_ylabel("Costo (km)")
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("convergencia.png", dpi=170)
    plt.close(fig)

    # ---- Grafica 2: comparacion de resultados (boxplot) ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot([res["HC1"], res["HC"], res["SA"]], tick_labels=["HC simple", "HC 5 reinicios", "Sim. Annealing"])
    ax.set_ylabel("Costo final (km), 30 semillas")
    ax.set_title("Comparacion de resultados")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("comparacion.png", dpi=170)
    plt.close(fig)

    # ---- Grafica 3: rutas ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    graficar_ruta(axs[0], best_hc[0], f"Mejor ruta HC (reinicios): {best_hc[1]:.1f} km")
    graficar_ruta(axs[1], best_sa[0], f"Mejor ruta SA: {best_sa[1]:.1f} km")
    fig.tight_layout()
    fig.savefig("rutas.png", dpi=170)
    plt.close(fig)

    # ---- Grafica 4: sensibilidad de T0 y alpha en SA ----
    configs = [(50, 0.99), (500, 0.99), (2000, 0.995), (2000, 0.999)]
    sens = []
    for T0, a in configs:
        vals = [simulated_annealing(T0=T0, alpha=a, iteraciones=40000, semilla=s)[1]
                for s in range(15)]
        sens.append({"T0": T0, "alpha": a, "media": statistics.mean(vals), "mejor": min(vals)})
    with open("sensibilidad_sa.json", "w") as f:
        json.dump(sens, f, indent=2)

    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(sens, indent=2))


if __name__ == "__main__":
    main()
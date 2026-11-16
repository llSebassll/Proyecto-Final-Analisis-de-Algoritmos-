import os
import statistics
import time
import tracemalloc

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from scipy.spatial import ConvexHull

ox.settings.log_console = True
G = ox.graph_from_place("Quindío, Colombia", network_type="drive")

# --- Pesos de las aristas: distancia (m) y tiempo de viaje (s) ---
# Cada arista guarda dos pesos:
#   length       distancia en metros. OSMnx la calcula al construir el grafo,
#                sumando la distancia de gran círculo entre los puntos de la
#                geometría del tramo.
#   travel_time  tiempo en segundos = length / (speed_kph * 1000 / 3600).
#
# Velocidad de cada tramo (speed_kph), con ox.add_edge_speeds:
#   1. Si el tramo tiene la etiqueta `maxspeed` de OpenStreetMap, se usa ese
#      límite (en km/h; si viene en mph se convierte). Si tiene varios valores
#      (p. ej. "60|40" por carril, o varios tramos fusionados al simplificar el
#      grafo), se usa su media.
#   2. Si no tiene `maxspeed` (o el valor no es numérico), se le asigna la media
#      de los `maxspeed` de las aristas del mismo tipo de vía (`highway`:
#      primary, secondary, residential, ...) dentro del Quindío.
#   3. Si ninguna arista de ese tipo de vía tiene `maxspeed`, se usa la media
#      de las velocidades de los tipos que sí la tienen.
#
# Atención: el tiempo así calculado es una ESTIMACIÓN del modelo, no el tiempo
# real de recorrido. Supone que se circula todo el tramo al límite de velocidad
# (o a la velocidad imputada), sin tráfico, semáforos, giros, pendientes ni
# paradas.
etiquetado = {(u, v, k): "maxspeed" in d for u, v, k, d in G.edges(keys=True, data=True)}
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

aristas = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)
aristas["tipo"] = aristas["highway"].map(lambda x: x[0] if isinstance(x, list) else x)
aristas["con_maxspeed"] = aristas.index.map(etiquetado)

print("\n=== Pesos: distancia (m) y tiempo de viaje (s) ===")
n_con = int(aristas["con_maxspeed"].sum())
print(f"Aristas con maxspeed en OSM:  {n_con:,} de {len(aristas):,} "
      f"({n_con / len(aristas):.1%}); el resto usa velocidad imputada")
print(f"\n  {'tipo de vía':<16}{'aristas':>9}{'con maxspeed':>14}{'vel. imputada (km/h)':>22}")
for tipo, grupo in aristas.groupby("tipo"):
    sin = grupo[~grupo["con_maxspeed"]]
    if sin.empty:
        imputada = "-"
    else:
        imputada = f"{sin['speed_kph'].iloc[0]:.1f}"
        if not grupo["con_maxspeed"].any():
            imputada += " (media global)"
    print(f"  {tipo:<16}{len(grupo):>9,}{int(grupo['con_maxspeed'].sum()):>14,}{imputada:>22}")

print(f"\nDistancia total:  {aristas['length'].sum() / 1000:,.1f} km")
print(f"Tiempo total:     {aristas['travel_time'].sum() / 3600:,.1f} h "
      f"(estimación del modelo, no tiempo real)")

# --- Tamaño real de la red ---
# G es un MultiDiGraph: cada arista es dirigida (una vía doble sentido aparece
# como dos aristas u->v y v->u) y puede haber aristas paralelas entre el mismo par.
V = G.number_of_nodes()
E = G.number_of_edges()
pares_unicos = len({(u, v) for u, v, _ in G.edges(keys=True)})

MB = 1024 ** 2  # 1 MB = 2^20 bytes (MiB)

print("\n=== Red vial del Quindío (network_type='drive') ===")
print(f"Nodos (V):                    {V:,}")
print(f"Aristas dirigidas (E):        {E:,}")
print(f"Pares (u,v) distintos:        {pares_unicos:,}")
print(f"Grado medio de salida (E/V):  {E / V:.2f}")
print(f"Densidad E / V^2:             {E / V**2:.6%}")

# --- Matriz de adyacencia ---
# Supuesto: matriz densa V x V, una celda por cada par (u, v), exista o no la arista.
# Las aristas paralelas se colapsan (se guarda p. ej. la de menor longitud).
celdas = V * V
print("\n--- Matriz de adyacencia (V x V celdas) ---")
print(f"Celdas: {V:,} x {V:,} = {celdas:,}")
for etiqueta, bytes_celda in [
    ("bool / uint8 (1 B: solo existe o no)", 1),
    ("float32 (4 B: longitud en metros)", 4),
    ("float64 (8 B: longitud en metros, por defecto en NumPy)", 8),
]:
    print(f"  {etiqueta:<55} {celdas * bytes_celda / MB:>12,.2f} MB")

# --- Listas de adyacencia ---
# Formato sencillo: los nodos se escriben uno detrás de otro (0, 1, 2, ...).
# Cada nodo ocupa:
#   - su grado (cuántas aristas salen de él): int64, 8 B
#   - una entrada por arista: id del vecino + distancia (m) + tiempo (s)
# Tamaño total = V * 8 + E * bytes_entrada.
#
# Ejemplo (vecino int32 + distancia float32 + tiempo float32, little-endian),
# 4 nodos y 5 aristas, a 36 km/h (10 m/s) salvo 2 -> 3 a 72 km/h (20 m/s):
#   0 -> 1 (120 m, 12 s), 0 -> 2 (80 m, 8 s)
#   1 -> 2 (45 m, 4.5 s)
#   2 -> 0 (80 m, 8 s),   2 -> 3 (200 m, 10 s)
#   3 -> (sin aristas)
#
#   Pos.  Bytes (hex)                               Significado
#   0x00  02 00 00 00 00 00 00 00                   nodo 0: grado 2
#   0x08  01 00 00 00 | 00 00 F0 42 | 00 00 40 41     vecino 1, 120 m, 12 s
#   0x14  02 00 00 00 | 00 00 A0 42 | 00 00 00 41     vecino 2, 80 m, 8 s
#   0x20  01 00 00 00 00 00 00 00                   nodo 1: grado 1
#   0x28  02 00 00 00 | 00 00 34 42 | 00 00 90 40     vecino 2, 45 m, 4.5 s
#   0x34  02 00 00 00 00 00 00 00                   nodo 2: grado 2
#   0x3C  00 00 00 00 | 00 00 A0 42 | 00 00 00 41     vecino 0, 80 m, 8 s
#   0x48  03 00 00 00 | 00 00 48 43 | 00 00 20 41     vecino 3, 200 m, 10 s
#   0x54  00 00 00 00 00 00 00 00                   nodo 3: grado 0
#   Total: 4 * 8 + 5 * 12 = 92 bytes
#
# Lectura: se lee el grado (8 B) y luego grado * 12 B de vecinos, y así con el
# siguiente nodo hasta el final del archivo:
#   with open("grafo.bin", "rb") as f:
#       while dato := f.read(8):
#           grado = struct.unpack("<q", dato)[0]
#           vecinos = list(struct.iter_unpack("<iff", f.read(grado * 12)))
#
# Contra: para llegar al nodo u hay que recorrer todos los anteriores.
print("\n--- Listas de adyacencia (V cabeceras + E entradas) ---")
cabecera = 8  # grado del nodo (int64)
for etiqueta, bytes_entrada in [
    ("vecino int32 + distancia y tiempo float32 (12 B/entrada)", 4 + 4 + 4),
    ("vecino int64 + distancia y tiempo float64 (24 B/entrada)", 8 + 8 + 8),
]:
    total = V * cabecera + E * bytes_entrada
    print(f"  {etiqueta:<58} {total / MB:>12,.2f} MB")

# Referencia: la peor de las listas contra la mejor de las matrices
peor_lista = V * cabecera + E * 24
mejor_matriz = celdas * 1
print(f"\nLa matriz más compacta (1 B/celda) ocupa {mejor_matriz / peor_lista:,.0f} veces "
      f"más que la lista más costosa (24 B/entrada).")
print(f"Celdas de la matriz realmente usadas: {pares_unicos / celdas:.6%}")


# =============================================================================
# 3. COMPONENTE FUERTEMENTE CONEXA
# =============================================================================
# Una componente fuertemente conexa (SCC, por sus siglas en inglés) es un
# conjunto de nodos en el que, respetando el sentido de las vías, se puede ir
# de cualquiera a cualquier otro. En una red vial quedan nodos fuera de la
# componente principal por, entre otras cosas:
#   - calles de un solo sentido que "entran" a una zona de la que no se puede
#     salir (o al revés) dentro del área descargada,
#   - vías cortadas por el borde del área de descarga,
#   - errores de etiquetado en OpenStreetMap (un `oneway` mal puesto).
#
# ¿Qué hace ox.truncate.largest_component(G, strongly=True)?
# (OSMnx 2.1.1, archivo osmnx/truncate.py)
#   1. Llama a nx.is_strongly_connected(G). Si el grafo ya es una sola SCC,
#      lo devuelve tal cual.
#   2. Si no, recorre nx.strongly_connected_components(G) y se queda con la
#      más grande:  max(componentes, key=len).
#   3. Construye el subgrafo inducido por esos nodos y lo copia en un
#      MultiDiGraph nuevo:  nx.MultiDiGraph(G.subgraph(mayor)).
#
# Algoritmo interno (NetworkX 3.7, networkx/algorithms/components/
# strongly_connected.py): versión ITERATIVA del algoritmo de TARJAN, con las
# mejoras descritas por Tarjan y Zwick (2024). Cómo funciona:
#   - Hace un recorrido en profundidad (DFS) con pilas explícitas en vez de
#     recursión (así no se desborda la pila de Python en grafos grandes).
#   - A cada nodo le asigna, al visitarlo, un número de orden de visita.
#     Ese número se guarda en `lowlink[v]` y puede ir BAJANDO: si desde v
#     (o desde un descendiente suyo) hay una arista hacia un nodo visitado
#     antes que todavía no pertenece a ninguna componente cerrada, lowlink[v]
#     toma el número de ese nodo. Significa "desde v puedo volver hasta ahí".
#   - Cuando termina de explorar todos los sucesores de v:
#       * si lowlink[v] no bajó, v es la RAÍZ de una componente: v y todos
#         los nodos apilados después de él (que no pudieron volver más
#         arriba) forman una SCC y se sacan de la pila;
#       * si bajó, v pertenece a la componente de algún antecesor: se deja en
#         la pila y le pasa su lowlink al padre.
#   - Atajo: si ya se visitaron todos los nodos y se vuelve a la raíz del
#     DFS, todo el grafo es una sola SCC y termina sin recorrer el resto.
#
# Complejidad:
#   Tiempo   O(V + E): cada nodo se visita una vez y cada arista se revisa
#            una vez (al recorrer los sucesores de su origen). El
#            is_strongly_connected previo y la copia del subgrafo también son
#            O(V + E), así que la función completa sigue siendo lineal.
#   Memoria  O(V) adicional para el algoritmo (diccionario lowlink, conjunto
#            de nodos ya asignados y pilas del DFS), más O(V' + E') para la
#            copia de la componente resultante (V', E' = su tamaño).
#
# Medición: se descargan redes cada vez más grandes alrededor de la plaza de
# Bolívar de Armenia y se cronometra la función (mediana de varias
# repeticiones con time.perf_counter, sin logs para no medir la impresión en
# pantalla). Aparte se mide solo el algoritmo de Tarjan y el pico de memoria
# con tracemalloc.
#
# Nota: graph_from_point(dist=r) descarga un CUADRADO de lado 2r centrado en
# el punto (dist_type="bbox", el valor por defecto), no un círculo. Con 20 km
# el cuadrado se sale del Quindío y entra en Valle del Cauca y Risaralda.
#
# Para comprobar si el tiempo es lineal, la última columna divide el tiempo
# entre V + E: si la complejidad es O(V + E), ese valor debe mantenerse más
# o menos constante aunque la red crezca.
#
# Resultado de una ejecución (31/10/2026; los tiempos varían según el equipo):
#   red                  V + E   función (ms)   µs / (V+E)
#   r = 2 km             5.297       20,7          3,91
#   r = 5 km            14.007       55,5          3,96
#   r = 10 km           19.335       84,2          4,35
#   r = 20 km           28.807      137,2          4,76
#   Quindío completo    28.949      123,8          4,28
# La red crece ~5,5 veces y el tiempo ~6 veces; el costo por elemento se
# mantiene entre 3,9 y 4,8 µs, lo que coincide con O(V + E). Las pequeñas
# variaciones se deben a la caché del procesador, al recolector de basura de
# Python y a la cantidad de atributos que se copian por arista. La memoria
# pico (3 -> 17 MB) también crece en proporción al tamaño, como O(V + E).
# Casi todo el tiempo de la función se va en COPIAR el subgrafo (el algoritmo
# de Tarjan solo tarda ~1/14 del total).
PLAZA_BOLIVAR_ARMENIA = (4.5325725, -75.6730412)  # (lat, lon), según Nominatim
REPETICIONES = 5


def medir_componente(grafo):
    """Devuelve (mediana s función, mediana s solo Tarjan, pico de memoria MB)."""
    log_previo = ox.settings.log_console
    ox.settings.log_console = False
    t_funcion, t_tarjan = [], []
    for _ in range(REPETICIONES):
        t0 = time.perf_counter()
        ox.truncate.largest_component(grafo, strongly=True)
        t_funcion.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        max(nx.strongly_connected_components(grafo), key=len)
        t_tarjan.append(time.perf_counter() - t0)

    # tracemalloc ralentiza la ejecución, por eso la memoria se mide aparte
    tracemalloc.start()
    ox.truncate.largest_component(grafo, strongly=True)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ox.settings.log_console = log_previo
    return statistics.median(t_funcion), statistics.median(t_tarjan), pico / MB


def descargar_radio(radio_km, intentos=5, espera_s=60):
    """graph_from_point con reintentos: Overpass a veces responde 502/504
    cuando está saturado. Las descargas exitosas quedan en cache/, así que
    al volver a ejecutar el script ya no se piden otra vez."""
    for intento in range(1, intentos + 1):
        try:
            return ox.graph_from_point(PLAZA_BOLIVAR_ARMENIA, dist=radio_km * 1000,
                                       network_type="drive")
        except ox._errors.ResponseStatusCodeError as error:
            if intento == intentos:
                raise
            print(f"Overpass falló ({error}); reintento {intento + 1}/{intentos} "
                  f"en {espera_s} s")
            time.sleep(espera_s)


redes = []
for radio_km in (2, 5, 10, 20):
    redes.append((f"Plaza de Bolívar, r = {radio_km} km", descargar_radio(radio_km)))
redes.append(("Departamento del Quindío", G))

print("\n=== 3. Componente fuertemente conexa: tiempo de "
      "ox.truncate.largest_component(G, strongly=True) ===")
print(f"(mediana de {REPETICIONES} repeticiones)\n")
print(f"  {'red':<30}{'V':>8}{'E':>9}{'V + E':>9}{'función (ms)':>14}"
      f"{'solo Tarjan (ms)':>18}{'pico mem. (MB)':>16}{'µs / (V+E)':>12}")
for nombre, g in redes:
    v, e = g.number_of_nodes(), g.number_of_edges()
    t_f, t_t, pico = medir_componente(g)
    print(f"  {nombre:<30}{v:>8,}{e:>9,}{v + e:>9,}{t_f * 1e3:>14.1f}"
          f"{t_t * 1e3:>18.1f}{pico:>16.1f}{t_f / (v + e) * 1e6:>12.2f}")

# A partir de aquí el proyecto trabaja SOLO con la componente más grande.
# ¿Por qué conviene?
#   - Garantiza que exista ruta entre cualquier par de nodos respetando los
#     sentidos de circulación. Si un lugar quedara asociado a un nodo fuera
#     de la componente, Dijkstra/A* podrían no encontrar camino (o solo de
#     ida y no de vuelta), y habría que tratar ese caso en cada consulta.
#   - Al asociar los lugares a nodos de la componente (sección 4), todas las
#     rutas entre lugares del catálogo son posibles, en los dos sentidos.
#   - Elimina restos que no sirven para rutas (tramos que solo se pueden
#     recorrer en un sentido, pedazos cortados por el borde), así que el
#     grafo es más pequeño y los algoritmos de rutas procesan menos datos.
G_completo = G
G = ox.truncate.largest_component(G_completo, strongly=True)


# =============================================================================
# 4. UBICACIÓN DE LOS LUGARES
# =============================================================================
# Cada lugar del catálogo (lugares.csv: nombre, lat, lon en grados WGS84) se
# asocia al nodo de la componente más cercano EN LÍNEA RECTA:
#   1. Se pasan a UTM zona 18N (EPSG:32618) las coordenadas de los lugares y
#      de los nodos. UTM es una proyección plana en metros, así que la
#      distancia euclidiana  sqrt((x1-x2)^2 + (y1-y2)^2)  es directamente una
#      distancia en metros. El Quindío (~75.7° O) está dentro de la zona 18
#      (de 78° O a 72° O), así que la deformación es pequeña.
#   2. Para cada lugar se calcula la distancia a todos los nodos y se toma
#      la menor (búsqueda exhaustiva con NumPy: O(V) por lugar; con pocos
#      lugares y unos miles de nodos no hace falta un KD-tree).
#   3. Ese nodo será el origen o destino de las rutas hacia el lugar.
#
# Lugares a más de 500 m de su nodo:
#   NO se descartan: el nodo asignado sigue siendo el punto de la red vial
#   más cercano por el que se puede llegar al lugar. Se marcan con
#   lejos = True y se guarda la distancia, para que en las rutas se pueda
#   advertir que el último tramo (nodo -> lugar) no está en la red de vías
#   para vehículos (sendero, camino privado, vía sin cargar en OSM...).
#   Para conocer la causa también se calcula la distancia al nodo más
#   cercano de la red COMPLETA (antes de filtrar la componente): si es mucho
#   menor, la vía cercana existía pero se eliminó al quedarnos con la
#   componente fuertemente conexa.
UMBRAL_LEJOS_M = 500
a_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)


def nodos_utm(grafo):
    """Ids de los nodos y sus coordenadas UTM 18N como arreglos NumPy."""
    ids = np.array(list(grafo.nodes))
    lon = np.array([grafo.nodes[n]["x"] for n in ids])
    lat = np.array([grafo.nodes[n]["y"] for n in ids])
    x, y = a_utm.transform(lon, lat)
    return ids, np.asarray(x), np.asarray(y)


def mas_cercano(ids, xs, ys, x, y):
    """Nodo más cercano a (x, y) en línea recta y su distancia en metros."""
    d = np.hypot(xs - x, ys - y)
    i = int(np.argmin(d))
    return int(ids[i]), float(d[i])


lugares = pd.read_csv("lugares.csv")
lugares["x_utm"], lugares["y_utm"] = a_utm.transform(lugares["lon"].to_numpy(),
                                                     lugares["lat"].to_numpy())
ids_c, xs_c, ys_c = nodos_utm(G)
ids_t, xs_t, ys_t = nodos_utm(G_completo)
nodo, dist, dist_completa = [], [], []
for x, y in zip(lugares["x_utm"], lugares["y_utm"]):
    n, d = mas_cercano(ids_c, xs_c, ys_c, x, y)
    nodo.append(n)
    dist.append(d)
    dist_completa.append(mas_cercano(ids_t, xs_t, ys_t, x, y)[1])
lugares["nodo"] = nodo
lugares["dist_nodo_m"] = dist
lugares["dist_red_completa_m"] = dist_completa
lugares["lejos"] = lugares["dist_nodo_m"] > UMBRAL_LEJOS_M

print("\n=== 4. Ubicación de los lugares (nodo más cercano, UTM 18N) ===")
print(f"  {'lugar':<40}{'nodo':>13}{'dist. (m)':>11}")
for _, l in lugares.iterrows():
    aviso = "   <-- más de 500 m" if l["lejos"] else ""
    print(f"  {l['nombre']:<40}{l['nodo']:>13}{l['dist_nodo_m']:>11,.1f}{aviso}")

lejanos = lugares[lugares["lejos"]]
print(f"\nLugares a más de {UMBRAL_LEJOS_M} m de su nodo: {len(lejanos)}")
for _, l in lejanos.iterrows():
    if l["dist_red_completa_m"] < l["dist_nodo_m"] - 1:
        causa = (f"en la red completa había un nodo a {l['dist_red_completa_m']:,.0f} m, "
                 f"pero quedó fuera de la componente fuertemente conexa")
    else:
        causa = "no hay vías para vehículos más cerca, ni siquiera en la red completa"
    print(f"  - {l['nombre']}: {l['dist_nodo_m']:,.0f} m; {causa}. "
          f"Se mantiene su nodo y se marca lejos = True.")


# =============================================================================
# 5. GRAFO PROCESADO
# =============================================================================
# Se guardan en la carpeta procesado/:
#   quindio_scc.graphml   la componente fuertemente conexa con sus pesos por
#                         arista (length en m, speed_kph, travel_time en s)
#                         y las coordenadas de los nodos.
#   lugares_nodos.csv     el catálogo con su nodo asociado, la distancia al
#                         nodo y la marca `lejos`.
# GraphML es el formato que usan OSMnx/NetworkX: un XML de texto que guarda
# todos los atributos de nodos y aristas. Así, los requerimientos siguientes
# pueden cargar el grafo sin descargar de nuevo la red ni recalcular nada:
#
#   G = ox.load_graphml("procesado/quindio_scc.graphml")
#   lugares = pd.read_csv("procesado/lugares_nodos.csv")
#
# (load_graphml convierte length, speed_kph y travel_time de vuelta a float.)
os.makedirs("procesado", exist_ok=True)
ox.save_graphml(G, "procesado/quindio_scc.graphml")
lugares.to_csv("procesado/lugares_nodos.csv", index=False)

v0, e0 = G_completo.number_of_nodes(), G_completo.number_of_edges()
v1, e1 = G.number_of_nodes(), G.number_of_edges()
print("\n=== 5. Grafo procesado ===")
print(f"  {'':<10}{'antes':>10}{'después':>10}{'eliminados':>20}")
print(f"  {'Nodos':<10}{v0:>10,}{v1:>10,}{v0 - v1:>11,} ({(v0 - v1) / v0:.1%})")
print(f"  {'Aristas':<10}{e0:>10,}{e1:>10,}{e0 - e1:>11,} ({(e0 - e1) / e0:.1%})")
print("Guardado en procesado/quindio_scc.graphml y procesado/lugares_nodos.csv")





# =============================================================================
# DIJKSTRA
# =============================================================================


# pred = predecesor
def reconstruir_ruta(pred, origen, destino):
    "Reconstruye el camino saltando hacia atrás usando el diccionario de predecesores."
    ruta = []
    actual = destino
    
    while actual is not None:
        ruta.append(actual)
        if actual == origen:
            break
        actual = pred.get(actual)

    # Si la ruta está vacía o el último nodo no es el origen, es inalcanzable
    if not ruta or ruta[-1] != origen:
        return []

    # Invertir la ruta para que esta vaya de origen a destino
    ruta.reverse() 
    return ruta


# =============================================================================
# 1. ESTRUCTURA DE APOYO
# =============================================================================
class ColaPrioridad:
    def __init__(self):
        self.heap = []

    def push(self, elemento):
        "Inserta un elemento y lo sube hasta su posición correcta."
        self.heap.append(elemento)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        "Extrae y retorna el elemento con menor valor (la raíz)."
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        
        raiz = self.heap[0]
        # Movemos el último elemento a la raíz y lo hundimos
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return raiz

    def _sift_up(self, idx):
        padre = (idx - 1) // 2
        # Comparamos la distancia (índice 0 de la tupla)
        if idx > 0 and self.heap[idx][0] < self.heap[padre][0]:
            self.heap[idx], self.heap[padre] = self.heap[padre], self.heap[idx]
            self._sift_up(padre)

    def _sift_down(self, idx):
        menor = idx
        izq = 2 * idx + 1
        der = 2 * idx + 2
        
        if izq < len(self.heap) and self.heap[izq][0] < self.heap[menor][0]:
            menor = izq
        if der < len(self.heap) and self.heap[der][0] < self.heap[menor][0]:
            menor = der
            
        if menor != idx:
            self.heap[idx], self.heap[menor] = self.heap[menor], self.heap[idx]
            self._sift_down(menor)

    def is_empty(self):
        return len(self.heap) == 0


# =============================================================================
# 2. ALGORITMO DIJKSTRA
# =============================================================================
def reconstruir_ruta(pred, origen, destino):
    """Reconstruye el camino saltando hacia atrás."""
    ruta = []
    actual = destino
    while actual is not None:
        ruta.append(actual)
        if actual == origen:
            break
        actual = pred.get(actual)
        
    if not ruta or ruta[-1] != origen:
        return []
    ruta.reverse() 
    return ruta

def dijkstraGrafoReal(grafo, origen, destino):
    # Inicializamos distancias en infinito para todos los nodos del grafo real
    dist = {nodo: float("inf") for nodo in grafo.nodes}
    pred = {nodo: None for nodo in grafo.nodes}
    
    # Aseguramos compatibilidad de tipos (string vs int)
    origen = str(origen) if str(origen) in dist else int(origen)
    destino = str(destino) if str(destino) in dist else int(destino)
    
    dist[origen] = 0
    nodos_explorados = 0
    
    # Instanciamos nuestra propia cola de prioridad (reemplaza pq = [])
    pq = ColaPrioridad()
    pq.push((0, origen))  # Tupla: (distancia_acumulada, nodo)

    while not pq.is_empty():
        # Reemplaza a heapq.heappop()
        d, nodo = pq.pop()

        # Invariante: descartar entradas obsoletas
        if d > dist[nodo]:
            continue  

        nodos_explorados += 1

        # Parada temprana
        if nodo == destino:
            break

        # Adaptación obligatoria para grafos MultiDiGraph de OSMnx
        for vecino, aristas in grafo[nodo].items():
            # Extraemos el tiempo de viaje de la primera vía paralela (0)
            peso = float(aristas[0].get('travel_time', 0))
            
            nueva_dist = d + peso
            
            if nueva_dist < dist[vecino]:
                dist[vecino] = nueva_dist
                pred[vecino] = nodo
                # Reemplaza a heapq.heappush()
                pq.push((nueva_dist, vecino))
                
    ruta = reconstruir_ruta(pred, origen, destino)
    return dist[destino], ruta, nodos_explorados


def dijkstraLineal(grafo, origen, destino):
    dist = {nodo: float("inf") for nodo in grafo.nodes}
    pred = {nodo: None for nodo in grafo.nodes}
    
    origen = str(origen) if str(origen) in dist else int(origen)
    destino = str(destino) if str(destino) in dist else int(destino)
    
    dist[origen] = 0
    noVisitados = set(grafo.nodes)
    nodosExplorados = 0

    while noVisitados:
        # Búsqueda O(V): Iterar sobre todos los no visitados buscando el menor
        nodoActual = min(noVisitados, key=lambda n: dist[n])
        
        if dist[nodoActual] == float("inf"):
            break

        noVisitados.remove(nodoActual)
        nodosExplorados += 1

        if nodoActual == destino:
            break

        # Adaptación para MultiDiGraph
        for vecino, aristas in grafo[nodoActual].items():
            peso = float(aristas[0].get('travel_time', 0))
            nuevaDistancia = dist[nodoActual] + peso
            
            if nuevaDistancia < dist[vecino]:
                dist[vecino] = nuevaDistancia
                pred[vecino] = nodoActual

    ruta = reconstruir_ruta(pred, origen, destino)
    return dist[destino], ruta, nodosExplorados



import pandas as pd
import itertools
import time

def cargarMatriz(rutaArchivo):
    df = pd.read_csv(rutaArchivo, index_col=0)
    # Convertimos el DataFrame a un diccionario bidimensional para acceso rápido O(1)
    matrizDict = df.to_dict(orient="index")
    return matrizDict, list(df.index)

def fuerzaBruta(matriz, nodosVisitar, nodoInicio):
    """Prueba todas las permutaciones posibles O(k!)."""
    mejorTiempo = float('inf')
    mejorRuta = []
    
    # Generamos todas las permutaciones posibles de los nodos a visitar
    for permutacion in itertools.permutations(nodosVisitar):
        tiempoActual = 0
        nodoActual = nodoInicio
        
        # Recorremos la permutación
        for destino in permutacion:
            tiempoActual += matriz[nodoActual][destino]
            nodoActual = destino
            
        # Sumamos el tiempo de regreso al nodo de inicio
        tiempoActual += matriz[nodoActual][nodoInicio]
        
        if tiempoActual < mejorTiempo:
            mejorTiempo = tiempoActual
            mejorRuta = list(permutacion)
            
    # Formateamos la ruta completa
    rutaCompleta = [nodoInicio] + mejorRuta + [nodoInicio]
    return mejorTiempo, rutaCompleta

def heldKarp(matriz, nodosVisitar, nodoInicio):
    """Algoritmo de Programación Dinámica O(k^2 * 2^k) usando máscaras de bits."""
    n = len(nodosVisitar)
    
    # Para facilitar los índices con los bits, mapeamos cada nodo a un número del 0 al n-1
    indiceANodo = {i: nodosVisitar[i] for i in range(n)}
    
    # memo: {(mascaraVisitados, nodoActual): (tiempoMinimo, nodoAnterior)}
    memo = {}
    
    # Caso base: ir desde el inicio a cada uno de los nodos individuales
    for i in range(n):
        mascaraInicial = 1 << i # Pone un 1 en la posición i
        nodoDestino = indiceANodo[i]
        memo[(mascaraInicial, i)] = (matriz[nodoInicio][nodoDestino], None)
        
    # Llenamos la tabla de programación dinámica iterando sobre tamaños de subconjuntos
    for tamanoSubconjunto in range(2, n + 1):
        # Iteramos sobre todas las posibles combinaciones (máscaras) de este tamaño
        for mascara in range(1, 1 << n):
            if mascara.bit_count() != tamanoSubconjunto:
                continue
                
            # Tratamos de que el último nodo visitado sea 'j'
            for j in range(n):
                if not (mascara & (1 << j)):
                    continue # 'j' no está en este subconjunto
                    
                mascaraSinJ = mascara ^ (1 << j)
                tiempoMinimo = float('inf')
                nodoAnterior = None
                
                # Buscamos el mejor nodo 'i' anterior a 'j'
                for i in range(n):
                    if i == j or not (mascara & (1 << i)):
                        continue
                        
                    tiempoPrevio, _ = memo[(mascaraSinJ, i)]
                    tiempoTransicion = matriz[indiceANodo[i]][indiceANodo[j]]
                    tiempoTotal = tiempoPrevio + tiempoTransicion
                    
                    if tiempoTotal < tiempoMinimo:
                        tiempoMinimo = tiempoTotal
                        nodoAnterior = i
                        
                memo[(mascara, j)] = (tiempoMinimo, nodoAnterior)
                
    # Paso final: encontrar el camino más corto sumando el regreso al origen
    mascaraCompleta = (1 << n) - 1
    tiempoOptimo = float('inf')
    ultimoNodo = None
    
    for j in range(n):
        tiempoPrevio, _ = memo[(mascaraCompleta, j)]
        tiempoRegreso = matriz[indiceANodo[j]][nodoInicio]
        tiempoTotal = tiempoPrevio + tiempoRegreso
        
        if tiempoTotal < tiempoOptimo:
            tiempoOptimo = tiempoTotal
            ultimoNodo = j
            
    # Reconstruir la ruta yendo hacia atrás a través del diccionario memo
    rutaReconstruida = []
    mascaraActual = mascaraCompleta
    nodoActual = ultimoNodo
    
    while nodoActual is not None:
        rutaReconstruida.append(indiceANodo[nodoActual])
        _, nodoAnterior = memo[(mascaraActual, nodoActual)]
        mascaraActual = mascaraActual ^ (1 << nodoActual)
        nodoActual = nodoAnterior
        
    rutaReconstruida.reverse()
    rutaCompleta = [nodoInicio] + rutaReconstruida + [nodoInicio]
    
    return tiempoOptimo, rutaCompleta


def dijkstraTodasLasDistancias(grafo, origen):
    """Versión que calcula la distancia a TODOS los nodos sin detenerse, usando la clase ColaPrioridad propia."""
    dist = {nodo: float("inf") for nodo in grafo.nodes}
    
    # Asegurarnos de que el origen exista y sea del tipo correcto (str o int)
    origen = str(origen) if str(origen) in dist else int(origen)
    
    dist[origen] = 0
    
    # Instanciamos nuestra propia estructura en lugar de la lista de heapq
    cola = ColaPrioridad()  
    cola.push((0, origen))  

    while not cola.is_empty():
        # Extraemos el de menor distancia usando nuestro método pop()
        distanciaActual, nodoActual = cola.pop()

        # Invariante: descartamos registros desactualizados
        if distanciaActual > dist[nodoActual]:
            continue

        # NO hay condicional de parada porque necesitamos el grafo completo

        # Iteramos sobre los vecinos en el MultiDiGraph
        for vecino, aristas in grafo[nodoActual].items():
            peso = float(aristas[0].get('travel_time', 0))
            nuevaDist = distanciaActual + peso
            
            if nuevaDist < dist[vecino]:
                dist[vecino] = nuevaDist
                # Insertamos a nuestra cola personalizada
                cola.push((nuevaDist, vecino))
                
    return dist

def generarMatrizTiempos():
    print("Cargando el grafo real y el catálogo de lugares para la Matriz...")
    try:
        grafoReal = ox.load_graphml("procesado/quindio_scc.graphml")
        lugares = pd.read_csv("procesado/lugares_nodos.csv")
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos base en 'procesado/'.")
        return
    
    nombresLugares = lugares["nombre"].tolist()
    matrizDatos = []
    
    print(f"Calculando matriz de tiempos asimétrica {len(nombresLugares)}x{len(nombresLugares)}...")
    inicioTiempo = time.perf_counter()
    
    for i, origenRow in lugares.iterrows():
        nodoOrigen = origenRow["nodo"]
        nombreOrigen = origenRow["nombre"]
        
        # OJO: Asegúrate de tener la función dijkstra_todas_las_distancias
        # (o dijkstraTodasLasDistancias) definida en tu código para que esto funcione
        distanciasDesdeOrigen = dijkstraTodasLasDistancias(grafoReal, nodoOrigen)
        
        fila = {"Origen": nombreOrigen}
        
        for j, destinoRow in lugares.iterrows():
            nodoDestino = destinoRow["nodo"]
            nombreDestino = destinoRow["nombre"]
            
            tiempoSegundos = distanciasDesdeOrigen.get(nodoDestino, float('inf'))
            fila[nombreDestino] = tiempoSegundos
            
        matrizDatos.append(fila)
        print(f"[{i+1}/{len(lugares)}] Procesado origen: {nombreOrigen}")
        
    dfMatriz = pd.DataFrame(matrizDatos)
    rutaSalida = "procesado/matriz_tiempos.csv"
    dfMatriz.to_csv(rutaSalida, index=False)
    
    finTiempo = time.perf_counter()
    print(f"\n✅ Matriz guardada exitosamente en: {rutaSalida}")
    print(f"⏱️ Tiempo total de generación: {(finTiempo - inicioTiempo):.2f} segundos")
generarMatrizTiempos()


# Heuristica Voras
def heuristicaVoraz(matriz, nodosVisitar, nodoInicio):

    nodosPendientes = set(nodosVisitar)
    nodoActual = nodoInicio
    tiempoTotal = 0
    rutaIntermedia = []

    while nodosPendientes:
        menorTiempo = float('inf')
        siguienteNodo = None

        #Busqueda del vecino mas cercano
        for candidato in nodosPendientes:
            tiempoCandidato = matriz[nodoActual][candidato]
            siguienteNodo = candidato

            if tiempoCandidato < menorTiempo:
                menorTiempo = tiempoCandidato
                siguienteNodo = candidato

        # Actualizar el recorrido
        tiempoTotal += menorTiempo
        rutaIntermedia.append(siguienteNodo)
        nodosPendientes.remove(siguienteNodo)
        nodoActual = siguienteNodo

    # Sumar el tiempo de regreso al origen
    tiempoRegreso = matriz[nodoActual][nodoInicio]    
    tiempoTotal += tiempoRegreso

    # Generar la ruta completa
    rutaCompleta = [nodoInicio] + rutaIntermedia + [nodoInicio]

    return tiempoTotal, rutaCompleta


# Comprobar Itinerario

def comprobarItinerario(matriz, dfLugares, nodoIicio, atractivos):
    
    limiteTiempo = 11 * 60
    duraciones = dict(zip(dfLugares['nombre'], dfLugares['duracionVisita']))

    nodosActuales = list(atractivos)

    while True:
        # Calcular la ruta optima del viaje
        tiempoViaje, rutaOptima = heldKarp(matriz, nodosActuales, nodoIicio)

        # Calcular el tiempo en cada destino
        tiempoVisitas = sum(duraciones.get(nodo, 0) for nodo in nodosActuales) 

        # Tiempo total
        tiempoTotal = tiempoViaje/60 + tiempoVisitas

        print(f"-> Evaluando ruta con {len(nodosActuales)} atractivos...")
        print(f"   🚗 Tiempo de viaje:   {tiempoViaje /60 :.1f}  min")
        print(f"   📸 Tiempo de visitas: {tiempoVisitas:.1f} min")
        print(f"   ⏱️ Tiempo Total:      {tiempoTotal:.1f} min (Límite: {limiteTiempo} min)")

        if tiempoTotal < limiteTiempo:
            print("\n✅ ¡El itinerario es VÁLIDO y cabe en 11 horas!")
            print(f"🛣️ Recorrido final: {' -> '.join(rutaOptima)}\n")
            break
        else:
            print("\n❌ El itinerario excede las 11 horas.")

            # Jistificacion para eliminar: Quitamos el lugar que quita mas tiempo en ir 
            atractivoAEliminar = max(nodosActuales, key=lambda x: matriz[nodoIicio] [x]) 
            duracionEliminado = matriz[nodoIicio][atractivoAEliminar]
            
            print(f" Justificación: Se elimina '{atractivoAEliminar}' porque es el lugar "
                            f"que toma más tiempo de viaje en llegar ({duracionEliminado / 60:.1f}) min ")

        nodosActuales.remove(atractivoAEliminar)

        if not nodosActuales:
            print("No es posible visitar ningun atractivo en el tiempo dado")
            break


def ejecutarItinerario():
    print("\nITINERARIO DE UN DÍA (11 HORAS) ===")
    matriz, _ = cargarMatriz("procesado/matriz_tiempos.csv")
    
    # Cargamos el catálogo para obtener las duraciones
    try:
        dfLugares = pd.read_csv("procesado/lugares_nodos.csv")
    except FileNotFoundError:
        dfLugares = pd.read_csv("lugares.csv")
        
    nodoInicio = "Armenia"
    
    dfAtractivos = dfLugares[dfLugares['categoria'] == 'Atractivo']
    
    # El requerimiento pide elegir entre 6 y 8 atractivos. Tomaremos 7 de la lista.
    atractivosSeleccionados = dfAtractivos['nombre'].head(7).tolist()
    
    if len(atractivosSeleccionados) < 6:
        print(f"⚠️ Advertencia: Solo encontré {len(atractivosSeleccionados)} atractivos en tu catálogo con 'duracion_visita' > 0.")
        print("Asegúrate de que tu archivo CSV tenga bien llena la columna 'duracion_visita' para los parques y museos.")
        
    print(f"📍 Lugar de partida: {nodoInicio}")
    print(f"🎯 Atractivos iniciales seleccionados ({len(atractivosSeleccionados)}): {atractivosSeleccionados}\n")
    
    # Llamamos a nuestro comprobador
    comprobarItinerario(matriz, dfLugares, nodoInicio, atractivosSeleccionados)

ejecutarItinerario()



def probarAlgoritmosTSP():
    matriz, listaLugares = cargarMatriz("procesado/matriz_tiempos.csv")
    
    # Configuramos el experimento (Requerimiento 4.1 y 4.4)
    lugarInicio = "Armenia" # Nodo origen por defecto según las instrucciones
    
    # Elegimos k atractivos aleatorios o específicos para probar
    cantidadK = 5
    lugaresTuristicos = [l for l in listaLugares if l != lugarInicio]
    nodosSeleccionados = lugaresTuristicos[:cantidadK] # Tomamos los k primeros para este ejemplo
    
    print(f"=== Prueba TSP con k={cantidadK} atractivos ===")
    print(f"Origen: {lugarInicio}")
    print(f"Atractivos: {nodosSeleccionados}\n")
    
    # Prueba Fuerza Bruta
    inicioReloj = time.perf_counter()
    tiempoFb, rutaFb = fuerzaBruta(matriz, nodosSeleccionados, lugarInicio)
    finReloj = time.perf_counter()
    print(f"[Fuerza Bruta] O(k!)")
    print(f"Tiempo de ejecución: {(finReloj - inicioReloj)*1000:.2f} ms")
    print(f"Tiempo de viaje: {tiempoFb / 60:.1f} minutos")
    print(f"Ruta: {' -> '.join(rutaFb)}\n")
    
    # Prueba Held-Karp
    inicioReloj = time.perf_counter()
    tiempoHk, rutaHk = heldKarp(matriz, nodosSeleccionados, lugarInicio)
    finReloj = time.perf_counter()
    print(f"[Held-Karp] O(k^2 * 2^k)")
    print(f"Tiempo de ejecución: {(finReloj - inicioReloj)*1000:.2f} ms")
    print(f"Tiempo de viaje: {tiempoHk / 60:.1f} minutos")
    print(f"Ruta: {' -> '.join(rutaHk)}\n")

    # Prueba Heuristca Voraz
    inicioReloj = time.perf_counter()
    tiempoVoraz, rutaVoraz = heuristicaVoraz(matriz, nodosSeleccionados, lugarInicio)
    finReloj = time.perf_counter()
    print(f"[Heurística Voraz] O(k^2)")
    print(f"Tiempo de ejecución: {(finReloj - inicioReloj)*1000:.2f} ms")
    print(f"Tiempo de viaje: {tiempoVoraz / 60:.1f} minutos")
    print(f"Ruta: {' -> '.join(rutaVoraz)}")

    # Calculo de calidad
    errorPorcentual = ((tiempoVoraz - tiempoHk) / tiempoHk) * 100
    print(f"⚠️ Diferencia contra el óptimo (Held-Karp): +{errorPorcentual:.2f}%\n")
    
    # Verificación
    assert abs(tiempoFb - tiempoHk) < 0.1, "Error: Los tiempos óptimos no coinciden"
    print("✅ Verificación exitosa: Ambos algoritmos exactos encontraron el mismo tiempo óptimo.")

# Ejecutar la prueba
probarAlgoritmosTSP()





#Prueba de escritorio 
def probarDatosReales():
    print("Cargando el grafo real y el catálogo de lugares...")
    try:
        # Cargamos el grafo desde el archivo generado en el Requerimiento 1
        G_real = ox.load_graphml("procesado/quindio_scc.graphml")
        
        # Cargamos el catálogo de lugares
        lugares = pd.read_csv("procesado/lugares_nodos.csv")
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos en 'procesado/'.")
        print("Debes ejecutar primero el código del Requerimiento 1 para generarlos.")
        return

    # Tomamos el primer lugar (ej. Armenia) y el último lugar del catálogo para la prueba
    lugarOrigen = lugares.iloc[0]
    lugarDestino = lugares.iloc[-1]
    
    nodoOrigen = lugarOrigen["nodo"]
    nodoDestino = lugarDestino["nodo"]
    
    print(f"\n🗺️ Calculando ruta desde: {lugarOrigen['nombre']} hasta {lugarDestino['nombre']}")
    print(f"ID Nodo Origen: {nodoOrigen} | ID Nodo Destino: {nodoDestino}")
    
    # Medir el tiempo de ejecución exacto (Requerimiento 6 del proyecto)
    inicioTiempo = time.perf_counter()
    
    # Ejecutamos el algoritmo
    tiempo_viaje_segundos, ruta, explorados = dijkstraGrafoReal(G_real, nodoOrigen, nodoDestino)
    
    fin_tiempo = time.perf_counter()
    tiempo_ejecucion_ms = (fin_tiempo - inicioTiempo) * 1000

    # Imprimir resultados
    print("\n--- RESULTADOS DIJKSTRA COLA PRIORIDAD ---")
    print(f"Tiempo de cómputo: {tiempo_ejecucion_ms:.2f} ms")
    print(f"Nodos explorados: {explorados:,}")
    print(f"Tiempo estimado de viaje: {tiempo_viaje_segundos / 60:.1f} minutos")
    print(f"Cantidad de tramos (calles) en la ruta: {len(ruta)}")
    
    if len(ruta) > 0:
        print(f"Ejemplo de los primeros 5 nodos de la ruta: {ruta[:5]}")
    else:
        print("❌ No se encontró una ruta válida.")

# Ejecutar la prueba
probarDatosReales()

import matplotlib.pyplot as plt

def ejecutarExperimentoR2():
    print("\n EXPERIMENTO REQUERIMIENTO 2: DIJSKTRA ====")
    print("Cargando grafo y catálogo...")
    grafo = ox.load_graphml("procesado/quindio_scc.graphml")
    lugares = pd.read_csv("procesado/lugares_nodos.csv")

    #1. Separar 10 pares cercanos()
    paresCercanos = []
    for _, grupo in lugares.groupby('municipio'):
        nodosMunicipio =  grupo['nodo'].tolist()
        if len(nodosMunicipio) >=2:
            for i in range(1, len(nodosMunicipio)):
                paresCercanos.append((nodosMunicipio[0], nodosMunicipio[i]))
                if len(paresCercanos) == 10:break
        if len(paresCercanos)== 10:break

    #Completar hasta tener 20 pares en total mezclando al azar
    nodosTotal = lugares['nodo'].tolist()
    paresTotal = paresCercanos.copy()
    for i in range(len(nodosTotal)-1):
        if len(paresTotal) >=20:break
        par = (nodosTotal[i], nodosTotal[-i-1])
        if par not in paresTotal:
            paresTotal.append(par)

    resultadosCp = []
    resultadosLin = []
    cocientes = []

    print("Iniciando mediciones...")
    for idx, (origen, destino) in enumerate(paresTotal):
        #medi cola de prioridad
        inicio = time.perf_counter()
        _, _, expCp = dijkstraGrafoReal(grafo, origen, destino)
        tiempoCp = (time.perf_counter() - inicio) * 1000
        resultadosCp.append((expCp, tiempoCp))

        #medir lineal 
        if idx < 10:
            inicio = time.perf_counter()
            _, _, expLin = dijkstraLineal(grafo, origen, destino)
            tiempoLin = (time.perf_counter() - inicio) * 1000
            resultadosLin.append((expLin, tiempoLin))

            cociente = tiempoLin / tiempoCp if tiempoCp > 0 else 0
            cocientes.append(cociente)
            print(f"Par {idx+1} (Cercano) | Nodos exp: {expCp} | CP: {tiempoCp:.1f}ms | Lin: {tiempoLin:.1f}ms | Cociente: {cociente:.1f}x")
        else:
            print(f"Par {idx+1} (Lejano)  | Nodos exp: {expCp} | CP: {tiempoCp:.1f}ms")

    print("\n--- ESTADÍSTICAS DEL COCIENTE (Lineal / Cola Prioridad) ---")
    print(f"Mínimo:  {min(cocientes):.2f}x")
    print(f"Máximo:  {max(cocientes):.2f}x")
    print(f"Mediana: {statistics.median(cocientes):.2f}x")

    # 2. Generar Gráfica
    xCp, yCp = zip(*resultadosCp)
    xLin, yLin = zip(*resultadosLin)

    plt.figure(figsize=(10, 6))
    plt.scatter(xCp, yCp, color='blue', label='O((V+E) log V) - Cola Prioridad', alpha=0.7)
    plt.scatter(xLin, yLin, color='red', label='O(V^2) - Búsqueda Lineal', alpha=0.7)
    
    plt.title("Rendimiento Dijkstra: Tiempo vs Nodos Explorados")
    plt.xlabel("Nodos Explorados")
    plt.ylabel("Tiempo de Ejecución (ms)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("procesado/grafica_dijkstra.png", dpi=300)
    print("✅ Gráfica guardada en 'procesado/grafica_dijkstra.png'")
# Llama a esta función al final de tu script
ejecutarExperimentoR2()


# =============================================================================
# REQUERIMIENTO 3: ENVOLVENTE CONVEXA (DIVIDE Y VENCERÁS)
# =============================================================================

def determinanteVectorial(p1, p2, p3):

    "Calcula el producto cruzado (determinante) de los vectores p1->p2 y p1->p3."
    "Retorna > 0 si p3 está a la 'izquierda' de la línea dirigida de p1 a p2."
    
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def quickHullRecursivo(puntos, p1, p2):
    if not puntos:
        return []

    #encontrar el punto mas alejado de la linea formada por p1 y p2
    puntoLejano = max(puntos, key= lambda p: abs(determinanteVectorial(p1, p2, p)))

    #Dividir los puntos restantes en dos nuevos subconjuntos
    puntosIzquierda1 = [p for p in puntos if determinanteVectorial(p1, puntoLejano, p) > 0]
    puntosIzquierda2 = [p for p in puntos if determinanteVectorial(puntoLejano, p2, p) > 0]

    # Recursividad
    bordeIzquierdo = quickHullRecursivo(puntosIzquierda1, p1, puntoLejano)
    bordeDerecho = quickHullRecursivo(puntosIzquierda2, puntoLejano, p2)

    #convinar resultados
    return bordeIzquierdo + [puntoLejano] + bordeDerecho

def envolventeConvexa(puntos):
    if len(puntos) <+ 2:
        return puntos

    #Encontrar los extremos en el eje x
    puntoMinX = min(puntos, key=lambda p: p[0]) 
    puntosMaxX = max(puntos, key=lambda p: p[0]) 

    #Separar los puntos que estan arriba y abajo
    puntosArriba = [p for p in puntos if determinanteVectorial(puntoMinX, puntosMaxX,p) > 0]
    puntosAbajo = [p for p in puntos if determinanteVectorial(puntosMaxX, puntoMinX, p) > 0]

    #Llamadas recursivas
    hullArriba = quickHullRecursivo(puntosArriba, puntoMinX, puntosMaxX)
    hullAbajo = quickHullRecursivo(puntosAbajo, puntosMaxX, puntoMinX)

    return [puntoMinX] + hullArriba + [puntosMaxX] + hullAbajo


def ejecutarRequerimiento3():
    print("\n" + "="*50)
    print("=== REQUERIMIENTO 3: ENVOLVENTE CONVEXA (QUICKHULL) ===")
    print("="*50)
    
    # Cargar coordenadas (longitud como X, latitud como Y)
    dfLugares = pd.read_csv("procesado/lugares_nodos.csv")
    puntosLista = list(zip(dfLugares['lon'], dfLugares['lat']))
    # Eliminamos duplicados por si hay lugares con la misma coordenada exacta
    puntosUnicos = list(set(puntosLista)) 
    
    print(f"Calculando envolvente convexa para {len(puntosUnicos)} lugares...\n")
    
    # Divide y Vencerás
    inicioPropio = time.perf_counter()
    hullPropio = envolventeConvexa(puntosUnicos)
    tiempoPropio = (time.perf_counter() - inicioPropio) * 1000
    
    # Algoritmo de SciPy 
    inicioScipy = time.perf_counter()
    puntosArray = np.array(puntosUnicos)
    # SciPy devuelve los índices de los vértices que forman el borde
    indicesScipy = ConvexHull(puntosArray).vertices
    hullScipy = [tuple(puntosArray[i]) for i in indicesScipy]
    tiempoScipy = (time.perf_counter() - inicioScipy) * 1000
    
    # --- 3. Resultados y Comparación ---
    print("--- RESULTADOS ---")
    print(f"Algoritmo Propio (O(n log n)): {tiempoPropio:.3f} ms | Vértices: {len(hullPropio)}")
    print(f"Librería SciPy (Qhull):        {tiempoScipy:.3f} ms | Vértices: {len(hullScipy)}")
    
    # Verificación estricta usando conjuntos (ignorando el orden en que se listan)
    setPropio = set(hullPropio)
    setScipy = set(hullScipy)
    
    if setPropio == setScipy:
        print("\n✅ VERIFICACIÓN EXITOSA: Tu algoritmo 'Divide y Vencerás' ha encontrado")
        print("exactamente el mismo polígono convexo que la librería SciPy.")
    else:
        print("\n❌ Diferencia detectada entre los algoritmos.")


ejecutarRequerimiento3()


if __name__ == "__main__":
    # Mete aquí adentro las llamadas a tus pruebas:
    generarMatrizTiempos()
    ejecutarItinerario()
    probarAlgoritmosTSP()
    probarDatosReales()
    ejecutarExperimentoR2()
    ejecutarRequerimiento3()
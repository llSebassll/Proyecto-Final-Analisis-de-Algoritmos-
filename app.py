import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import osmnx as ox

# Importamos las funciones vitales desde tu main.py
from main import (
    cargarMatriz, heldKarp, heuristicaVoraz, 
    dijkstraGrafoReal, envolventeConvexa
)

# Configuración inicial de la página
st.set_page_config(page_title="Proyecto Quindío - Algoritmos", layout="wide")

# =============================================================================
# CACHÉ DE DATOS
# =============================================================================
@st.cache_data
def cargarDatosBase():
    matriz, listaLugares = cargarMatriz("procesado/matriz_tiempos.csv")
    dfLugares = pd.read_csv("procesado/lugares_nodos.csv")
    return matriz, listaLugares, dfLugares

@st.cache_resource
def cargarGrafo():
    return ox.load_graphml("procesado/quindio_scc.graphml")

matrizTiempos, listaNombres, dfLugares = cargarDatosBase()
grafoQuindio = cargarGrafo()

st.title("🗺️ Rutas del Paisaje Cultural Cafetero")
st.markdown("Aplicación demostrativa para la evaluación de algoritmos sobre grafos reales.")

# =============================================================================
# PESTAÑAS DE EVALUACIÓN (Según Rúbrica)
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Ruta Simple (Dijkstra)", 
    "2. Envolvente Convexa", 
    "3. Itinerarios TSP (Held-Karp vs Voraz)", 
    "4. Resultados y Gráficas"
])

# -----------------------------------------------------------------------------
# PESTAÑA 1: RUTA ORIGEN A DESTINO (DIJKSTRA)
# -----------------------------------------------------------------------------
with tab1:
    st.header("Punto a Punto")
    colA, colB = st.columns(2)
    origenRuta = colA.selectbox("Lugar de Origen:", dfLugares['nombre'].tolist(), index=0, key="orig_r1")
    destinoRuta = colB.selectbox("Lugar de Destino:", dfLugares['nombre'].tolist(), index=1, key="dest_r1")
    
    if st.button("Trazar Ruta", type="primary", key="btn_trazar"):
        st.session_state['mostrar_ruta'] = True
        
    if st.session_state.get('mostrar_ruta', False):
        with st.spinner("Calculando camino más corto con Dijkstra (Cola de Prioridad)..."):
            nodoO = dfLugares[dfLugares['nombre'] == origenRuta]['nodo'].values[0]
            nodoD = dfLugares[dfLugares['nombre'] == destinoRuta]['nodo'].values[0]
            
            tiempoSegundos, rutaNodos, nodosExplorados = dijkstraGrafoReal(grafoQuindio, nodoO, nodoD)
            
            st.success(f"**Tiempo de Viaje:** {tiempoSegundos / 60:.1f} minutos | **Nodos explorados:** {nodosExplorados}")
            
            mapaDijkstra = folium.Map(location=[4.53, -75.67], zoom_start=11, tiles="OpenStreetMap")
            
            coordenadasRuta = []
            for n in rutaNodos:
                lat = grafoQuindio.nodes[n]['y']
                lon = grafoQuindio.nodes[n]['x']
                coordenadasRuta.append((lat, lon))
            
            folium.PolyLine(coordenadasRuta, color="blue", weight=5, opacity=0.8).add_to(mapaDijkstra)
            if coordenadasRuta:
                folium.Marker(coordenadasRuta[0], popup="Origen", icon=folium.Icon(color="green")).add_to(mapaDijkstra)
                folium.Marker(coordenadasRuta[-1], popup="Destino", icon=folium.Icon(color="red")).add_to(mapaDijkstra)
            
            st_folium(mapaDijkstra, width=900, height=500, key="mapa_tab1")

# -----------------------------------------------------------------------------
# PESTAÑA 2: ENVOLVENTE CONVEXA (TODO EL DEPARTAMENTO)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Envolvente Convexa del Catálogo")
    st.write("Cálculo del polígono que encierra todos los puntos turísticos usando la técnica Divide y Vencerás (QuickHull).")
    
    if st.button("Generar Envolvente Convexa", key="btn_env"):
        st.session_state['mostrar_env'] = True
        
    if st.session_state.get('mostrar_env', False):
        puntosTotales = list(zip(dfLugares['lon'], dfLugares['lat']))
        puntosUnicos = list(set(puntosTotales))
        
        hull = envolventeConvexa(puntosUnicos)
        
        mapaEnvolvente = folium.Map(location=[4.53, -75.67], zoom_start=10, tiles="OpenStreetMap")
        
        for p in puntosUnicos:
            folium.CircleMarker([p[1], p[0]], radius=3, color="gray", fill=True).add_to(mapaEnvolvente)
        
        hullFolium = [[p[1], p[0]] for p in hull] 
        folium.Polygon(
            locations=hullFolium,
            color='purple',
            fill=True,
            fill_opacity=0.2,
            weight=3,
            tooltip="Envolvente Convexa (Divide y Vencerás)"
        ).add_to(mapaEnvolvente)
        
        st_folium(mapaEnvolvente, width=900, height=500, key="mapa_tab2")

# -----------------------------------------------------------------------------
# PESTAÑA 3: ITINERARIOS (HELD-KARP VS VORAZ) - VERSIÓN VISUAL COMPLETA
# -----------------------------------------------------------------------------
with tab3:
    st.header("Problema del Viajante (TSP) e Itinerario Interactivo")
    
    nodoInicio = st.selectbox("Lugar de partida:", dfLugares['nombre'].tolist(), index=dfLugares['nombre'].tolist().index("Armenia"), key="origen_tsp")
    
    dfAtractivos = dfLugares[dfLugares['categoria'] == 'Atractivo']
    opcionesAtractivos = [l for l in dfAtractivos['nombre'].tolist() if l != nodoInicio]
    
    destinosSeleccionados = st.multiselect(
        "Selecciona entre 6 y 8 atractivos para el itinerario:",
        opcionesAtractivos,
        default=opcionesAtractivos[:6],
        key="destinos_tsp"
    )
    
    # Recuperamos el selector de algoritmo de la versión 1
    algoritmoTSP = st.radio(
        "Algoritmo base para trazar el mapa de 11 horas:",
        ["Programación Dinámica (Óptimo)", "Heurística Voraz (Rápido)"],
        key="algo_tsp"
    )
    
    if st.button("Comparar Algoritmos y Generar Mapa", type="primary", key="btn_tsp"):
        st.session_state['mostrar_tsp'] = True
        
    if st.session_state.get('mostrar_tsp', False):
        if len(destinosSeleccionados) < 2:
            st.warning("Selecciona al menos 2 atractivos.")
        else:
            with st.spinner("Calculando rutas, ajustando tiempo a 11h y dibujando el mapa..."):
                
                # --- 1. COMPARATIVA SEGÚN RÚBRICA ---
                nodosActuales = list(destinosSeleccionados)
                
                tiempoHkSegundos, rutaHk = heldKarp(matrizTiempos, nodosActuales, nodoInicio)
                tiempoVorazSegundos, rutaVoraz = heuristicaVoraz(matrizTiempos, nodosActuales, nodoInicio)
                
                tiempoHkMinutos = tiempoHkSegundos / 60
                tiempoVorazMinutos = tiempoVorazSegundos / 60
                diferencia = tiempoVorazMinutos - tiempoHkMinutos
                errorPorcentual = (diferencia / tiempoHkMinutos) * 100 if tiempoHkMinutos > 0 else 0
                
                st.subheader("1. Comparación de Algoritmos (Solo Viaje)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Held-Karp (Óptimo)", f"{tiempoHkMinutos:.1f} min")
                c2.metric("Heurística Voraz", f"{tiempoVorazMinutos:.1f} min")
                c3.metric("Penalización Voraz", f"+{diferencia:.1f} min", f"+{errorPorcentual:.1f}% error", delta_color="inverse")
                
                # --- 2. AJUSTE DE 11 HORAS (660 min) ---
                st.subheader("2. Itinerario Ajustado y Mapa del Recorrido (11 Horas)")
                
                duraciones = dict(zip(dfLugares['nombre'], dfLugares['duracionVisita'].fillna(0)))
                lugaresEliminados = []
                
                while True:
                    if algoritmoTSP == "Programación Dinámica (Óptimo)":
                        tiempoViaje, rutaOptima = heldKarp(matrizTiempos, nodosActuales, nodoInicio)
                    else:
                        tiempoViaje, rutaOptima = heuristicaVoraz(matrizTiempos, nodosActuales, nodoInicio)
                        
                    tiempoVisitas = sum(duraciones.get(nodo, 0) for nodo in nodosActuales)
                    tiempoTotalJornada = (tiempoViaje / 60) + tiempoVisitas
                    
                    if tiempoTotalJornada <= 660:
                        break
                    else:
                        atractivoAEliminar = max(nodosActuales, key=lambda x: duraciones.get(x, 0))
                        lugaresEliminados.append(atractivoAEliminar)
                        nodosActuales.remove(atractivoAEliminar)
                        if not nodosActuales: break
                
                if lugaresEliminados:
                    st.warning(f"⚠️ Para cumplir con la jornada de 11h, se eliminaron automáticamente los atractivos con las visitas más largas: **{', '.join(lugaresEliminados)}**")
                
                if not nodosActuales:
                    st.error("No es posible visitar ningún atractivo y volver en menos de 11 horas.")
                else:
                    st.success(f"**Ruta Final:** {' ➔ '.join(rutaOptima)}")
                    st.write(f"**Desglose:** {tiempoViaje/60:.1f} min viajando + {tiempoVisitas:.1f} min en visitas = **{tiempoTotalJornada:.1f} min totales**")
                    
                    # --- 3. DIBUJAR MAPA INTERACTIVO (DIJKSTRA Y ENVOLVENTE) ---
                    mapaTsp = folium.Map(location=[4.53, -75.67], zoom_start=11, tiles="OpenStreetMap")
                    puntosParaEnvolvente = []
                    
                    for i in range(len(rutaOptima) - 1):
                        origenNombre = rutaOptima[i]
                        destinoNombre = rutaOptima[i+1]
                        
                        nodoO = dfLugares[dfLugares['nombre'] == origenNombre]['nodo'].values[0]
                        nodoD = dfLugares[dfLugares['nombre'] == destinoNombre]['nodo'].values[0]
                        
                        latO = dfLugares[dfLugares['nombre'] == origenNombre]['lat'].values[0]
                        lonO = dfLugares[dfLugares['nombre'] == origenNombre]['lon'].values[0]
                        puntosParaEnvolvente.append((lonO, latO))
                        
                        color = "green" if i == 0 else "blue"
                        folium.Marker(
                            [latO, lonO], 
                            popup=origenNombre,
                            icon=folium.Icon(color=color, icon="info-sign")
                        ).add_to(mapaTsp)
                        
                        _, rutaNodos, _ = dijkstraGrafoReal(grafoQuindio, nodoO, nodoD)
                        
                        coordenadasRuta = []
                        for n in rutaNodos:
                            lat = grafoQuindio.nodes[n]['y']
                            lon = grafoQuindio.nodes[n]['x']
                            coordenadasRuta.append((lat, lon))
                            
                        folium.PolyLine(coordenadasRuta, color="red", weight=4, opacity=0.7).add_to(mapaTsp)
                        
                    # Marcador final
                    latFin = dfLugares[dfLugares['nombre'] == rutaOptima[-1]]['lat'].values[0]
                    lonFin = dfLugares[dfLugares['nombre'] == rutaOptima[-1]]['lon'].values[0]
                    folium.Marker([latFin, lonFin], popup="Llegada", icon=folium.Icon(color="red", icon="flag")).add_to(mapaTsp)
                    puntosParaEnvolvente.append((lonFin, latFin))
                    
                    # Envolvente Convexa específica para esta ruta
                    puntosUnicos = list(set(puntosParaEnvolvente))
                    if len(puntosUnicos) > 2:
                        hull = envolventeConvexa(puntosUnicos)
                        hullFolium = [[p[1], p[0]] for p in hull] 
                        folium.Polygon(
                            locations=hullFolium,
                            color='purple',
                            fill=True,
                            fill_opacity=0.1,
                            weight=2,
                            dash_array='5, 5',
                            tooltip="Área del Recorrido (Convex Hull)"
                        ).add_to(mapaTsp)

                    st_folium(mapaTsp, width=900, height=600, key="mapa_tab3")

# -----------------------------------------------------------------------------
# PESTAÑA 4: RESULTADOS DOCUMENTALES
# -----------------------------------------------------------------------------
with tab4:
    st.header("Resultados y Gráficas del Documento Técnico")
    st.write("Visualización de las métricas de rendimiento recolectadas en los requerimientos.")
    
    st.subheader("Rendimiento Dijkstra (Req 2)")
    try:
        st.image("procesado/grafica_dijkstra.png", caption="Búsqueda Lineal O(V²) vs Cola de Prioridad O((V+E) log V)")
    except:
        st.info("Ejecuta primero el Requerimiento 2 desde la consola para generar la gráfica (grafica_dijkstra.png).")
        
    st.subheader("Análisis de Memoria (Req 1)")
    st.markdown("""
    * **Lista de Adyacencia:** Más eficiente en este proyecto porque el grafo de carreteras es disperso (pocos vecinos por intersección).
    * **Matriz de Adyacencia:** Descartada por consumir memoria excesiva guardando conexiones inexistentes.
    """)
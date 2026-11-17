# Especificación: Requerimiento 5 - Interfaz Interactiva

**Objetivo:** 
Proveer una aplicación demostrativa para visualizar los resultados de la lógica de grafos.

**Comportamiento Esperado:**
1. Desarrollar una interfaz con Streamlit y mapas de Folium usando tiles de OpenStreetMap.
2. Pestaña 1: Selector de Origen/Destino para trazar la ruta de Dijkstra en rojo.
3. Pestaña 2: Botón para generar y pintar la Envolvente Convexa sobre el Quindío.
4. Pestaña 3: Selector múltiple (6 a 8 atractivos). Comparar Held-Karp vs Voraz, aplicar el filtro automático de 11 horas y dibujar el mapa del recorrido final.
5. El sistema debe usar caché (`@st.cache_data`) y persistencia de memoria (`st.session_state`) para evitar recargas del grafo base.

**Casos de Prueba (Interacción UI):**
- Seleccionar 8 atractivos de larga duración; comprobar que el sistema muestra la advertencia de eliminación y grafica solo los lugares que caben en 11h.
- Mover el zoom del mapa; comprobar que la interfaz no desaparece ni reinicia su estado al interactuar.
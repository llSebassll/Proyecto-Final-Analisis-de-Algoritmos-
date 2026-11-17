# Especificación: Requerimiento 1 - Grafo y Conectividad

**Objetivo:** 
Cargar la red vial del Quindío usando OSMnx, imputar pesos de viaje (segundos) y garantizar que todos los nodos sean alcanzables.

**Comportamiento Esperado:**
1. El sistema debe descargar el grafo real (drive) y calcular `travel_time` basándose en la distancia y la velocidad `maxspeed`.
2. Se debe extraer la Componente Fuertemente Conexa (SCC) principal utilizando el algoritmo de Tarjan para evitar nodos "muertos" o vías de un solo sentido sin salida.
3. Se deben asociar las coordenadas geográficas de los 40 lugares turísticos del archivo CSV al nodo válido más cercano del grafo usando proyección UTM.

**Casos de Prueba (Prueba de Consola):**
- Imprimir la diferencia de cantidad de nodos y aristas (V, E) antes y después de aplicar Tarjan.
- Imprimir una alerta para los lugares turísticos que queden a más de 500 metros de su nodo asignado en la SCC.
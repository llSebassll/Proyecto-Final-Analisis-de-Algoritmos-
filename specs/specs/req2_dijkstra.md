# Especificación: Requerimiento 2 - Búsqueda de Ruta Óptima

**Objetivo:** 
Encontrar el camino más corto (en tiempo) entre cualquier par de nodos de la red vial.

**Comportamiento Esperado:**
1. Implementar el Algoritmo de Dijkstra desde cero.
2. Construir e integrar una clase propia `ColaPrioridad` (Min-Heap) para garantizar una complejidad de O((V+E) log V).
3. Implementar una versión secundaria de Dijkstra basada en Búsqueda Lineal O(V^2) exclusivamente con fines comparativos.

**Casos de Prueba (`ejecutarExperimentoR2`):**
- Seleccionar 20 pares de nodos aleatorios (10 del mismo municipio y 10 cruzados).
- Ejecutar ambas versiones de Dijkstra sobre los pares.
- Verificar que ambas devuelven exactamente la misma distancia.
- Generar una gráfica de dispersión (Nodos explorados vs. Tiempo en ms) que demuestre empíricamente la superioridad de la Cola de Prioridad.
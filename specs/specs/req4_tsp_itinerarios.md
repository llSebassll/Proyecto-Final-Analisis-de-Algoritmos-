# Especificación: Requerimiento 4 - Problema del Viajante y Jornada

**Objetivo:** 
Calcular la ruta óptima para visitar un subconjunto de atractivos turísticos, partiendo y regresando a un origen (Armenia), cumpliendo la restricción de tiempo de una jornada.

**Comportamiento Esperado:**
1. Implementar tres aproximaciones al TSP: Fuerza Bruta O(k!), Programación Dinámica (Held-Karp) O(k^2 * 2^k) y Heurística Voraz O(k^2).
2. El sistema debe sumar el tiempo de viaje entre nodos (matriz de distancias) y el tiempo estático (visita a cada lugar).
3. Si el tiempo total excede los 660 minutos (11 horas), el sistema debe eliminar automáticamente el atractivo con la visita más larga iterativamente hasta que el itinerario sea válido.

**Casos de Prueba (`pruebasTSP` y App Web):**
- Medir y comparar los tiempos de ejecución de Fuerza Bruta vs Held-Karp vs Voraz para k=5.
- Verificar que Held-Karp y Fuerza Bruta arrojan el mismo tiempo total.
- Imprimir la justificación por consola cada vez que el itinerario requiera descartar un lugar para encajar en las 11 horas.
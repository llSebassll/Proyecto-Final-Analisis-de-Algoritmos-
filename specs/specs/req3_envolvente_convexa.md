# Especificación: Requerimiento 3 - Envolvente Convexa

**Objetivo:** 
Calcular el polígono exterior que encierra todos los lugares del catálogo turístico.

**Comportamiento Esperado:**
1. Implementar el algoritmo QuickHull aplicando estrictamente el paradigma de "Divide y Vencerás".
2. Partir el conjunto de coordenadas usando los extremos en el eje X y podar geométricamente los puntos interiores usando el determinante vectorial (producto cruz).
3. Lograr una complejidad promedio esperada de O(n log n).

**Casos de Prueba (`ejecutarRequerimiento3`):**
- Ejecutar el algoritmo propio sobre las coordenadas de los lugares.
- Ejecutar en paralelo la función `scipy.spatial.ConvexHull`.
- Verificar mediante aserción de conjuntos que los vértices frontera retornados por el algoritmo propio son exactamente los mismos que los de la librería profesional.
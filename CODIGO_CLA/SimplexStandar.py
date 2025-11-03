"""
    Problema:
      max c^T x
      s.a. A x ≤ b, x ≥ 0
    Método:
      Agrega holguras y usa tableau de Simplex para maximización.
    """
  

# Resolver 
# máx Z = 3x1 + 2x2 
# s.a. 
#     x1 + 2x2 ≤ 8 
#     4x1 + x2 ≤ 12 
#            x ≥ 0


import math
import numpy as np

# Simplex Standar - maximización con restricciones ≤ y variables ≥ 0
def simplex_max_leq(A, b, c):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    m, n = A.shape

    # Construir tableau con holguras
    # Variables: [x (n), s (m)]
    T = np.zeros((m + 1, n + m + 1), dtype=float)
    # Restricciones
    T[:m, :n] = A
    T[:m, n:n+m] = np.eye(m)
    T[:m, -1] = b
    # Fila objetivo (forma: Z - c^T x = 0) => colocamos -c en el tableau
    T[-1, :n] = -c

    basis = list(range(n, n + m))  # índices de holguras como base inicial

    def pivot(row, col):
        # Normalizar fila pivote
        T[row, :] = T[row, :] / T[row, col]
        # Hacer ceros en la columna pivote
        for r in range(T.shape[0]):
            if r != row:
                T[r, :] -= T[r, col] * T[row, :]

    # repetir hasta optimalidad
    while True:
        # variable entrante: el coeficiente más negativo en la fila Z (si hay)
        cost_row = T[-1, :-1]
        col = None
        min_val = 0.0
        for j, v in enumerate(cost_row):
            if v < min_val - 1e-12:
                min_val = v
                col = j
        if col is None:
            # optimo alcanzado
            break

        # ratio test para variable saliente
        row = None
        best_ratio = math.inf
        for i in range(m):
            if T[i, col] > 1e-12:
                ratio = T[i, -1] / T[i, col]
                if ratio < best_ratio - 1e-12:
                    best_ratio = ratio
                    row = i
        if row is None:
            raise RuntimeError("Problema no acotado en Simplex.")

        # pivotear
        pivot(row, col)
        basis[row] = col

    # extraer solución
    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i, -1]
    z = T[-1, -1]

    # precios sombra (multiplicadores duales) = coeficientes de las restricciones en la fila Z respecto a holguras
    y = T[-1, n:n+m].copy()

    # costos reducidos: coeficientes finales de la fila Z (solo para x)
    reduced_costs = T[-1, :n].copy()

    return {
        "x": x,
        "z": z,
        "y": y,
        "reduced_costs": reduced_costs,
        "tableau": T,
        "basis": basis,
    }

# datos del LP
A = [[1, 2],
     [4, 1]]
b = [8, 12]
c = [3, 2]

res_custom = simplex_max_leq(A, b, c)

print("=== Resultados con Simplex Standar ===")
print(f"x* = {res_custom['x']}")
print(f"Z* = {res_custom['z']}")
print(f"Precios sombra (y) = {res_custom['y']}")
print(f"Costos reducidos (c_j - z_j) en convención de fila Z = {res_custom['reduced_costs']}")
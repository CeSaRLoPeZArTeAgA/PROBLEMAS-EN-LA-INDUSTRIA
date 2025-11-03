# simplex_dos_fases_verbose.py
# Implementación del Método Símplex de DOS FASES (tableau) con impresión paso a paso.
# Problema de prueba:
#   Max Z = 3 x1 + 2 x2
#   s.a.
#     x1 + x2           = 4
#     x1 + 2 x2      >= 6
#     x1, x2 >= 0
# Forma estándar:
#   x1 + x2           + a1         = 4
#   x1 + 2 x2 - s2    + a2         = 6
# Fase I: min W = a1 + a2
# Fase II: max Z = 3 x1 + 2 x2

from typing import List, Tuple
import numpy as np
import math

def print_tableau(T: np.ndarray, var_names: List[str], base_vars: List[str], title: str = ""):
    rows, cols = T.shape
    col_w = 11
    head = ["Base"] + var_names + ["RHS"]
    if title:
        print(f"\n=== {title} ===")
    print(" | ".join(h.rjust(col_w) for h in head))
    print("-" * (len(head)*(col_w+3)))
    for i in range(rows-1):
        row = [base_vars[i]] + [f"{T[i,j]:.6g}" for j in range(cols-1)] + [f"{T[i,-1]:.6g}"]
        print(" | ".join(s.rjust(col_w) for s in row))
    print("-" * (len(head)*(col_w+3)))
    row = ["Obj"] + [f"{T[-1,j]:.6g}" for j in range(cols-1)] + [f"{T[-1,-1]:.6g}"]
    print(" | ".join(s.rjust(col_w) for s in row))

def pivot(T: np.ndarray, i: int, j: int):
    """Pivot Gauss-Jordan en (i,j)."""
    piv = T[i, j]
    T[i, :] /= piv
    for r in range(T.shape[0]):
        if r != i:
            T[r, :] -= T[r, j]*T[i, :]

def ratio_test(T: np.ndarray, col: int, m: int, tol: float = 1e-12) -> Tuple[int, float]:
    """Devuelve (fila_salida, ratio_min). Si no hay, (None, inf)."""
    best = math.inf
    row = None
    for i in range(m):
        a = T[i, col]
        if a > tol:
            r = T[i, -1] / a
            if r < best - 1e-12:
                best = r
                row = i
    return row, best

def select_entering_for_min(cost_row: np.ndarray, eligible_cols: List[int], T: np.ndarray, m: int, tol: float = 1e-10):
    """Minimización: selecciona una columna con coeficiente > 0 y viable (existe ratio)."""
    for j in eligible_cols:
        if cost_row[j] > tol:
            # ver si hay algún positivo en la columna (ratio viable)
            for i in range(m):
                if T[i, j] > tol:
                    return j
    return None

def select_entering_for_max(cost_row: np.ndarray, eligible_cols: List[int], T: np.ndarray, m: int, tol: float = 1e-10):
    """Maximización: selecciona una columna con coeficiente < 0 y viable (existe ratio)."""
    for j in eligible_cols:
        if cost_row[j] < -tol:
            for i in range(m):
                if T[i, j] > tol:
                    return j
    return None

def two_phase_simplex_verbose():
    # Definición del problema de prueba
    # Variables: x1, x2, s2, a1, a2
    var_names = ["x1", "x2", "s2", "a1", "a2"]
    idx = {name: k for k, name in enumerate(var_names)}
    m = 2
    n = len(var_names)

    # Restricciones (matriz y RHS)
    A = np.array([
        [1, 1,  0, 1, 0],   # R1: x1 + x2 + a1 = 4
        [1, 2, -1, 0, 1],   # R2: x1 + 2x2 - s2 + a2 = 6
    ], dtype=float)
    b = np.array([4.0, 6.0], dtype=float)

    # ========= FASE I: Minimizar W = a1 + a2
    # Tableau T1: filas = m + 1 (restricciones + fila W), columnas = n + 1 (variables + RHS)
    T1 = np.zeros((m+1, n+1), dtype=float)
    T1[:m, :n] = A
    T1[:m, -1] = b

    # Base inicial: a1, a2
    basis = [idx["a1"], idx["a2"]]
    base_vars = ["a1", "a2"]

    # Fila de W (minimización): W = a1 + a2.
    # Convención de tableau: última fila almacena W - (a1 + a2) = 0, es decir:
    # coef en a1,a2 = -1, RHS = 0. Luego, para poner W de forma canónica, eliminamos columnas de a1,a2
    # sumando las filas básicas correspondientes.
    T1[-1, idx["a1"]] = -1.0
    T1[-1, idx["a2"]] = -1.0
    # "Eliminación" de a1 y a2 en fila W (sumando filas donde son básicas=1):
    T1[-1, :] += T1[0, :]
    T1[-1, :] += T1[1, :]
    # Nota: con esta convención, el valor de W en RHS queda b1 + b2 (=10).
    # Pero ojo: como la ecuación es W - sum(...) = 0 y hemos hecho W := W + R1 + R2,
    # el número en RHS de la fila W ahora representa W (10). Para minimizar W, buscamos
    # que los coeficientes de x sean >0 para entrar y reducir W.

    print_tableau(T1, var_names, base_vars, "Fase I - Tableau inicial (min W)")

    step = 0
    # En Fase I, permitimos entrar solo variables de decisión (x1,x2). No s2 ni artificiales.
    eligible_phase1 = [idx["x1"], idx["x2"]]

    while True:
        step += 1
        entering = select_entering_for_min(T1[-1, :n], eligible_phase1, T1, m)
        if entering is None:
            print("\n[Fase I] Óptimo de W alcanzado (no hay coeficientes positivos viables en fila W).")
            break
        leaving, _ = ratio_test(T1, entering, m)
        if leaving is None:
            raise RuntimeError("[Fase I] Problema no acotado (no hay ratio).")

        print(f"\n[Fase I] Paso {step}: ENTRA '{var_names[entering]}', SALE '{var_names[basis[leaving]]}' (fila {leaving}).")
        pivot(T1, leaving, entering)
        basis[leaving] = entering
        base_vars[leaving] = var_names[entering]
        print_tableau(T1, var_names, base_vars, f"Fase I - Después del pivote {step}")

    W_star = T1[-1, -1]
    print(f"\n[Fase I] W* = {W_star:.6g}")
    if W_star > 1e-9:
        print("[Fase I] INVIABLE (W* > 0).")
        return

    # ========= FASE II: Restaurar Z = 3 x1 + 2 x2 y maximizar
    T2 = np.zeros_like(T1)
    T2[:m, :n] = T1[:m, :n]
    T2[:m, -1] = T1[:m, -1]
    # Fila Z en forma Z - 3 x1 - 2 x2 = 0
    c = np.zeros(n)
    c[idx["x1"]] = 3.0
    c[idx["x2"]] = 2.0
    T2[-1, :n] = -c
    T2[-1, -1] = 0.0
    # Eliminar contribución de las básicas en fila Z:
    for i in range(m):
        bj = basis[i]
        coef = T2[-1, bj]
        if abs(coef) > 1e-12:
            T2[-1, :] -= coef * T2[i, :]

    print_tableau(T2, var_names, base_vars, "Fase II - Tableau inicial (max Z)")

    step2 = 0
    eligible_phase2 = [idx["x1"], idx["x2"], idx["s2"]]
    while True:
        step2 += 1
        entering = select_entering_for_max(T2[-1, :n], eligible_phase2, T2, m)
        if entering is None:
            print("\n[Fase II] Óptimo alcanzado (no hay coeficientes negativos viables en fila Z).")
            break
        leaving, _ = ratio_test(T2, entering, m)
        if leaving is None:
            raise RuntimeError("[Fase II] No acotado.")
        print(f"\n[Fase II] Paso {step2}: ENTRA '{var_names[entering]}', SALE '{var_names[basis[leaving]]}' (fila {leaving}).")
        pivot(T2, leaving, entering)
        basis[leaving] = entering
        base_vars[leaving] = var_names[entering]
        print_tableau(T2, var_names, base_vars, f"Fase II - Después del pivote {step2}")

    # Solución final
    x_all = np.zeros(n)
    for i in range(m):
        x_all[basis[i]] = T2[i, -1]
    Z_star = T2[-1, -1]

    print("\n=== SOLUCIÓN FINAL ===")
    print(f"x1 = {x_all[idx['x1']]:.6g}, x2 = {x_all[idx['x2']]:.6g}")
    print(f"Z* = {Z_star:.6g}")

if __name__ == "__main__":
    two_phase_simplex_verbose()

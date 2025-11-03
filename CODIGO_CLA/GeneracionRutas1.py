# Código para resolver por enumeración completa el ejemplo pequeño del usuario.
# Este código busca todas las asignaciones binarias x_{i,S} y encuentra la de costo mínimo
# que cumple las restricciones:
# 1) cada vehículo i asignado a lo sumo a un subconjunto,
# 2) la carga total asignada a cada vehículo no excede su capacidad,
# 3) cada cliente k es atendido exactamente una vez.
#
# No requiere paquetes externos (solo itertools y math).

from itertools import product, combinations

# Datos del ejemplo
I = [1, 2]  # vehículos
J = ['A', 'B']  # clientes
s = {1: 10, 2: 8}  # capacidades
d = {'A': 4, 'B': 5}  # demandas por cliente

# Costos r_S para cada subconjunto no vacío de J
r = {
    ('A',): 2,
    ('B',): 3,
    ('A','B'): 6
}

# Generar lista ordenada de subconjuntos (tuplas) para iterar
P_J = list(r.keys())  # [('A',), ('B',), ('A','B')]

# Crear índices para variables x_{i,S}
vars_idx = [(i, S) for i in I for S in P_J]

def is_feasible_assignment(assign):
    # assign: dict {(i,S): 0/1}
    # 1) Cada vehículo i a lo sumo un subconjunto
    for i in I:
        if sum(assign[(i,S)] for S in P_J) > 1:
            return False
    # 2) Capacidad de cada vehículo no excedida
    for i in I:
        total_load = 0
        for S in P_J:
            if assign[(i,S)] == 1:
                total_load += sum(d[c] for c in S)
        if total_load > s[i]:
            return False
    # 3) Cada cliente atendido exactamente una vez
    for k in J:
        served = 0
        for i in I:
            for S in P_J:
                if k in S:
                    served += assign[(i,S)]
        if served != 1:
            return False
    return True

best_cost = float('inf')
best_assignments = []

# Enumerar todas las combinaciones binarias para las variables (2^(|I|*|P_J|))
nvars = len(vars_idx)
for bits in product([0,1], repeat=nvars):
    assign = {vars_idx[i]: bits[i] for i in range(nvars)}
    if is_feasible_assignment(assign):
        cost = sum(assign[(i,S)] * r[S] for (i,S) in vars_idx)
        if cost < best_cost:
            best_cost = cost
            best_assignments = [assign]
        elif cost == best_cost:
            best_assignments.append(assign)

# Mostrar resultados
print("Número de variables:", nvars)
print("Cantidad de soluciones factibles encontradas:", len(best_assignments))
print("Costo mínimo:", best_cost)
print()
for idx, sol in enumerate(best_assignments, 1):
    print(f"Solución óptima {idx}:")
    for (i,S), val in sorted(sol.items()):
        if val == 1:
            print(f"  Vehículo {i} atiende subconjunto {S} con costo r_{S} = {r[S]}")
    print()

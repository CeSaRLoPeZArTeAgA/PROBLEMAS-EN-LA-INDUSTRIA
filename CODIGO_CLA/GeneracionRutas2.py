# Heurística pura en Python (sin dependencias externas) para el mismo ejemplo m=3, n=5.
# Genera subconjuntos factibles, luego asigna iterativamente el subconjunto con mejor "ratio":
# ratio = costo_r_S / (#clientes_del_subconjunto_no_atendidos_por_ahora)
# Respeta que cada vehículo use a lo sumo un subconjunto y que cada cliente sea atendido una vez.
from itertools import combinations

# Datos del ejemplo
I = [1, 2, 3]  # vehículos
J = ['A', 'B', 'C', 'D', 'E']  # clientes
s = {1: 10, 2: 8, 3: 7}  # capacidades por vehículo
d = {'A': 4, 'B': 5, 'C': 3, 'D': 6, 'E': 2}  # demandas por cliente

# Generar todos los subconjuntos no vacíos de J
all_subsets = []
for rlen in range(1, len(J)+1):
    for comb in combinations(J, rlen):
        all_subsets.append(tuple(comb))

# Calcular demanda total D_S y costo r_S para cada subconjunto S
D = {S: sum(d[c] for c in S) for S in all_subsets}
r = {S: 2.0 + 0.5 * len(S) + 0.15 * D[S] for S in all_subsets}

# Filtrar subconjuntos factibles por vehículo (D_S <= s_i)
feasible_subsets = {i: [S for S in all_subsets if D[S] <= s[i]] for i in I}

# Heurística greedy por ratio
assigned = []  # lista de (i,S)
used_vehicle = set()
covered_clients = set()

# Repetir hasta cubrir todos los clientes o no poder asignar más
while len(covered_clients) < len(J):
    candidates = []
    for i in I:
        if i in used_vehicle:
            continue
        for S in feasible_subsets[i]:
            new_clients = [c for c in S if c not in covered_clients]
            if not new_clients:
                continue
            ratio = r[S] / len(new_clients)
            candidates.append((ratio, r[S], i, S, len(new_clients)))
    if not candidates:
        break
    # elegir el candidato con menor ratio (mejor costo por cliente nuevo)
    candidates.sort(key=lambda x: (x[0], x[1]))
    ratio, costS, veh, S, newc = candidates[0]
    assigned.append((veh, S, costS, D[S]))
    used_vehicle.add(veh)
    for c in S:
        covered_clients.add(c)

# Verificar factibilidad final (cada cliente atendido exactamente una vez)
feasible = len(covered_clients) == len(J)

# Mostrar resultados
print("Heurística greedy - Ejemplo más grande (m=3, n=5)")
print("Vehículos:", I)
print("Clientes:", J)
print("\nSubconjuntos factibles por vehículo (resumen):")
for i in I:
    print(f"  Vehículo {i}: {len(feasible_subsets[i])} subconjuntos factibles")

print("\nAsignaciones heurísticas:")
total_cost = 0.0
for veh, S, costS, DS in assigned:
    print(f"  Vehículo {veh} -> {S}, costo r_S={costS:.2f}, demanda D_S={DS}")
    total_cost += costS

print(f"\nCosto total heurístico: {total_cost:.2f}")
print("Clientes cubiertos:", sorted(list(covered_clients)))
print("Solución factible completa?:", feasible)

# Si la heurística no cubre todos los clientes, mostrar cuáles faltan
if not feasible:
    faltantes = [c for c in J if c not in covered_clients]
    print("Clientes no cubiertos:", faltantes)

# También muestro la lista de candidatos ordenados por costo por nuevo cliente para inspección rápida
print("\nPrimeros 8 candidatos (ratio, costo, veh, subconjunto, nuevos_clientes):")
cands = []
for i in I:
    if i in used_vehicle:
        continue
    for S in feasible_subsets[i]:
        new_clients = [c for c in S if c not in covered_clients]
        if not new_clients:
            continue
        ratio = r[S] / len(new_clients)
        cands.append((ratio, r[S], i, S, len(new_clients)))
cands.sort(key=lambda x: (x[0], x[1]))
for cand in cands[:8]:
    print(" ", cand)

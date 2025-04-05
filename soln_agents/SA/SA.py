
import numpy as np
import time
import random

# --- Parser for .tsp file ---
def parse_tsp_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords = []
    start = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            start = True
            continue
        if "EOF" in line or line.strip() == "":
            break
        if start:
            parts = line.strip().split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

    coords = np.array(coords)
    n = len(coords)
    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                adj_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    return adj_matrix, coords

# --- Cost computation ---
def total_distance(path, adj_matrix):
    dist = 0.0
    for i in range(len(path)):
        dist += adj_matrix[path[i]][path[(i + 1) % len(path)]]
    return dist

def simulated_annealing(adj_matrix, initial_temp=1000, final_temp=1, alpha=1, max_iter=1000000, timeout=600):
    n = len(adj_matrix)
    current_path = list(range(n))
    np.random.shuffle(current_path)
    current_cost = total_distance(current_path, adj_matrix)
    best_path = list(current_path)
    best_cost = current_cost

    start_time = time.time()
    T = initial_temp
    iteration = 0

    while T > final_temp and iteration < max_iter:
        if time.time() - start_time > timeout:
            break

        i, j = random.sample(range(n), 2)
        new_path = list(current_path)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        new_cost = total_distance(new_path, adj_matrix)

        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_path = new_path
            current_cost = new_cost
            if new_cost < best_cost:
                best_path = new_path
                best_cost = new_cost

        T -= alpha
        iteration += 1

    return best_cost, time.time() - start_time, best_path

def run_tsp_experiment_from_file(filepath, runs=5, timeout=600):
    adj_matrix, coords = parse_tsp_file(filepath)

    times = []
    best_costs = []

    for run in range(runs):
        print(f"Run {run+1}...")
        best_cost, elapsed, _ = simulated_annealing(adj_matrix, timeout=timeout)
        times.append(elapsed)
        best_costs.append(best_cost)
        print(f"  Time taken: {elapsed:.2f}s | Best cost: {best_cost:.4f}")

    avg_time = np.mean(times)
    avg_cost = np.mean(best_costs)

    print(f"\nAverage time over {runs} runs: {avg_time:.2f} seconds")
    print(f"Average best cost: {avg_cost:.4f}")
    return times, best_costs

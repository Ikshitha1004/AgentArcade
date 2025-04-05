import os
import numpy as np
import time
import matplotlib.pyplot as plt
from SA.SA import SimulatedAnnealing

def parse_tsp_file(filepath):
    """Parses .tsp file and returns adjacency matrix and coordinates"""
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
            else:
                adj_matrix[i][j] = float('inf')

    return adj_matrix, coords

def plot_times(all_times):
    avg_time = np.mean(all_times)
    runs = np.arange(1, len(all_times) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(runs, all_times, marker='o', linestyle='-', color='blue', label='Time per Run')
    plt.axhline(y=avg_time, color='red', linestyle='--', label=f'Average Time = {avg_time:.2f}s')

    for x, y in zip(runs, all_times):
        plt.text(x, y + 0.05, f"{y:.2f}s", ha='center', fontsize=8)

    plt.title("Execution Time per Run (Simulated Annealing)")
    plt.xlabel("Run Number")
    plt.ylabel("Time (seconds)")
    plt.xticks(runs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("sa_run_times.png")
    plt.close()


def run_sa_experiment(filepath, runs=5, timeout=60):
    adj_matrix, coords = parse_tsp_file(filepath)
    n = len(adj_matrix)

    all_costs = []
    all_times = []
    best_overall_cost = float('inf')
    best_overall_path = None
    best_run_index = -1

    print(f"Running Simulated Annealing on {filepath} for {runs} runs...\n")

    for i in range(runs):
        print(f"Run {i+1}...")
        sa = SimulatedAnnealing(n, adj_matrix, coords)
        path, cost, time_taken = sa.solve(timeout=timeout, save_gif=(i == 0))  # GIF only for 1st run

        print(f"  Cost: {cost:.2f}, Time: {time_taken:.2f} seconds\n")
        all_costs.append(cost)
        all_times.append(time_taken)

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_path = path
            best_run_index = i

    # Save best run GIF if not already saved
    if best_run_index != 0:
        print(f"Generating GIF for best run (Run {best_run_index+1})...")
        sa = SimulatedAnnealing(n, adj_matrix, coords)
        sa.solve(timeout=timeout, save_gif=True)

    print(f"--- Summary ---")
    print(f"Average Cost: {np.mean(all_costs):.2f}")
    print(f"Average Time: {np.mean(all_times):.2f} seconds")
    print(f"Best Cost: {best_overall_cost:.2f} (Run {best_run_index+1})")

    # Plot and save time graph
    plot_times(all_times)
    print("Execution time graph saved as 'sa_run_times.png'.")

if __name__ == "__main__":
    tsp_file = "ch130.tsp"  # Change path as needed
    run_sa_experiment(tsp_file, runs=5, timeout=60)

import argparse
import time
import matplotlib.pyplot as plt
from SA.SA import SimulatedAnnealing
from HC.HC import HillClimbing
from tsp_env import TravellingSalesmanEnv


def plot_times(times, title, filename, label_color="blue"):
    avg_time = sum(times) / len(times)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(times) + 1), times, marker='o', linestyle='-', color=label_color, label='Run Time')
    plt.axhline(avg_time, color='red', linestyle='--', label=f'Avg Time: {avg_time:.6f}s')
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Execution time plot saved as '{filename}'.")
    print(f"Average Execution Time: {avg_time:.6f}s\n")


def run_sa(filepath):
    env = TravellingSalesmanEnv(filepath)
    adj_matrix = env.dist_matrix
    coords = env.points
    n = env.nPoints

    all_costs = []
    all_times = []
    best_cost = float('inf')
    best_path = None
    best_run = -1

    print(f" Simulated Annealing on {filepath}...\n")
    for i in range(5):
        print(f"Run {i+1}...")
        sa = SimulatedAnnealing(n, adj_matrix, coords)
        path, cost, t = sa.solve(timeout=60, save_gif=False)
        print(f"  Cost: {cost:.2f}, Time: {t:.2f}s\n")

        all_costs.append(cost)
        all_times.append(t)

        if cost < best_cost:
            best_cost = cost
            best_path = path
            best_run = i

    print(f"\nGenerating GIF for best run (Run {best_run+1})...")
    sa = SimulatedAnnealing(n, adj_matrix, coords)
    sa.solve(timeout=60, save_gif=True)

    print(f"Best Cost: {best_cost:.2f}")
    plot_times(all_times, "Simulated Annealing Execution Time per Run", "sa_run_times.png", label_color="blue")


def run_hc(filepath):
    env = TravellingSalesmanEnv(filepath)
    dist_matrix = env.dist_matrix
    coords = env.points
    n = env.nPoints

    best_cost = float('inf')
    best_run = -1
    times = []

    print(f" Hill Climbing on {filepath}...\n")
    for i in range(5):
        print(f"Run {i+1}...")
        hc = HillClimbing(n, dist_matrix, coords)
        _, cost, t = hc.solve(
            save_gif=False,
            gif_name=f"hill_climbing_run{i+1}.gif" if i == 0 else None
        )
        print(f"  Cost: {cost:.2f}, Time: {t:.2f}s\n")

        times.append(t)
        if cost < best_cost:
            best_cost = cost
            best_run = i

    print(f"Best Cost: {best_cost:.2f}")
    print(f"GIF saved for first run as 'hill_climbing_run1.gif'")
    plot_times(times, "Hill Climbing Execution Time per Run", "hc_run_times.png", label_color="green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=["sa", "hc"],
                        help="Choose agent: 'sa' for Simulated Annealing or 'hc' for Hill Climbing")
    parser.add_argument("--file", type=str, default="tsp_datasets/ch130.tsp",
                        help="Path to TSP file (default: ch130.tsp)")
    args = parser.parse_args()

    if args.agent == "sa":
        run_sa(args.file)
    elif args.agent == "hc":
        run_hc(args.file)

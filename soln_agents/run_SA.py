from SA.SA import run_tsp_experiment_from_file
import matplotlib.pyplot as plt

def plot_times(times):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(times) + 1), times, marker='o', label='Time per run')
    plt.axhline(y=sum(times)/len(times), color='r', linestyle='--', label='Average Time')
    plt.title("Simulated Annealing TSP - Time per Run")
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tsp_path = "ch130.tsp"  # Update path as needed
    times, costs = run_tsp_experiment_from_file(tsp_path, runs=5, timeout=600)
    plot_times(times)

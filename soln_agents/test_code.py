import time
import matplotlib.pyplot as plt
from HC.HC import HillClimbing  # Import Hill Climbing class
import random


num_cities = 200
random.seed(42) 
routes = [[random.randint(10, 100) if i != j else 0 for j in range(num_cities)] for i in range(num_cities)]

# Initialize the solver
tsp_solver = HillClimbing(num_cities, routes)


Timeout = 1000  # 10 seconds per run
num_runs = 5
convergence_times = []
best_distances = []

def run_with_time_limit():
    """Runs the TSP solver and records convergence time."""
    start_time = time.time()
    best_solution, best_distance = tsp_solver.solve()
    elapsed_time = time.time() - start_time

    convergence_times.append(elapsed_time)
    best_distances.append(best_distance)
    print(f"Run completed in {elapsed_time:.4f} seconds. Best Distance: {best_distance}")

# Run multiple times
for _ in range(num_runs):
    run_with_time_limit()

# Plot convergence times
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_runs + 1), convergence_times, marker='o', linestyle='-', color='b', label='Convergence Time')
plt.xlabel('Run Number')
plt.ylabel('Convergence Time (seconds)')
plt.title('Hill Climbing TSP Convergence Time Over Multiple Runs')
plt.legend()
plt.grid()
plt.show()

# Print results
print("Best Distances per Run:", best_distances)

import numpy as np
import matplotlib.pyplot as plt
import math
from HC.HC import HillClimbing

def distance_euc(point1, point2):
    """Calculate Euclidean distance between two cities."""
    return math.sqrt((float(point1[0]) - float(point2[0]))**2 + (float(point1[1]) - float(point2[1]))**2)

class TravellingSalesmanEnv:
    def __init__(self, name_tsp):
        """Initialize the TSP environment."""
        self.dist_matrix = None

        # Read raw data
        self.file_name = name_tsp
        with open(self.file_name, 'r') as file:
            data = file.read()

        self.lines = data.splitlines()

        # Store dataset information
        self.name = self.lines[0].split(': ')[1]
        self.nPoints = int(self.lines[3].split(': ')[1])

        # Read all data points and store them
        self.points = np.zeros((self.nPoints, 3))
        for i in range(self.nPoints-7):
            line_i = self.lines[7 + i].split()
            self.points[i, 0] = int(line_i[0])  # City number
            self.points[i, 1] = float(line_i[1])  # X coordinate
            self.points[i, 2] = float(line_i[2])  # Y coordinate

        self.create_dist_matrix()

    def create_dist_matrix(self):
        """Create a distance matrix using Euclidean distance."""
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))

        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = distance_euc(self.points[i][1:3], self.points[j][1:3])
        self.dist_matrix += self.dist_matrix.T  # Make it symmetric

def plot_tsp_env(env):
    """Plot the TSP environment (just cities and connections)."""
    plt.figure(figsize=(8, 6))

    # Extract coordinates
    x_coords = env.points[:, 1]
    y_coords = env.points[:, 2]

    # Plot cities
    plt.scatter(x_coords, y_coords, c='red', marker='o', label="Cities")

    # Connect each city with every other city (fully connected graph)
    for i in range(env.nPoints):
        for j in range(i + 1, env.nPoints):
            plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'gray', alpha=0.3)

    # Highlight start city
    plt.scatter(x_coords[0], x_coords[0], c='green', s=100, label="Start City")

    plt.title(f"TSP Environment: {env.name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
 print("hii from tsp env")
#     # Load TSP Environment from a File
#     tsp_env = TravellingSalesmanEnv("ch130.tsp")

#     # Plot the environment (before solving)
#     hc_solver = HillClimbing(130,tsp_env.dist_matrix)
#     best_path, best_distance = hc_solver.solve()

#     print(f"Best Path: {best_path}")
#     print(f"Best Distance: {best_distance}")

#     # Plot best solution
#     #plot_solution(tsp_env, best_path)
#     plot_tsp_env(tsp_env)

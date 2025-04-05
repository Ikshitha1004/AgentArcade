# SA.py
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio
from TSP.TSP import TravellingSalesman

class SimulatedAnnealing(TravellingSalesman):
    def __init__(self, no_of_cities, routes, points):
        super().__init__(no_of_cities, routes)
        self.points = points
        self.image_dir = "sa_frames"
        os.makedirs(self.image_dir, exist_ok=True)
        self.frames = []

    def cleanup_frames(self):
        """Remove old frames before a new run."""
        if os.path.exists(self.image_dir):
            for f in os.listdir(self.image_dir):
                os.remove(os.path.join(self.image_dir, f))
        self.frames = []

    def is_valid(self, path):
        for i in range(len(path) - 1):
            if self.routes[path[i]][path[i + 1]] == float('inf'):
                return False
        if self.routes[path[-1]][path[0]] == float('inf'):
            return False
        return True

    def plot_path(self, path, step):
        """Create and save plot of the path."""
        x = [self.points[i][0] for i in path] + [self.points[path[0]][0]]
        y = [self.points[i][1] for i in path] + [self.points[path[0]][1]]

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', color='blue')
        for i, city in enumerate(path):
            plt.text(self.points[city][0], self.points[city][1], str(city), fontsize=9)
        plt.title(f"Simulated Annealing Step {step}")
        path_img = os.path.join(self.image_dir, f"frame_{step}.png")
        plt.savefig(path_img)
        plt.close()
        self.frames.append(path_img)

    def create_gif(self, gif_path="simulated_annealing_result.gif"):
        """Compile saved frames into a GIF."""
        images = [imageio.imread(frame) for frame in self.frames]
        imageio.mimsave(gif_path, images, duration=0.5)

    def solve(self, initial_temp=10000, final_temp=1e-7, alpha=0.95, max_iter=100000, timeout=60, save_gif=False):
        
              
        self.cleanup_frames()
        n = self.n
        current_path = self.construct_initial_path()
        current_cost = self.calculate_distance(current_path)
        best_path = list(current_path)
        best_cost = current_cost

        T = initial_temp
        start_time = time.perf_counter()
        step = 0
        self.plot_path(current_path, step)
        step += 1

        while T > final_temp and step < max_iter:
            if time.perf_counter() - start_time > timeout:
                break

            i, j = random.sample(range(n), 2)
            new_path = list(current_path)
            new_path[i], new_path[j] = new_path[j], new_path[i]

            if not self.is_valid(new_path):
                continue

            new_cost = self.calculate_distance(new_path)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < np.exp(-delta / T):
                current_path = new_path
                current_cost = new_cost
                self.plot_path(current_path, step)
                step += 1

                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost

            T *= alpha

        if save_gif:
            self.create_gif()

        return best_path, best_cost, time.perf_counter() - start_time

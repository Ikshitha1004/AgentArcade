import matplotlib.pyplot as plt
import imageio
import os
from TSP.TSP import TravellingSalesman

class HillClimbing(TravellingSalesman):
    def __init__(self, no_of_cities, routes,points):
        super().__init__(no_of_cities, routes)
        self.frames = []  
        self.cities = points
        self.image_dir = "hc_frames"
        os.makedirs(self.image_dir, exist_ok=True)

    def plot_path(self, path, iteration):
        """Plot the current path and save it as an image."""
        x = [self.cities[city][0] for city in path] + [self.cities[path[0]][0]]
        y = [self.cities[city][1] for city in path] + [self.cities[path[0]][1]]

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'o-', color='blue')
        plt.title(f"Hill Climbing - Iteration {iteration}")
        for i, city in enumerate(path):
            plt.text(self.cities[city][0], self.cities[city][1], str(city), fontsize=12)
        filepath = os.path.join(self.image_dir, f"frame_{iteration}.png")
        plt.savefig(filepath)
        plt.close()
        self.frames.append(filepath)
    
    def is_valid(self, path):
        """Check if the given path is valid in the graph."""
        for i in range(len(path) - 1):
            if self.routes[path[i]][path[i + 1]] == float('inf'):
                return False
        if self.routes[path[-1]][path[0]] == float('inf'):  
            return False
        return True
    def neighborhood(self, path):
   
        neighbors = []
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                new_path = path[:]
                new_path[i], new_path[j] = new_path[j], new_path[i]
                
                if self.is_valid(new_path): 
                    neighbors.append(new_path)
                    
        return neighbors

    def solve(self, save_gif=False, gif_name="hill_climbing_result.gif"):
        self.frames = [] 
        os.makedirs(self.image_dir, exist_ok=True)

        current_solution = self.construct_initial_path()
        current_distance = self.calculate_distance(current_solution)
        self.plot_path(current_solution, 0)

        iteration = 1
        while True:
            neighbors = self.neighborhood(current_solution)
            if not neighbors:
                break

            next_solution = min(neighbors, key=self.calculate_distance)
            next_distance = self.calculate_distance(next_solution)

            if next_distance < current_distance:
                current_solution, current_distance = next_solution, next_distance
                self.plot_path(current_solution, iteration)
                iteration += 1
            else:
                break

        if save_gif:
            self.create_gif(gif_name)

        return current_solution, current_distance


    def create_gif(self, gif_name):
        """Create and save a GIF from stored image frames."""
        images = [imageio.imread(frame) for frame in self.frames]
        imageio.mimsave(gif_name, images, duration=0.7)  
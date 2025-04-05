import random
from TSP.TSP import TravellingSalesman

class HillClimbing(TravellingSalesman):
    def __init__(self, no_of_cities, routes):
        super().__init__(no_of_cities, routes)

    def is_valid(self, path):
        """Check if the given path is valid in the graph."""
        for i in range(len(path) - 1):
            if self.routes[path[i]][path[i + 1]] == float('inf'):
                return False
        if self.routes[path[-1]][path[0]] == float('inf'):  # Check return to start
            return False
        return True

    def neighborhood(self, path):
        """Generate neighbors by swapping 2 cities, ensuring validity."""
        neighbors = []
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                new_path = path[:]
                new_path[i], new_path[j] = new_path[j], new_path[i]
                
                if self.is_valid(new_path):  # Only keep valid paths
                    neighbors.append(new_path)
                    
        return neighbors

    def solve(self):
        """Run Hill Climbing to optimize the TSP route."""
        current_solution = self.construct_initial_path()
        current_distance = self.calculate_distance(current_solution)

        while True:
            neighbors = self.neighborhood(current_solution)
            
            if not neighbors:  # No valid neighbors, return current best
                break

            next_solution = min(neighbors, key=self.calculate_distance)
            next_distance = self.calculate_distance(next_solution)

            if next_distance < current_distance:  # Accept better solution
                current_solution, current_distance = next_solution, next_distance
            else:
                break  # Stop if no better solution is found

        return current_solution, current_distance

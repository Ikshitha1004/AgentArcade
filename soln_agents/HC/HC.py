from TSP.TSP import TravellingSalesman
class HillClimbing(TravellingSalesman):
 def __init__(self,no_of_cities,routes):
  super().__init__(no_of_cities,routes)
  
  def is_valid(self,path):
   for i in range(len(path)-1):
    if self.routes[path[i]][path[i+1]] == float('inf'):
     return False
    if self.routes[path[-1]][path[0]] == float('inf'):  # Check return to start
        return False
    return True
  
  def neighborhood(self,path):
   """Generate neighbors by swapping 2 cities"""
   neighbors = []
   for i in range(len(path)):
    for j in range(i+1,len(path)):
     new_path = path[:]
     new_path[i],new_path[j] = new_path[j],new_path[i]
     
     if self.is_valid(new_path):
      neighbors.append(new_path)
      
   return neighbors
  
  def solve(self):
   current_solution = list(range(self.n))
   ramdom.shuffle(current_solution)
   current_distance = self.calculate_distance(current_solution)
  #TODO : add a valid initial path to start off  
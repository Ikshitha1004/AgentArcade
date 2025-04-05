import math
from abc import ABC , abstractmethod
import random
class TravellingSalesman:
  def __init__(self, no_of_cities,routes):
   self.n = no_of_cities
   self.routes =  routes
  
  def calculate_distance(self,path):
   """Objective function to the agent"""
   distance = 0
   for i in range(self.n-1):
    distance += self.routes[path[i]][path[i+1]]
   distance += self.routes[path[-1]][path[0]]   #returning to the start node
   return distance
  
  def construct_initial_path(self):
   start_city = random.randint(0, self.n - 1)  # Pick a random start city
   path = [start_city]
   unvisited = set(range(self.n)) - {start_city}
   
   while unvisited:
     last_city = path[-1]
     next_city = min(
                 unvisited, 
                 key= lambda city:self.routes[last_city][city] if self.routes[last_city][city] != float('inf')else float('inf'),default = None)
     if next_city is None or self.routes[last_city][next_city] == float('inf'):
           return self.construct_initial_path()  # Restart if no valid path

     path.append(next_city)
     unvisited.remove(next_city)
     if self.routes[path[-1]][path[0]] == float('inf'):
       return self.construct_initial_path()  # Restart if no return path

   return path  
  @abstractmethod
  def solve(self):
   pass
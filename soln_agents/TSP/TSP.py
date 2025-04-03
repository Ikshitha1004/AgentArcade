import math
from abc import ABC , abstractmethod
class TravellingSalesman:
 def __init__(self, no_of_cities,routes):
  self.n = no_of_citiescities
  self.routes =  routes
  
  def calculate_distance(self,path):
   """Objective function to the agent"""
   distance = 0
   for i in range(self.n-1):
    distance += self.routes[path[i]][path[i+1]]
   distance += self.routes[path[-1]][path[0]]   #returning to the start node
   return distance
    
  @abstractmethod
  def solve(self):
   pass
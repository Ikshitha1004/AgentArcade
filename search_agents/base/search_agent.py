from abc import ABC , abstractmethod

class SearchAgent(ABC):
 def __init__(self, env, timeout = 60):
  self.env = env
  self.timeout = timeout
  
  @abstractmethod
  def driver(self):
   pass
  
  @abstractmethod
  def get_best_cost(self):
   pass
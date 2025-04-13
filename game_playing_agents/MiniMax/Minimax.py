import sys
import os

# Add the path to GameEnv folder (update the path accordingly)
sys.path.append(os.path.abspath("../GameEnv"))

from GameEnv import GameEnv

## -------- k-ply minimax ----------
class Minimax():
  def __init__(self,max_depth,env:GameEnv):
   self.max_depth = max_depth
   self.env = env
  
   
  def minimax(self,state,depth,is_max):
   if depth == 0 or self.env.is_terminal:
    return self.env.evaluate(state)
   if is_max:
    max_eval = float(-inf)
    for action in self.env.get_possible_actions(state,is_max):
     next_state = self.env.simulate_action(state,action,is_max)
     eval_score , _ = self.minimax(next_state,depth-1,False)
     if eval_score > max_eval:
      max_eval = eval_score
      best_action = action
    return max_eval,best_action
   else :
    min_eval = float(inf)
    for action in self.env.get_possible_actions(state,is_max):
     next_state = self.env.simulate_action(state,action,is_max)
     eval_score , _ = self.minimax(next_state,depth-1,True)
     if eval_score < min_eval: 
      min_eval = eval_score
      best_action = action
   return min_eval,best_action
 
 
 
  def get_best_action(self,state):
   _,best_action = self.minimax(state,self.max_depth,True)
   return best_action
    
    
   
   
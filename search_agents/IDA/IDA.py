import math
import time
from search_agents.base.search_agent import SearchAgent

class IterativeDeepeningAStar(SearchAgent):
    def __init__(self, env, timeout=60):
        super().__init__(env, timeout)
        self.U = math.inf
        self.q = 0
        self.goal_position = env.observation_space.n - 1 
        self.mini = self.heuristic(self.env.reset()[0])
        self.start_time = time.time()  
        self.execution_time = 0

    def heuristic(self, state):
        """ Heuristic function: Manhattan distance to the goal state. """
        grid_size = int(math.sqrt(self.env.observation_space.n)) 
        goal_x, goal_y = divmod(self.goal_position, grid_size)
        state_x, state_y = divmod(state, grid_size)
        return abs(goal_x - state_x) + abs(goal_y - state_y)  

    def is_goal(self, state):
        """ Check if the current state is the goal state."""
        return state == self.goal_position

    def expand(self, state):
        """ Return the possible actions from the given state."""
        return list(range(self.env.action_space.n))

    def driver(self):
        start_time = time.time()
        flag = False
        
        while not flag and (time.time() - start_time) < self.timeout:
            self.U = self.mini
            self.mini = math.inf
            visited = set()
            flag = self.IDA(0, self.mini, self.U, self.env.reset()[0],visited)
        self.execution_time = time.time() - self.start_time

    def IDA(self, q, mini, U, state, visited):
        if self.is_goal(state):
            self.U = q
            return True
        
        for action in self.expand(state):
            if (state,action) in visited:
             continue
            visited.add((state,action))
            prev_state = state 
            next_state, reward, terminated, turcated, _ = self.env.step(action)
            done = turcated or terminated
            
            if done and next_state != self.goal_position: 
                self.env.env.s = prev_state
                continue
            
            if q + 1 + self.heuristic(next_state) <= U:
                self.q = q + 1
                if self.IDA(self.q, mini, U, next_state,visited):
                    return True
            else:
                if q + 1 + self.heuristic(next_state) < self.mini:
                    self.mini = q + 1 + self.heuristic(next_state)
            
            self.env.env.s = prev_state 
        
        return False
       
    def get_best_cost(self):
     return self.U

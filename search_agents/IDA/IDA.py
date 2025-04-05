import time
from base.search_agent import SearchAgent  # Update if your base class is elsewhere

class IterativeDeepeningAStar(SearchAgent):
    def __init__(self, env):
        self.env = env
        self.goal_position = env.observation_space.n - 1
        self.size = int(env.observation_space.n ** 0.5)
        self.q = 0
        self.execution_time = 0
        self.final_path = []  # To store the path (state, action)

    def heuristic(self, state):
        x1, y1 = divmod(state, self.size)
        x2, y2 = divmod(self.goal_position, self.size)
        return abs(x1 - x2) + abs(y1 - y2)

    def is_goal(self, state):
        return state == self.goal_position

    def is_pit(self, state):
        row, col = divmod(state, self.size)
        return self.env.unwrapped.desc[row][col] == b'H'

    def expand(self, state):
        successors = []
        for action in range(self.env.action_space.n):
            self.env.unwrapped.s = state
            next_state, _, done, _, _ = self.env.step(action)
            if not self.is_pit(next_state):
                successors.append((next_state, action))
        return successors

    def driver(self):
        self.env.reset()
        start_time = time.perf_counter()
        threshold = self.heuristic(0)
        path = [(0, None)]
        visited = set()
        while True:
            t = self.IDA(0, threshold, 0, visited, path)
            if isinstance(t, list):
                self.final_path = t
                self.q = len(t) - 1  # Exclude dummy first step
                break
            if t == float('inf'):
                break
            threshold = t
        self.execution_time = time.perf_counter() - start_time

    def IDA(self, g, threshold, state, visited, path):
        f = g + self.heuristic(state)
        if f > threshold:
            return f
        if self.is_goal(state):
            return path.copy()
        min_threshold = float('inf')
        visited.add(state)

        for succ, action in self.expand(state):
            if succ not in visited:
                path.append((succ, action))
                t = self.IDA(g + 1, threshold, succ, visited, path)
                if isinstance(t, list):
                    return t
                if t < min_threshold:
                    min_threshold = t
                path.pop()
        visited.remove(state)
        return min_threshold

    def get_best_cost(self):
        return self.q

    def get_final_path(self):
        return self.final_path

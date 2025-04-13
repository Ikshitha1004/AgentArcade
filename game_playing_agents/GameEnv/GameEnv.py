from abc import ABC, abstractmethod
import chess
import copy

class GameEnv(ABC):
    @abstractmethod
    def get_possible_actions(self, state, is_maximizing: bool):
        pass

    @abstractmethod
    def simulate_action(self, state, action, is_maximizing: bool):
        pass

    @abstractmethod
    def evaluate(self, state):
        pass

    @abstractmethod
    def is_terminal(self, state, depth):
        pass

class SlimeVolleyEnvWrapper(GameEnv):
    def __init__(self, env):
        self.env = env

    def get_possible_actions(self, state, is_maximizing):
        return list(range(8))

    def simulate_action(self, state, action, is_maximizing):
        env_copy = copy.deepcopy(self.env)
        if is_maximizing:
            obs, _, _, _ = env_copy.step((action, 0))  # Opponent does nothing
        else:
            obs, _, _, _ = env_copy.step((0, action))
        return env_copy

    def evaluate(self, state):
        obs = state.observation  
        return -abs(obs[0] - obs[2])  # ball_x - slime_x

    def is_terminal(self, state, depth):
        return depth == 0


class ChessEnvWrapper(GameEnv):
     def __init__(self, env):
        self.env = env

     def get_possible_actions(self, state, is_maximizing):
         return list(range(self.env.action_space.n))

     def simulate_action(self, state, action, is_maximizing):
         import copy
         env_copy = copy.deepcopy(self.env)
         obs, reward, done, info = env_copy.step(action)
         return env_copy

     def evaluate(self, state):
         # Simplified evaluation from UCI representation
         board = state.board
         return self.material_score(board)

     def material_score(self, board):
         import chess
         values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
         score = 0
         for piece_type in values:
             score += len(board.pieces(piece_type, chess.WHITE)) * values[piece_type]
             score -= len(board.pieces(piece_type, chess.BLACK)) * values[piece_type]
         return score

     def is_terminal(self, state, depth):
         return depth == 0 or state.done
import sys
import os

# Add the path to GameEnv folder (update if needed)
sys.path.append(os.path.abspath("../GameEnv"))

from GameEnv import GameEnv
import chess
class AlphaBeta:
    def __init__(self, max_depth, env: GameEnv):
        self.max_depth = max_depth
        self.env = env
        self.nodes_explored = 0  # Counter to track nodes

    def alpha_beta(self, state, depth, alpha, beta, is_max):
        self.nodes_explored += 1  # Increment for each node visited

        board = state.unwrapped._board

        if depth == 0 or self.env.is_terminal(state, depth):
            return self.env.evaluate(state), None

        legal_moves = list(board.legal_moves)

        if is_max:
            max_eval = float('-inf')
            best_action = None
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.alpha_beta(state, depth - 1, alpha, beta, False)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # β cutoff
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.alpha_beta(state, depth - 1, alpha, beta, True)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # α cutoff
            return min_eval, best_action

    def get_best_action(self, state, is_max):
        self.nodes_explored = 0  # Reset counter at each top-level call
        _, best_action = self.alpha_beta(state, self.max_depth, float('-inf'), float('inf'), is_max)
        return best_action
import sys
import os

# Add the path to GameEnv folder (update the path accordingly)
sys.path.append(os.path.abspath("../GameEnv"))

from GameEnv import GameEnv
import chess

class Minimax:
    def __init__(self, max_depth, env: GameEnv):
        self.max_depth = max_depth
        self.env = env

    def minimax(self, state, depth, is_max):
        board = state.unwrapped._board

        if depth == 0 or self.env.is_terminal(state, depth):
            return self.env.evaluate(state), None

        legal_moves = list(board.legal_moves)

        if is_max:
            max_eval = float('-inf')
            best_action = None
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(state, depth - 1, False)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = move
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(state, depth - 1, True)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = move
            return min_eval, best_action

    def get_best_action(self, state):
        _, best_action = self.minimax(state, self.max_depth, True)
        return best_action

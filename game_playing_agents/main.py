# import gym
# import gym_chess
# import time
# import os
# import platform
# import argparse
# from MiniMax.Minimax import Minimax
# from AlphaBeta.alphabeta import AlphaBeta
# from GameEnv.GameEnv import ChessEnvWrapper



# def clear_terminal():
#     if platform.system() == "Windows":
#         os.system("cls")
#     else:
#         os.system("clear")

# def main(agent_type):
#     # 1. Setup environment
#     raw_env = gym.make("Chess-v0")
#     state = raw_env.reset()
#     wrapped_env = ChessEnvWrapper(raw_env)

#     # 2. Create the agent based on the argument passed
#     if agent_type == "minimax":
#         white_agent = Minimax(max_depth=5, env=wrapped_env)
#         black_agent = Minimax(max_depth=5, env=wrapped_env)
#     elif agent_type == "alphabeta":
#         white_agent = AlphaBeta(max_depth=5, env=wrapped_env)
#         black_agent = AlphaBeta(max_depth=5, env=wrapped_env)
#     else:
#         print(f"Unknown agent type: {agent_type}. Please use 'minimax' or 'alphabeta'.")
#         return

#     done = False
#     turn = 0

#     while not done:
#         clear_terminal()
#         # Print turn and whose turn it is
#         print(f"\nTurn {turn + 1}: {'WHITE (Max)' if turn % 2 == 0 else 'BLACK (Min)'}")

#         # Determine whose turn it is and get the best action for that player
#         if turn % 2 == 0:
#             best_action = white_agent.get_best_action(raw_env, is_max=True)
#             player = "White"
#         else:
#             best_action = black_agent.get_best_action(raw_env, is_max=False)
#             player = "Black"

#         if best_action is not None:
#             # Print move in a more readable format (e.g., e2 to e4)
#             print(f"{player}'s move: {best_action}")
#             obs, reward, done, info = raw_env.step(best_action)
#         else:
#             print(f"{player} has no valid move, skipping turn.")
#             reward, done = 0, True

#         print("Reward:", reward, "Done:", done)

#         # Print the current board state
#         print(raw_env.render(mode="unicode"))
#         time.sleep(5)

#         turn += 1

#     print("\nGame Over!")
#     print("Final reward:", reward)

# if __name__ == "__main__":
#     # Argument parser
#     parser = argparse.ArgumentParser(description="Chess agent selection.")
#     parser.add_argument(
#         '--agent', 
#         type=str, 
#         choices=['minimax', 'alphabeta'], 
#         required=True, 
#         help="Choose the agent: 'minimax' or 'alphabeta'."
#     )
#     args = parser.parse_args()

#     # Call the main function with the chosen agent
#     main(args.agent)
import gym
import gym_chess
import time
import os
import platform
import argparse
from MiniMax.Minimax import Minimax
from AlphaBeta.alphabeta import AlphaBeta
from GameEnv.GameEnv import ChessEnvWrapper
import matplotlib.pyplot as plt

def clear_terminal():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def main(agent_type):
    # 1. Setup environment
    raw_env = gym.make("Chess-v0")
    state = raw_env.reset()
    wrapped_env = ChessEnvWrapper(raw_env)

    # 2. Create the agent based on the argument passed
    if agent_type == "minimax":
        white_agent = Minimax(max_depth=2, env=wrapped_env)
        black_agent = Minimax(max_depth=2, env=wrapped_env)
    elif agent_type == "alphabeta":
        white_agent = AlphaBeta(max_depth=2, env=wrapped_env)
        black_agent = AlphaBeta(max_depth=2, env=wrapped_env)
    else:
        print(f"Unknown agent type: {agent_type}. Please use 'minimax' or 'alphabeta'.")
        return

    done = False
    turn = 0

    # Tracking nodes explored per turn
    white_nodes = []
    black_nodes = []

    while not done:
        clear_terminal()
        # Print turn and whose turn it is
        print(f"\nTurn {turn + 1}: {'WHITE (Max)' if turn % 2 == 0 else 'BLACK (Min)'}")

        # Determine whose turn it is and get the best action for that player
        if turn % 2 == 0:
            best_action = white_agent.get_best_action(raw_env, is_max=True)
            white_nodes.append(white_agent.nodes_explored)  # Track nodes explored for White
            player = "White"
        else:
            best_action = black_agent.get_best_action(raw_env, is_max=False)
            black_nodes.append(black_agent.nodes_explored)  # Track nodes explored for Black
            player = "Black"

        if best_action is not None:
            # Print move in a more readable format (e.g., e2 to e4)
            print(f"{player}'s move: {best_action}")
            obs, reward, done, info = raw_env.step(best_action)
        else:
            print(f"{player} has no valid move, skipping turn.")
            reward, done = 0, True

        print("Reward:", reward, "Done:", done)

        # Print the current board state
        print(raw_env.render(mode="unicode"))
        time.sleep(5)

        turn += 1

    print("\nGame Over!")
    print("Final reward:", reward)

    # Plot the number of nodes explored per turn
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(white_nodes)+1), white_nodes, label='White Agent (Max)', marker='o')
    plt.plot(range(1, len(black_nodes)+1), black_nodes, label='Black Agent (Min)', marker='s')

    plt.xlabel('Turn')
    plt.ylabel('Nodes Explored')
    plt.title(f'Nodes Explored per Turn ({agent_type.title()})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'nodes_explored_{agent_type}.png')  # Save the plot
    plt.show()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Chess agent selection.")
    parser.add_argument(
        '--agent', 
        type=str, 
        choices=['minimax', 'alphabeta'], 
        required=True, 
        help="Choose the agent: 'minimax' or 'alphabeta'."
    )
    args = parser.parse_args()

    # Call the main function with the chosen agent
    main(args.agent)

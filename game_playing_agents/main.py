import gym
import gym_chess 
from MiniMax.Minimax import Minimax
from GameEnv.GameEnv import ChessEnvWrapper
import time
from gym.wrappers import Monitor  
import os
import platform

def clear_terminal():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
        
def main():
    # 1. Setup environment
    raw_env = gym.make("Chess-v0")
    raw_env = Monitor(raw_env, directory="videos", force=True)  # Wrap environment to record video
    state = raw_env.reset()
    wrapped_env = ChessEnvWrapper(raw_env)

    # 2. Create two Minimax agents (both use same logic but alternate roles)
    white_agent = Minimax(max_depth=2, env=wrapped_env)
    black_agent = Minimax(max_depth=2, env=wrapped_env)

    # 3. Alternate turns until game ends
    done = False
    turn = 0  # 0 for White (Max), 1 for Black (Min)

    while not done:
        clear_terminal()
        print(f"\nTurn {turn + 1}: {'WHITE (Max)' if turn % 2 == 0 else 'BLACK (Min)'}")

        if turn % 2 == 0:
            # White's turn (maximizing)
            best_action = white_agent.get_best_action(raw_env)
        else:
            # Black's turn (minimizing) — reverse evaluation in get_best_action
            best_action = black_agent.get_best_action(raw_env)

        print(f"Best action: {best_action}")
        if best_action is not None:
            obs, reward, done, info = raw_env.step(best_action)
            if done:
                print("Game over.")
                break
        else:
            print("Skipping move — no valid action.")
        
        print("Reward:", reward, "Done:", done)

        turn += 1
        print(raw_env.render(mode="unicode"))
        time.sleep(5)  

    print("\nGame Over!")
    print("Final reward:", reward)

   
    print("Video saved in 'videos' folder.")

if __name__ == "__main__":
    main()

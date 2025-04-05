# import gym
# #from gymnasium.envs.toy_text.frozen_lake import generate_random_map
# import numpy as np
# #from search_agents.BnB.BnB import BranchAndBound
# from search_agents.IDA.IDA import IterativeDeepeningAStar
# import matplotlib.pyplot as plt

# custom_map = [
#     "SFFF",
#     "FFFH",
#     "FFFF",
#     "HFFG"
# ]

# #env = gym.make("FrozenLake-v1",desc = custom_map, is_slippery= False,render_mode="rgb_array")


# def run_experiments(agent_class,env,runs = 7):
#     times = []
#     for _ in range(runs):
#         agent = agent_class(env)
#         agent.driver()
#         print("Best cost:" , agent.get_best_cost())
#         times.append(agent.execution_time)
        
#     plt.plot(range(1, runs + 1), times, marker='o', linestyle='-', label=agent_class.__name__)
#     plt.xlabel('Run')
#     plt.ylabel('Time (s)')
#     plt.title(f'Execution Time for {agent_class.__name__}')
#     plt.legend()
#     plt.show()
    
#     print(f'Execution Times: {times}')
    
# if __name__ =="__main__":
#     env =gym.make('FrozenLake-v1', desc=custom_map,render_mode="human")
#    # env = gym.make("FrozenLake-v1",desc = custom_map, is_slippery= False,render_mode="human")
#     env.reset()
#     run_experiments(IterativeDeepeningAStar,env)
import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time

from search_agents.IDA.IDA import IterativeDeepeningAStar  # Adjust path if needed

custom_map =  [
     "SFFF",
    "FFFH",
    "FFFF",
    "HFFG"
]

def save_path_as_gif(env, path, filename="ida_final_path.gif"):
    frames = []
    obs, _ = env.reset()
    env.unwrapped.s = 0  # Reset state to start

    frame = env.render()
    frames.append(frame)

    for state, action in path[1:]:  # Skip initial dummy (0, None) entry
        obs, _, done, _, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        if done:
            break

    imageio.mimsave(filename, frames, duration=0.8)
    print(f"\n🎥 GIF saved as: {filename}")

def run_experiments(agent_class, env, runs=7):
    times = []
    final_path = []

    for i in range(runs):
        agent = agent_class(env)
        agent.driver()
        print("Best cost:", agent.get_best_cost())
        times.append(agent.execution_time)

        if i == runs - 1:  # Save path only for the last run
            final_path = agent.get_final_path()

    # Plotting execution times
    plt.plot(range(1, runs + 1), times, marker='o', linestyle='-', label=agent_class.__name__)
    plt.xlabel('Run')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time for {agent_class.__name__}')
    plt.legend()
    plt.show()

    print(f'\nExecution Times: {times}')

    # Save path to GIF
    if final_path:
        save_path_as_gif(env, final_path)

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=custom_map, render_mode="human", is_slippery=False)
    env.reset()
    run_experiments(IterativeDeepeningAStar, env)

import gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
#from search_agents.BnB.BnB import BranchAndBound
from search_agents.IDA.IDA import IterativeDeepeningAStar
import matplotlib.pyplot as plt

custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

#env = gym.make("FrozenLake-v1",desc = custom_map, is_slippery= False,render_mode="rgb_array")


def run_experiments(agent_class,env,runs = 7):
    times = []
    for _ in range(runs):
        agent = agent_class(env)
        agent.driver()
        print("Best cost:" , agent.get_best_cost())
        times.append(agent.execution_time)
        
    plt.plot(range(1, runs + 1), times, marker='o', linestyle='-', label=agent_class.__name__)
    plt.xlabel('Run')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time for {agent_class.__name__}')
    plt.legend()
    plt.show()
    
    print(f'Execution Times: {times}')
    
if __name__ =="__main__":
    env =gym.make('FrozenLake-v1', desc=generate_random_map(size=4),render_mode="human")
    env.reset()
    run_experiments(IterativeDeepeningAStar,env)
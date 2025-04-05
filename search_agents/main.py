import argparse
import gym
import matplotlib.pyplot as plt
import imageio
import time

from BnB.BnB import BranchAndBound
from IDA.IDA import IterativeDeepeningAStar

custom_map = [
    "SFFF",
    "FFFH",
    "FFFF",
    "HFFG"
]

def save_path_as_gif(env, path, filename):
    frames = []
    env.reset()
    env.unwrapped.s = 0
    frames.append(env.render())

    for state, action in path[1:]:  # Skip dummy (0, None)
        _, _, done, _, _ = env.step(action)
        frames.append(env.render())
        if done:
            break

    imageio.mimsave(filename, frames, duration=0.8)
    print(f"\nGIF saved as: {filename}")


def run_ida():
    runs = 5
    env = gym.make('FrozenLake-v1', desc=custom_map, render_mode="rgb_array", is_slippery=False)
    times = []
    costs = []
    final_path = []

    for i in range(runs):
        print(f"\n--- Run {i + 1} ---")
        agent = IterativeDeepeningAStar(env)
        agent.driver()
        cost = agent.get_best_cost()
        exec_time = agent.execution_time
        print(f"Run {i + 1} - Best Cost: {cost}, Time: {exec_time:.6f}s")
        times.append(exec_time)
        costs.append(cost)

        if i == runs - 1:
            final_path = agent.get_final_path()

    avg_time = sum(times) / runs
    avg_cost = sum(costs) / runs if all(c != float('inf') for c in costs) else float('inf')
    plot_name = "IDA_times.png"

    plt.plot(range(1, runs + 1), times, marker='o', label="IDA* Time")
    plt.axhline(avg_time, color='r', linestyle='--', label=f'Avg Time: {avg_time:.6f}s')
    plt.xlabel('Run')
    plt.ylabel('Time (s)')
    plt.title('Execution Times: IDA*')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()
    print(f"Plot saved as {plot_name}")
    print(f"\nAverage Time: {avg_time:.6f}s")
    print(f"Average Cost: {avg_cost:.2f}")

    if final_path:
        gif_name = "IDA_final_path.gif"
        save_path_as_gif(env, final_path, gif_name)

    env.close()



def run_bnb():
    agent = BranchAndBound(timeout_sec=600)
    agent.run_multiple_searches(runs=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run search experiments on FrozenLake-v1")
    parser.add_argument("--agent", type=str, required=True, choices=["bnb", "ida"],
                        help="Choose the agent: 'bnb' for Branch and Bound, 'ida' for IDA*")

    args = parser.parse_args()

    if args.agent == "bnb":
        run_bnb()
    elif args.agent == "ida":
        run_ida()

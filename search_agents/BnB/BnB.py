import gymnasium as gym
import time
import matplotlib.pyplot as plt
import imageio

action_names = ['←', '↓', '→', '↑']

def dfbnb_frozenlake_run(timeout_sec=600):
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    desc = env.unwrapped.desc.astype(str)
    n_rows, n_cols = desc.shape
    n_states = env.observation_space.n
    transitions = env.unwrapped.P

    goal_state = n_states - 1
    start_state = 0
    U = [float('inf')]
    best_path = []
    best_action_path = []

    def is_safe(state):
        row = state // n_cols
        col = state % n_cols
        return desc[row][col] != 'H'

    def dfbnb(state, cost_so_far, path, actions, start_time):
        nonlocal best_path, best_action_path

        # Check timeout
        if time.time() - start_time > timeout_sec:
            raise TimeoutError("DFBnB terminated due to timeout.")

        if state == goal_state:
            if cost_so_far < U[0]:
                U[0] = cost_so_far
                best_path[:] = path
                best_action_path[:] = actions
                print(f"New best path found: {[action_names[a] for a in actions]} (cost: {cost_so_far})")
            return

        for action in range(4):
            for prob, next_state, reward, done in transitions[state][action]:
                if not is_safe(next_state) or next_state in path:
                    continue

                new_cost = cost_so_far + 1
                current_action_path = actions + [action]
                print(f"Exploring path: {[action_names[a] for a in current_action_path]} (cost: {new_cost})")

                if new_cost >= U[0]:
                    print(f"Bounded: cost {new_cost} ≥ current best {U[0]} — backtracking")
                    continue

                dfbnb(next_state, new_cost, path + [next_state], current_action_path, start_time)

    try:
        start_time = time.time()
        dfbnb(start_state, 0, [start_state], [], start_time)
        total_time = time.time() - start_time
    except TimeoutError as e:
        print(str(e))
        return [], float('inf'), timeout_sec

    return best_action_path, len(best_action_path), total_time


def render_final_path_gif(actions, gif_path="dfbnb_path.gif"):
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    frames = []
    obs, _ = env.reset()
    frames.append(env.render())

    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        frames.append(env.render())
        if done:
            break

    env.close()
    imageio.mimsave(gif_path, frames, fps=1)


def run_experiments(runs=5, timeout_sec=600):
    results = []

    for i in range(runs):
        print(f"\n--- Run {i+1} ---")
        path, cost, runtime = dfbnb_frozenlake_run(timeout_sec=timeout_sec)
        if cost == float('inf'):
            print("Goal not reached within time limit.")
        else:
            print(f"Path: {[action_names[a] for a in path]}")
            print(f"Cost: {cost}, Time taken: {runtime:.3f} seconds")
        results.append((cost, runtime, path))

    # Plot average time
    times = [r[1] for r in results]
    avg_time = sum(times) / len(times)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, runs + 1), times, marker='o')
    plt.axhline(avg_time, color='r', linestyle='--', label=f'Average time: {avg_time:.3f}s')
    plt.title("DFBnB Execution Time per Run")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dfbnb_times.png")
    plt.show()

    # Generate GIF for best (shortest) path if any valid
    valid_runs = [r for r in results if r[0] != float('inf')]
    if valid_runs:
        best_run = min(valid_runs, key=lambda x: x[0])
        render_final_path_gif(best_run[2])
        print("\nGIF saved as 'dfbnb_path.gif'")
    else:
        print("No successful runs to generate GIF.")

    print("Plot saved as 'dfbnb_times.png'")

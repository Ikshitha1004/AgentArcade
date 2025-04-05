import gymnasium as gym
import time
import matplotlib.pyplot as plt
import imageio

class BranchAndBound:
    def __init__(self, env_name="FrozenLake-v1", timeout_sec=600):
        self.env_name = env_name
        self.timeout_sec = timeout_sec
        self.action_names = ['←', '↓', '→', '↑']

    def run_single_search(self):
        env = gym.make(self.env_name, is_slippery=False, render_mode="rgb_array")
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

            if time.perf_counter() - start_time > self.timeout_sec:
                raise TimeoutError("DFBnB terminated due to timeout.")

            if state == goal_state:
                if cost_so_far < U[0]:
                    U[0] = cost_so_far
                    best_path[:] = path
                    best_action_path[:] = actions
                    print(f"New best path: {[self.action_names[a] for a in actions]} (cost: {cost_so_far})")
                return

            for action in range(4):
                for prob, next_state, reward, done in transitions[state][action]:
                    if not is_safe(next_state) or next_state in path:
                        continue

                    new_cost = cost_so_far + 1
                    current_action_path = actions + [action]

                    if new_cost >= U[0]:
                        print(f"Pruned: cost {new_cost} ≥ UB {U[0]}")
                        continue

                    dfbnb(next_state, new_cost, path + [next_state], current_action_path, start_time)

        try:
            start_time = time.perf_counter()
            dfbnb(start_state, 0, [start_state], [], start_time)
            total_time = time.perf_counter() - start_time
        except TimeoutError as e:
            print(str(e))
            return [], float('inf'), self.timeout_sec

        return best_action_path, len(best_action_path), total_time

    def render_gif(self, actions, gif_path="dfbnb_path.gif"):
        env = gym.make(self.env_name, is_slippery=False, render_mode="rgb_array")
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

    def run_multiple_searches(self, runs=5):
        results = []

        for i in range(runs):
            print(f"\n--- Run {i+1} ---")
            path, cost, runtime = self.run_single_search()
            if cost == float('inf'):
                print("Goal not reached in time.")
            else:
                print(f"Path: {[self.action_names[a] for a in path]}")
                print(f"Cost: {cost}, Time: {runtime:.3f}s")
            results.append((cost, runtime, path))

        times = [r[1] for r in results]
        avg_time = sum(times) / len(times)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, runs + 1), times, marker='o')
        plt.axhline(avg_time, color='r', linestyle='--', label=f'Avg: {avg_time:.3f}s')
        plt.title("DFBnB Runtime per Run")
        plt.xlabel("Run")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("dfbnb_times.png")
        plt.show()

        # Render best path as GIF
        valid_runs = [r for r in results if r[0] != float('inf')]
        if valid_runs:
            best_run = min(valid_runs, key=lambda x: x[0])
            self.render_gif(best_run[2])
            print("\nGIF saved as 'dfbnb_path.gif'")
        else:
            print("No successful runs to render.")

        print("Time plot saved as 'dfbnb_times.png'")

import pickle
import random
from env import DynamicTaxiEnv
from state import obs_to_state

with open('q_table.pkl', 'rb') as f:
    Q = pickle.load(f)

from collections import Counter
phase_counts = Counter(state[0] for state in Q)
under_counts = Counter(state[7] for state in Q)
wrongway_count = sum(1 for state in Q if state[7] == 3)
print("Phase distribution:", phase_counts)
print("Under-tile distribution:", under_counts)
print("States with wrong-way tile:", wrongway_count)

print("\n--- Running 3 episodes with full logging ---")

action_names = ['turn_left','turn_right','forward','pickup',
                'dropoff','highway','refuel','wait']

for ep in range(3):
    env = DynamicTaxiEnv(grid_size=7, fuel_limit=500)
    obs, _ = env.reset()
    done = False
    step = 0
    total = 0

    print(f"\n=== Episode {ep+1} ===")

    while not done and step < 50:
        state = obs_to_state(obs)
        pre_obs = obs

        if state not in Q:
            action = random.randint(0, 7)
            src = "random"
        else:
            action = max(range(8), key=lambda a: Q[state].get(a, 0.0))
            src = "Q"

        obs, reward, done, _ = env.step(action)
        total += reward
        step += 1

        print(f"  step {step:3d} | phase={state[0]} zone={pre_obs[3]} "
              f"carried={pre_obs[4]} under={state[7]} | "
              f"{action_names[action]:12s} ({src}) "
              f"| reward={reward:7.2f} | total={total:8.2f}")

    print(f"  Episode ended: steps={step}, total={total:.2f}, done={done}")
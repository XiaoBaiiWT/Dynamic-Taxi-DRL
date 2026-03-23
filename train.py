import random
import pickle
from collections import defaultdict
from env import DynamicTaxiEnv
from state import obs_to_state
from shaping import phi, action_penalty
from intrinsic import VisitCounter
 
NUM_EPISODES    = 200000
ALPHA           = 0.3
GAMMA           = 0.99
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = (EPS_START - EPS_END) / (0.8 * NUM_EPISODES)
 
CURRICULUM_STEP = 50000
GRID_SMALL      = (5, 7)
GRID_FULL       = (5, 10)   # match eval range exactly
 
MAX_STEPS       = 300       # hard cap, matches eval budget
 
Q       = defaultdict(lambda: defaultdict(float))
counter = VisitCounter(beta=0.5)
epsilon = EPS_START
total_rewards   = []
episode_lengths = []
 
for episode in range(NUM_EPISODES):
    if episode < CURRICULUM_STEP:
        grid_size = random.randint(*GRID_SMALL)
    else:
        grid_size = random.randint(*GRID_FULL)
 
    env = DynamicTaxiEnv(grid_size=grid_size, fuel_limit=500)
    obs, _ = env.reset()
    state   = obs_to_state(obs)
    done    = False
    episode_reward = 0
    step = 0
 
    while not done:
        if random.random() < epsilon:
            action = random.randint(0, 7)
        else:
            action = max(range(8), key=lambda a: Q[state][a])
 
        next_obs, r_env, done, _ = env.step(action)
        episode_reward += r_env
        next_state = obs_to_state(next_obs)
 
        if done:
            F = -phi(obs)
        else:
            F = GAMMA * phi(next_obs) - phi(obs)
 
        r_total = r_env + F + counter.get_bonus(state) + action_penalty(obs, action)
 
        next_best = max(Q[next_state].values()) if Q[next_state] else 0.0
        Q[state][action] += ALPHA * (r_total + GAMMA * next_best - Q[state][action])
 
        counter.update(state)
        counter.decay_beta()
        obs   = next_obs
        state = next_state
 
        step += 1
        if step >= MAX_STEPS:
            break
 
    total_rewards.append(episode_reward)
    episode_lengths.append(step)
    epsilon = max(EPS_END, epsilon - EPS_DECAY)
 
    if episode % 1000 == 0 and episode > 0:
        avg = sum(total_rewards[-1000:]) / 1000
        avg_len = sum(episode_lengths[-1000:]) / 1000
        print(f'Episode {episode}/{NUM_EPISODES}, eps={epsilon:.3f}, '
              f'avg={avg:.1f}, avg_len={avg_len:.0f}, Q_states={len(Q)}')
 
with open('q_table.pkl', 'wb') as f:
    pickle.dump(dict(Q), f)
print('Training complete. Q-table saved.')
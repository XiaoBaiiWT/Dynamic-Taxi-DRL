from state import get_phase, get_target, tile_under
 
def phi(obs):
    phase  = get_phase(obs)
    target = get_target(obs, phase)
    d      = abs(obs[0]-target[0]) + abs(obs[1]-target[1])
    load   = obs[4] / 4
 
    if phase == 0:
        scale = 3.0 if obs[27] > 0 else 1.0
        return scale / (1 + d)
    elif phase == 1:
        return 18.0 + 9*load + (9 / (1 + d))
    elif phase == 2:
        return 36.0 + 6*load + (9 / (1 + d))
 
def action_penalty(obs, action):
    phase = get_phase(obs)
    under = tile_under(obs)
    zone = obs[3]

    # Movement actions (0,1,2) — always allowed, no penalty
    if action in (0, 1, 2):
        return 0.0

    # pickup: only phase 0 on station
    if action == 3:
        if phase == 0 and under == 1:
            return 0.0
        return -5.0

    # dropoff: only phase 2, zone 2, on station
    if action == 4:
        if phase == 2 and zone == 2 and under == 1:
            return 0.0
        return -5.0

    # highway: only on valid highway tile
    if action == 5:
        if under == 2:
            return 0.0
        if under == 3:
            return -20.0
        return -5.0

    # refuel: only on gas station
    if action == 6:
        if under == 4:
            return 0.0
        return -5.0

    # wait: only on red light
    if action == 7:
        if -10 <= obs[14] <= -1:
            return 0.0
        return -5.0

    return 0.0
 
def shaped_reward(obs, r_env, next_obs, done, gamma=0.99):
    """Evaluation only — do not use in the training Q-update."""
    if done:
        return r_env - phi(obs)
    return r_env + gamma * phi(next_obs) - phi(obs)
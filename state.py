import math

def get_phase(obs):
    if obs[3] == 2 and obs[4] > 0:
        return 2
    elif obs[3] == 2 and obs[4] == 0:
        return 1
    elif obs[3] != 2 and obs[4] > 0:
        return 1
    else:
        return 0

def get_target(obs, phase):
    if phase == 0:
        if obs[3] == 3:
            return (obs[8], obs[9])
        else:
            station = [(obs[19+(2*i)], obs[20+(2*i)]) for i in range(4)]
            return min(station, key=lambda s: abs(s[0]-obs[0]) + abs(s[1]-obs[1]))

    if phase == 1:
        if obs[3] == 1:
            return (obs[28], obs[29])
        elif obs[3] == 3:
            return (obs[44], obs[45])
        elif obs[3] == 2:
            if obs[4] == 0:
                return (obs[6], obs[7])
            else:
                station = [(obs[32+(2*i)], obs[33+(2*i)]) for i in range(4)]
                return min(station, key=lambda s: abs(s[0]-obs[0]) + abs(s[1]-obs[1]))
        else:
            return (obs[0], obs[1])

    if phase == 2:
        station = [(obs[32+(2*i)], obs[33+(2*i)]) for i in range(4)]
        return min(station, key=lambda s: abs(s[0]-obs[0]) + abs(s[1]-obs[1]))

def relative_direction(obs, target):
    if target == (obs[0], obs[1]):
        return 8
    facing_angles = [math.pi/2, 0, -math.pi/2, math.pi]
    facing_angle = facing_angles[obs[2]]
    dx = target[0] - obs[0]
    dy = target[1] - obs[1]
    world_angle = math.atan2(-dy, dx)
    rel = (world_angle - facing_angle + math.pi) % (2 * math.pi) - math.pi
    bucket = int(((rel + 3 * math.pi / 8) % (2 * math.pi)) / (math.pi / 4)) % 8
    return bucket

def bucket_distance(obs, target):
    d = abs(obs[0]-target[0]) + abs(obs[1]-target[1])
    if d == 0:  return 0
    if d == 1:  return 1
    if d == 2:  return 2
    if d == 3:  return 3
    if d <= 5:  return 4
    if d <= 8:  return 5
    return 6

def front_summary(obs):
    """
    5-bit passability mask using correct egocentric indices.

    3x3 egocentric layout (forward = up = dy=-1):
        obs[10]=(−1,−1)  obs[11]=(0,−1)  obs[12]=(+1,−1)  ← front row
        obs[13]=(−1, 0)  obs[14]=(0, 0)  obs[15]=(+1, 0)  ← middle (taxi)
        obs[16]=(−1,+1)  obs[17]=(0,+1)  obs[18]=(+1,+1)  ← back row

    Bits (MSB to LSB):
        bit 4: left        (obs[13])
        bit 3: right       (obs[15])
        bit 2: left-front  (obs[10])
        bit 1: front       (obs[11])
        bit 0: right-front (obs[12])

    Impassable: -20 (obstacle/OOB) or -30/-31/-32 (wrong-way highway).
    """
    impassable = (-20, -30, -31, -32)
    left        = int(obs[13] not in impassable)
    right       = int(obs[15] not in impassable)
    left_front  = int(obs[10] not in impassable)
    front       = int(obs[11] not in impassable)
    right_front = int(obs[12] not in impassable)
    return (left << 4) | (right << 3) | (left_front << 2) | (front << 1) | right_front

def fuel_bucket(obs):
    return int(obs[5] >= 100)

def tile_under(obs):
    v = obs[14]
    if v in (20, 21):        return 1  # station with passengers
    if v in (30, 31, 32):    return 2  # valid highway entrance
    if v in (-30, -31, -32): return 3  # wrong-way highway
    if v == 25:              return 4  # gas station
    if -10 <= v <= -1:       return 5  # red light
    return 0

def obs_to_state(obs):
    phase    = get_phase(obs)
    target   = get_target(obs, phase)
    rel_dir  = relative_direction(obs, target)
    dist     = bucket_distance(obs, target)
    carried  = obs[4]
    front    = front_summary(obs)
    fuel     = fuel_bucket(obs)
    pax_left = int(obs[27] > 0)
    under    = tile_under(obs)
    zone     = obs[3]
    return phase, rel_dir, dist, carried, front, fuel, pax_left, under, zone
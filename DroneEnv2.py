"""
DroneEnv.py
────────────────────────────────────────────────────────────────────────────────
Gymnasium-compatible waypoint-navigation environment for ArduCopter via
MAVLink (SITL or real hardware).

IMPORTANT — Adding observations to an already-trained model
────────────────────────────────────────────────────────────────
You CANNOT add new observation slots to a model that has already been
trained. The PPO neural network's input layer size is fixed at model
creation time. If you change the obs size, old weights are incompatible
and you must train from scratch.

This is exactly why obstacle-distance slots (indices 15-22) are included
NOW as fixed placeholder values (999.0 = "nothing detected"). The network
learns to mostly ignore them while they are constant, but the input layer
is already the correct size. When you later connect real rangefinder
sensors, simply replace the placeholder values in get_obstacle_distances()
and fine-tune — no architecture change required.

Observation vector (23 floats):
┌──────┬──────────────────┬───────────────────────────────────────────────┐
│ Idx  │ Name             │ Description                                   │
├──────┼──────────────────┼───────────────────────────────────────────────┤
│  0   │ north            │ NED north position from origin (m)            │
│  1   │ east             │ NED east  position from origin (m)            │
│  2   │ alt              │ altitude above ground, up-positive (m)        │
│  3   │ dist_north       │ target_north - north (m)                      │
│  4   │ dist_east        │ target_east  - east  (m)                      │
│  5   │ dist             │ Euclidean distance to target (m)              │
│  6   │ vx               │ north velocity (m/s)                          │
│  7   │ vy               │ east  velocity (m/s)                          │
│  8   │ vz               │ vertical velocity, up-positive (m/s)          │
│  9   │ roll             │ roll  angle (rad)                             │
│ 10   │ pitch            │ pitch angle (rad)                             │
│ 11   │ yaw              │ yaw   angle (rad)                             │
│ 12   │ heading_error    │ signed angle from yaw to target bearing (rad) │
│ 13   │ battery_pct      │ battery remaining (%)                         │
│ 14   │ ep_mah           │ mAh consumed since episode start              │
│ 15   │ obs_dist_N       │ obstacle distance North   (m) [placeholder]   │
│ 16   │ obs_dist_NE      │ obstacle distance NE      (m) [placeholder]   │
│ 17   │ obs_dist_E       │ obstacle distance East    (m) [placeholder]   │
│ 18   │ obs_dist_SE      │ obstacle distance SE      (m) [placeholder]   │
│ 19   │ obs_dist_S       │ obstacle distance South   (m) [placeholder]   │
│ 20   │ obs_dist_SW      │ obstacle distance SW      (m) [placeholder]   │
│ 21   │ obs_dist_W       │ obstacle distance West    (m) [placeholder]   │
│ 22   │ obs_dist_NW      │ obstacle distance NW      (m) [placeholder]   │
└──────┴──────────────────┴───────────────────────────────────────────────┘

Action vector (2 floats) — horizontal velocity only:
    vx in [-5, 5]  m/s  north
    vy in [-5, 5]  m/s  east
    Altitude is held by ArduCopter's built-in altitude controller.
    vz is OBSERVED (idx 8) but NOT commanded — intentional.

Reward components:
    +150    Target reached         (dist < 5 m)
    -100    Out of bounds          (|north| or |east| > 250 m)
    -100    Crash detected         (alt < 1 m after warm-up step 20)
    - 50    Battery critical       (battery_pct < 5 %)
    -20..0  Timeout                (proximity partial credit up to +20)
    -0.05   Per-step time penalty
    +/-5.0  Distance progress      (approach/retreat, clipped)
    +/-1.5  Heading alignment      (cos of heading_error x 1.5)
    -∗      Battery efficiency     (ep_mah x 0.001)
────────────────────────────────────────────────────────────────────────────────
"""

from pymavlink import mavutil
import time
import gymnasium as gym
import numpy as np


class DroneEnv(gym.Env):

    # ── Named observation indices ─────────────────────────────────────────
    IDX_NORTH       = 0
    IDX_EAST        = 1
    IDX_ALT         = 2
    IDX_DIST_NORTH  = 3
    IDX_DIST_EAST   = 4
    IDX_DIST        = 5
    IDX_VX          = 6
    IDX_VY          = 7
    IDX_VZ          = 8
    IDX_ROLL        = 9
    IDX_PITCH       = 10
    IDX_YAW         = 11
    IDX_HEADING_ERR = 12
    IDX_BATTERY_PCT = 13
    IDX_EP_MAH      = 14
    # Obstacle distance slots — 8 compass directions (N, NE, E, SE, S, SW, W, NW)
    IDX_OBS_N       = 15
    IDX_OBS_NE      = 16
    IDX_OBS_E       = 17
    IDX_OBS_SE      = 18
    IDX_OBS_S       = 19
    IDX_OBS_SW      = 20
    IDX_OBS_W       = 21
    IDX_OBS_NW      = 22

    OBS_SIZE        = 23        # increase here + extend get_obs() for new sensors
    OBS_PLACEHOLDER = 999.0     # sentinel: "no obstacle within sensor range"

    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, DEVICE: str, env_id: str):
        """
        Parameters
        ----------
        DEVICE  : MAVLink connection string, e.g. 'tcp:127.0.0.1:5762'
        env_id  : Human-readable label used in console output
        """
        # ── Episode state ─────────────────────────────────────────────────
        self.pre_dist   = None   # distance from previous step (progress reward)
        self.start_mah  = 0.0   # mAh at episode start (efficiency reward baseline)
        self.ground_alt = 0.0   # ground altitude reference
        self.steps      = 0

        # ── Target position (randomised each episode) ─────────────────────
        self.TARGET_NORTH = None
        self.TARGET_EAST  = None
        self.TARGET_DIST  = None   # initial straight-line distance (timeout bonus)

        # ── Config ─────────────────────────────────────────────────────────
        self.DEVICE      = DEVICE
        self.env_id      = env_id
        self.TAKEOFF_ALT = 10.0   # cruise altitude held by autopilot (m)
        self.MAX_STEPS   = 600    # ~60 s at 10 Hz

        # ── MAVLink connection ──────────────────────────────────────────────
        self.master = mavutil.mavlink_connection(device=self.DEVICE)
        self.master.wait_heartbeat()
        print(f"[{self.env_id}] Connected to MAVLink!")

        # Request all data streams at 20 Hz
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20,  # bump to 20Hz so get_fresh_msg always finds recent data
            1
        )

        # ── Gymnasium spaces ───────────────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_SIZE,),
            dtype=np.float32
        )

        # 2-D horizontal velocity — altitude is managed by the autopilot
        self.action_space = gym.spaces.Box(
            low=np.array([-5.0, -5.0], dtype=np.float32),
            high=np.array([ 5.0,  5.0], dtype=np.float32),
            dtype=np.float32
        )

    # ═════════════════════════════════════════════════════════════════════
    # MAVLink helpers
    # ═════════════════════════════════════════════════════════════════════

    def is_armed(self) -> bool:
        """True when the HEARTBEAT message confirms ARMED state."""

        """
        msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        """

        msg = self.get_fresh_msg(type='HEARTBEAT', timeout=2)
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

        return False


    def get_fresh_msg(self, type, timeout=0.5):  # 0.5s max wait, not 2s
        """This Function get the Latest message, not stale queued ones"""
        msg = None
        while True:
            new_msg = self.master.recv_match(type=type, blocking=False)
            if new_msg is None:
                break
            msg = new_msg
        if msg is None:
            msg = self.master.recv_match(type=type, blocking=True, timeout=timeout)
        return msg

    def get_pos(self):
        """
        Read LOCAL_POSITION_NED.
        Returns (north, east, alt, vx, vy, vz) all in up-positive convention.
        NED z-down is negated so alt > 0 = above ground; vz > 0 = climbing.
        """

        """
        msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=2)
        if msg:
            return msg.x, msg.y, -msg.z, msg.vx, msg.vy, -msg.vz
        """

        msg = self.get_fresh_msg(type='LOCAL_POSITION_NED', timeout=2)
        if msg:
            return msg.x, msg.y, -msg.z, msg.vx, msg.vy, -msg.vz
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def get_attitude(self):
        """Return (roll, pitch, yaw) in radians from the ATTITUDE message."""

        """
        msg = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=2)
        if msg:
            return msg.roll, msg.pitch, msg.yaw
        """

        msg = self.get_fresh_msg(type='ATTITUDE', timeout=2)
        if msg:
            return msg.roll, msg.pitch, msg.yaw

        return 0.0, 0.0, 0.0

    def get_battery(self):
        """
        Return (battery_remaining %, current_consumed mAh).
        Falls back to (100.0, 0.0) on message timeout so training never crashes.
        """

        '''
        msg = self.master.recv_match(type='BATTERY_STATUS', blocking=True, timeout=2)
        if msg:
            return float(msg.battery_remaining), float(msg.current_consumed)
        '''

        msg = self.get_fresh_msg(type='BATTERY_STATUS', timeout=2)
        if msg:
            return float(msg.battery_remaining), float(msg.current_consumed)

        return 100.0, 0.0

    def get_obstacle_distances(self):
        """
        Return obstacle distances (m) for 8 compass directions:
            [N, NE, E, SE, S, SW, W, NW]  →  obs indices 15-22

        ── PLACEHOLDER — replace when sensors are available ──────────────
        All values are OBS_PLACEHOLDER (999.0) = "clear, no obstacle".
        The PPO network will train with these always-constant values and
        will learn to ignore them. That is fine and intentional.

        How to wire up real rangefinders later:
            1. Enable DISTANCE_SENSOR messages in ArduCopter / SITL plugin.
            2. Read them here (one MAVLink msg per sensor orientation).
            3. Map msg.orientation (0=N, 1=NE, ..., 7=NW) to list slots.
            4. Return real distances capped at OBS_PLACEHOLDER.
            5. Add obstacle penalty in get_reward() (hook already in code).

        Skeleton:
            distances = [self.OBS_PLACEHOLDER] * 8
            msg = self.master.recv_match(type='DISTANCE_SENSOR',
                                         blocking=False, timeout=0.05)
            if msg and 0 <= msg.orientation <= 7:
                distances[msg.orientation] = min(
                    msg.current_distance / 100.0,  # cm -> m
                    self.OBS_PLACEHOLDER
                )
            return distances
        """
        return [self.OBS_PLACEHOLDER] * 8   # <-- swap with real reads later

    def set_guided(self):
        """Switch ArduCopter to GUIDED mode (custom_mode = 4)."""
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4   # GUIDED
        )
        time.sleep(1)

    def arm(self, timeout_s: float = 15.0):
        """
        Send ARM command and poll until heartbeat confirms armed.
        Prints a warning after `timeout_s` instead of hanging forever.
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        deadline = time.time() + timeout_s
        while not self.is_armed():
            if time.time() > deadline:
                print(f"[{self.env_id}] WARNING: Arm timeout after {timeout_s}s")
                break
            time.sleep(0.1)

    def force_disarm(self, timeout_s: float = 10.0):
        """
        Force-disarm using MAVLink magic param 21196.
        Prints a warning after `timeout_s` instead of hanging forever.
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0
        )
        deadline = time.time() + timeout_s
        while self.is_armed():
            if time.time() > deadline:
                print(f"[{self.env_id}] WARNING: Disarm timeout after {timeout_s}s")
                break
            time.sleep(0.1)

    def takeoff(self, alt: float):
        """
        Command takeoff to `alt` metres and wait 8 s for SITL to reach it.
        Reports actual altitude reached for monitoring.
        """
        self.ground_alt = 0.1  # SITL always resets z-origin to ground
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, alt
        )

        # Wait until alt is actually reached, not just a fixed sleep
        deadline = time.time() + 20  # 20s max
        while time.time() < deadline:
            actual_alt = self.get_pos()[2]
            if actual_alt >= alt * 0.85:  # 85% of target = close enough
                break
            time.sleep(0.5)

        actual_alt = self.get_pos()[2]
        print(f"[{self.env_id}] Post-takeoff alt: {actual_alt:.2f} m  (target {alt} m)")

    def teleport_home(self):
        """
        Set SIM_ERES_RESET = 1 to trigger the reset_only.lua script.
        Lua teleports the drone to home, resets battery, and disarms.
        We wait 1.5 s so the 50 ms Lua loop finishes before we re-arm.
        """
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'SIM_ERES_RESET',
            float(1),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        time.sleep(1.5)


    def send_velocity(self, vx: float, vy: float):
        """
        Send a horizontal-only velocity setpoint in LOCAL_NED frame.

        vx = north velocity (m/s)
        vy = east  velocity (m/s)
        vz is always 0.0 — altitude held by ArduCopter's own controller.

        Type mask 0b0000_1111_1100_0111: velocity bits active,
        position / acceleration / yaw bits masked off.
        """
        self.master.mav.set_position_target_local_ned_send(
            0,                                   # time_boot_ms (ignored by AP)
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,                  # velocity-only bitmask
            0, 0, 0,                             # pos setpoints  — ignored
            vx, vy, 0,                           # vel: NED (vz=0 = hold alt)
            0, 0, 0,                             # accel setpoints — ignored
            0, 0                                 # yaw / yaw_rate — ignored
        )

    # ═════════════════════════════════════════════════════════════════════
    # Observation builder
    # ═════════════════════════════════════════════════════════════════════

    def get_obs(self) -> np.ndarray:
        """
        Build and return the full (OBS_SIZE = 23,) observation vector.

        Obstacle distance slots [15-22] come from get_obstacle_distances().
        Replace that method's return values when sensors are real —
        no other code changes needed.
        """
        # ── Sensor reads ──────────────────────────────────────────────────
        north, east, alt, vx, vy, vz = self.get_pos()
        roll, pitch, yaw              = self.get_attitude()
        battery_pct, current_consumed = self.get_battery()
        obstacle_dists                = self.get_obstacle_distances()  # 8 values

        # ── Derived navigation quantities ─────────────────────────────────
        dist_north = self.TARGET_NORTH - north
        dist_east  = self.TARGET_EAST  - east
        dist       = float(np.sqrt(dist_north**2 + dist_east**2))

        # Heading error: signed angle from current yaw to target bearing
        # Normalised to [-pi, pi]:  >0 = turn right,  <0 = turn left
        target_bearing = np.arctan2(dist_east, dist_north)
        heading_error  = float((target_bearing - yaw + np.pi) % (2 * np.pi) - np.pi)

        # Energy consumed this episode (delta from post-takeoff snapshot)
        ep_mah = float(current_consumed - self.start_mah)

        return np.array([
            # Position                        [0-2]
            north, east, alt,
            # Relative to target             [3-5]
            dist_north, dist_east, dist,
            # Velocity                        [6-8]
            vx, vy, vz,
            # Attitude                        [9-11]
            roll, pitch, yaw,
            # Derived                         [12-14]
            heading_error,
            battery_pct,
            ep_mah,
            # Obstacle distances (8 dirs)     [15-22]
            *obstacle_dists,
        ], dtype=np.float32)

    # ═════════════════════════════════════════════════════════════════════
    # Reward
    # ═════════════════════════════════════════════════════════════════════

    def get_reward(self, obs: np.ndarray):
        """
        Compute (reward: float, done: bool) for the current timestep.

        Obstacle slots obs[15:23] are present but NOT used in reward here.
        When real sensor values arrive, uncomment the hook at the bottom of
        the shaping section and tune the penalty weight.
        """
        # ── Unpack via named indices (no magic numbers) ───────────────────
        north         = obs[self.IDX_NORTH]
        east          = obs[self.IDX_EAST]
        alt           = obs[self.IDX_ALT]
        dist          = obs[self.IDX_DIST]
        heading_error = obs[self.IDX_HEADING_ERR]
        battery_pct   = obs[self.IDX_BATTERY_PCT]
        ep_mah        = obs[self.IDX_EP_MAH]
        # obstacle distances available as: obs[self.IDX_OBS_N : self.IDX_OBS_NW + 1]

        reward = 0.0

        # ═════════════════════════════════════════════════════════════════
        # TERMINATION CONDITIONS  (first match wins, checked in priority order)
        # ═════════════════════════════════════════════════════════════════

        # 1. Target reached
        if dist < 5.0:
            reward += 150.0
            return reward, True

        # 2. Arena boundary exceeded
        if abs(north) > 250.0 or abs(east) > 250.0:
            reward -= 100.0
            return reward, True

        # 3. Crash — skip first 20 steps (drone is still climbing after takeoff)
        if alt < 1.0 and self.steps > 20:
            reward -= 100.0
            return reward, True

        # 4. Battery dead
        if battery_pct < 5.0:
            reward -= 50.0
            return reward, True

        # 5. Episode timeout — proximity partial credit (0 to +20)
        if self.steps >= self.MAX_STEPS:
            proximity_ratio = max(0.0, 1.0 - dist / max(self.TARGET_DIST, 1.0))
            if proximity_ratio > 0.8:
                reward += 5.0  # got within 20% of target distance, acknowledge it
            reward -= (20.0 - proximity_ratio * 20.0)
            return reward, True

        # ═════════════════════════════════════════════════════════════════
        # STEP SHAPING  (non-terminal)
        # ═════════════════════════════════════════════════════════════════

        # 1. Time penalty — constant pressure to reach the target quickly
        reward -= 0.05

        # 2. Distance progress — rewards moving closer, penalises drifting away
        if self.pre_dist is not None:
            dist_change = self.pre_dist - dist          # positive = closer
            reward += float(np.clip(dist_change * 5.0, -5.0, 5.0))
        self.pre_dist = dist

        # 3. Heading alignment — facing the target is rewarded
        #    cos(0) = 1.0 (perfect),  cos(pi) = -1.0 (facing away)
        reward += float(np.cos(heading_error)) * 0.5

        # 4. Battery efficiency — penalise accumulated energy waste per step
        reward -= ep_mah * 0.001

        # ── Obstacle avoidance hook (disabled — enable when sensors are real) ──
        # nearest = float(np.min(obs[self.IDX_OBS_N : self.IDX_OBS_NW + 1]))
        # if nearest < 10.0:                              # obstacle within 10 m
        #     reward -= (10.0 - nearest) * 1.0           # penalty scales with proximity

        return reward, False

    # ═════════════════════════════════════════════════════════════════════
    # Gymnasium interface
    # ═════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, **kwargs):
        """
        Start a fresh episode:
          1. Teleport drone to home via Lua trigger.
          2. Randomise target position (>= 20 m from origin).
          3. Switch to GUIDED, arm, and takeoff.
          4. Snapshot mAh for per-episode energy tracking.
        Returns (obs, {}).
        """
        self.steps    = 0
        self.pre_dist = None

        # ── Reset drone pose ──────────────────────────────────────────────
        self.teleport_home()

        # ── Randomise target ──────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        while True:
            self.TARGET_NORTH = int(rng.integers(-100, 101))
            self.TARGET_EAST  = int(rng.integers(-100, 101))
            # Reject targets inside the 20 m dead-zone around origin
            if abs(self.TARGET_NORTH) >= 20 or abs(self.TARGET_EAST) >= 20:
                break

        self.TARGET_DIST = float(np.sqrt(self.TARGET_NORTH**2 + self.TARGET_EAST**2))
        print(
            f"[{self.env_id}] Target: "
            f"N={self.TARGET_NORTH:+4d} m  "
            f"E={self.TARGET_EAST:+4d} m  "
            f"Dist={self.TARGET_DIST:.1f} m"
        )

        # ── Arm and takeoff ───────────────────────────────────────────────
        self.set_guided()
        self.arm()
        time.sleep(0.2)
        self.takeoff(self.TAKEOFF_ALT)

        # Snapshot energy so ep_mah starts near 0 in the first observation
        _, self.start_mah = self.get_battery()

        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        """
        Execute one control step.

        Parameters
        ----------
        action : np.ndarray shape (2,)
            [vx, vy]  horizontal velocity command in m/s

        Returns
        -------
        obs    : np.ndarray (23,)
        reward : float
        done   : bool
        False  : bool  — truncated (always False; SB3 compatibility)
        {}     : dict  — empty info
        """
        self.steps += 1
        step_start = time.time()

        # Defensive clamp — SB3 clips too, but be safe
        vx = float(np.clip(action[0], -5.0, 5.0))
        vy = float(np.clip(action[1], -5.0, 5.0))

        self.send_velocity(vx, vy)

        obs          = self.get_obs()
        reward, done = self.get_reward(obs)

        if done:
            self.force_disarm()

        # Enforce ~10Hz step rate
        elapsed = time.time() - step_start
        sleep_time = 0.1 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        return obs, reward, done, False, {}

    def close(self):
        """Disarm and close the MAVLink connection cleanly."""
        try:
            self.force_disarm()
        except Exception as e:
            print(f"[{self.env_id}] close() disarm error (ignored): {e}")
        self.master.close()
        print(f"[{self.env_id}] Environment closed.")


# ═════════════════════════════════════════════════════════════════════════
# Smoke test — run this file directly to verify connectivity
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    env = DroneEnv(DEVICE='tcp:127.0.0.1:5762', env_id='Test')

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}\n")

    obs, _ = env.reset()
    print(f"Initial obs ({len(obs)} values):\n{obs}\n")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {i+1:2d} | reward: {reward:+.4f} | done: {done}")
        if done:
            break

    env.close()
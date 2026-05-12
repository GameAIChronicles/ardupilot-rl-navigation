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
    vx in [-3, 3]  m/s  north   (reduced from ±5 for finer control)
    vy in [-3, 3]  m/s  east
    Altitude is held by ArduCopter's built-in altitude controller.
    vz is OBSERVED (idx 8) but NOT commanded — intentional.

Action repeat: ACTION_REPEAT = 4
    Same velocity command sent 4 times per step with 50ms sleep between
    each repeat. Rewards accumulated across repeats. This aligns the
    action→effect timing with MAVLink latency and autopilot response.

Reward components (scaled ÷10 for value function stability):
    +80       Target reached          (dist < 5 m)
    -20       Out of bounds           (|north| or |east| > 250 m)
    -20       Crash detected          (alt < 1 m after warm-up step 20)
    -15       Battery critical        (battery_pct < 5 %)
    -10..+5   Timeout                 (proximity partial credit)
    - 0.1     Per-step time penalty
    +/-0.5    Distance progress       (clipped ±0.5)

Curriculum stages (auto-advanced by TrainSaveCallback):
    1: ±20m  targets, min_dist=5,  MAX_STEPS=200
    2: ±35m  targets, min_dist=8,  MAX_STEPS=300
    3: ±50m  targets, min_dist=10, MAX_STEPS=400
    4: ±70m  targets, min_dist=15, MAX_STEPS=500
    5: ±100m targets, min_dist=20, MAX_STEPS=600
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
    IDX_OBS_N       = 15
    IDX_OBS_NE      = 16
    IDX_OBS_E       = 17
    IDX_OBS_SE      = 18
    IDX_OBS_S       = 19
    IDX_OBS_SW      = 20
    IDX_OBS_W       = 21
    IDX_OBS_NW      = 22

    OBS_SIZE        = 23
    OBS_PLACEHOLDER = 999.0
    ACTION_REPEAT   = 4     # same action applied N times per step

    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, DEVICE: str, env_id: str):
        """
        Parameters
        ----------
        DEVICE  : MAVLink connection string, e.g. 'tcp:127.0.0.1:5762'
        env_id  : Human-readable label used in console output
        """
        # ── Episode state ─────────────────────────────────────────────────
        self.curriculum_stage = 5
        self.pre_dist         = None
        self.start_mah        = 0.0
        self.ground_alt       = 0.0
        self.steps            = 0
        self.ep_reward        = 0.0   # cumulative reward for debug print in reset()

        # ── Target position ───────────────────────────────────────────────
        self.TARGET_NORTH = None
        self.TARGET_EAST  = None
        self.TARGET_DIST  = None

        # ── Config ────────────────────────────────────────────────────────
        self.DEVICE      = DEVICE
        self.env_id      = env_id
        self.TAKEOFF_ALT = 10.0
        self.MAX_STEPS   = 200    # overridden by curriculum stage each episode

        # ── MAVLink connection ─────────────────────────────────────────────
        self.master = mavutil.mavlink_connection(device=self.DEVICE)
        self.master.wait_heartbeat()
        print(f"[{self.env_id}] Connected to MAVLink!")

        # Request all data streams at 20 Hz
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20, 1
        )
        #─────────────────────────────────────────────────────────────────────
        # Disables internal Dataflash logging to save disk space
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'LOG_BITMASK',
            0,
            1  # MAV_PARAM_TYPE_REAL32
        )

        # ── Gymnasium spaces ───────────────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_SIZE,),
            dtype=np.float32
        )

        # Action range ±3 m/s — reduced from ±5 for finer control on close targets
        self.action_space = gym.spaces.Box(
            low=np.array([-3.0, -3.0], dtype=np.float32),
            high=np.array([ 3.0,  3.0], dtype=np.float32),
            dtype=np.float32
        )

    # ═════════════════════════════════════════════════════════════════════
    # MAVLink helpers
    # ═════════════════════════════════════════════════════════════════════

    def is_armed(self) -> bool:
        """
        Return True when the latest HEARTBEAT confirms ARMED state.
        Uses get_fresh_msg() to avoid stale queued heartbeats.
        Returns False on message timeout.
        """
        msg = self.get_fresh_msg(type='HEARTBEAT', timeout=2)
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        return False

    def get_fresh_msg(self, type, timeout=0.5):
        """
        Drain the MAVLink receive queue and return the most recent message
        of the given type, discarding any stale queued messages.

        If the queue is empty, falls back to a single blocking recv with
        `timeout` seconds. Returns None if no message arrives in time.

        Parameters
        ----------
        type    : str   MAVLink message type, e.g. 'LOCAL_POSITION_NED'
        timeout : float Max seconds to wait if queue is empty (default 0.5)

        Why this exists:
            pymavlink buffers incoming messages internally. A naive
            recv_match(blocking=True) returns the oldest queued message,
            which may be several steps stale at 20 Hz stream rates.
            This method guarantees the observation always reflects the
            drone's current state, not a buffered snapshot.
        """
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
        Read the latest LOCAL_POSITION_NED message via get_fresh_msg().

        Returns (north, east, alt, vx, vy, vz) in up-positive convention:
            north, east : horizontal position from NED origin (m)
            alt         : altitude above ground, up-positive (m)
                          NED z-down is negated: alt = -msg.z
            vx, vy      : horizontal velocity (m/s), NED convention
            vz          : vertical velocity, up-positive (m/s) = -msg.vz

        Falls back to (0, 0, 0, 0, 0, 0) on timeout — training continues
        but a zero-alt fallback may trigger the crash check after step 20.
        """
        msg = self.get_fresh_msg(type='LOCAL_POSITION_NED', timeout=2)
        if msg:
            return msg.x, msg.y, -msg.z, msg.vx, msg.vy, -msg.vz
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def get_attitude(self):
        """
        Return (roll, pitch, yaw) in radians from the latest ATTITUDE message.
        Uses get_fresh_msg() to avoid stale queued data.
        Falls back to (0.0, 0.0, 0.0) on timeout.
        """
        msg = self.get_fresh_msg(type='ATTITUDE', timeout=2)
        if msg:
            return msg.roll, msg.pitch, msg.yaw
        return 0.0, 0.0, 0.0

    def get_battery(self):
        """
        Return (battery_remaining %, current_consumed mAh) from the latest
        BATTERY_STATUS message via get_fresh_msg().

        Falls back to (100.0, 0.0) on message timeout so training never
        crashes — but a 100% fallback means the battery-dead termination
        condition will never fire during a timeout, which is intentional.
        """
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
        """
        return [self.OBS_PLACEHOLDER] * 8

    def set_guided(self):
        """Switch ArduCopter to GUIDED mode (custom_mode = 4)."""
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4
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
        Command takeoff to `alt` metres and wait until altitude is reached.

        Polls LOCAL_POSITION_NED every 0.5 s until alt >= 85% of target,
        with a 20 s hard deadline. Reports actual altitude on completion.

        Note: ground_alt is hardcoded to 0.1 m because SITL always resets
        the NED z-origin to ground level after teleport_home().
        """
        self.ground_alt = 0.1
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, alt
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            actual_alt = self.get_pos()[2]
            if actual_alt >= alt * 0.85:
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
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, 0,
            0, 0, 0,
            0, 0
        )

    # ═════════════════════════════════════════════════════════════════════
    # Observation builder
    # ═════════════════════════════════════════════════════════════════════

    def get_obs(self) -> np.ndarray:
        """
        Build and return the full (OBS_SIZE = 23,) observation vector.

        All four MAVLink reads (position, attitude, battery, obstacles) use
        get_fresh_msg() to guarantee fresh data, not stale buffered packets.

        Obstacle distance slots [15-22] come from get_obstacle_distances().
        Replace that method's return values when sensors are real —
        no other code changes needed here.

        Note: heading_error is normalised to [-pi, pi].
              ep_mah is delta from the post-takeoff snapshot in reset(),
              so it starts near 0 at the beginning of each episode.
        """
        north, east, alt, vx, vy, vz = self.get_pos()
        roll, pitch, yaw              = self.get_attitude()
        battery_pct, current_consumed = self.get_battery()
        obstacle_dists                = self.get_obstacle_distances()

        dist_north = self.TARGET_NORTH - north
        dist_east  = self.TARGET_EAST  - east
        dist       = float(np.sqrt(dist_north**2 + dist_east**2))

        target_bearing = np.arctan2(dist_east, dist_north)
        heading_error  = float((target_bearing - yaw + np.pi) % (2 * np.pi) - np.pi)

        ep_mah = float(current_consumed - self.start_mah)

        return np.array([
            north, east, alt,
            dist_north, dist_east, dist,
            vx, vy, vz,
            roll, pitch, yaw,
            heading_error,
            battery_pct,
            ep_mah,
            *obstacle_dists,
        ], dtype=np.float32)

    # ═════════════════════════════════════════════════════════════════════
    # Reward
    # ═════════════════════════════════════════════════════════════════════

    def get_reward(self, obs: np.ndarray):
        """
        Fine-tuning reward function — extended from curriculum base.

        Changes from curriculum version:
        - Target reached threshold tightened to 3m (was 5m)
        - Heading alignment added (cos(heading_error) × 0.3)
        - Distance progress scaled by TARGET_DIST for large targets
        - Absolute distance penalty added (dist × 0.001)
        - Timeout uses proximity ratio partial credit (0 to +20)
        - Boundary/crash penalties increased to -30 (was -20)
        """
        north = obs[self.IDX_NORTH]
        east = obs[self.IDX_EAST]
        alt = obs[self.IDX_ALT]
        dist = obs[self.IDX_DIST]
        heading_error = obs[self.IDX_HEADING_ERR]  # already normalised [-pi, pi]
        battery_pct = obs[self.IDX_BATTERY_PCT]

        reward = 0.0

        # ═════════════════════════════════════════════════════════════════
        # TERMINATION CONDITIONS
        # ═════════════════════════════════════════════════════════════════

        # 1. Target reached — tighter threshold for fine-tuning
        if dist < 5.0:
            reward += 100.0
            print(f'[{self.env_id}] Target reached!')
            return reward, True

        # 2. Out of bounds — use absolute position, not relative to target
        if abs(north) > 250.0 or abs(east) > 250.0:
            reward -= 30.0
            return reward, True

        # 3. Crash
        if alt < 1.0 and self.steps > 20:
            reward -= 30.0
            return reward, True

        # 4. Battery dead
        if battery_pct < 5.0:
            reward -= 20.0
            return reward, True

        # 5. Timeout — proximity partial credit
        if self.steps >= self.MAX_STEPS:
            proximity_ratio = max(0.0, 1.0 - dist / max(self.TARGET_DIST, 1.0))
            reward += proximity_ratio * 20.0
            reward -= 10.0
            return reward, True

        # ═════════════════════════════════════════════════════════════════
        # STEP SHAPING
        # ═════════════════════════════════════════════════════════════════

        # 1. Time penalty
        reward -= 1*(1.1**(self.steps/30))


        # 2. Distance progress — scaled for large targets
        if self.pre_dist is not None:
            dist_change = self.pre_dist - dist
            scale = np.clip(self.TARGET_DIST / 50.0, 1.0, 3.0)
            reward += float(np.clip(dist_change * scale, -1.0, 1.0))
        self.pre_dist = dist

        # 3. Velocity direction alignment — replaces pure heading reward
        vx = obs[self.IDX_VX]
        vy = obs[self.IDX_VY]
        speed = float(np.sqrt(vx ** 2 + vy ** 2))
        if speed > 0.1:
            dist_north = obs[self.IDX_DIST_NORTH]
            dist_east = obs[self.IDX_DIST_EAST]
            target_dir_n = dist_north / max(dist, 1.0)
            target_dir_e = dist_east / max(dist, 1.0)
            vel_dir_n = vx / speed
            vel_dir_e = vy / speed
            alignment = vel_dir_n * target_dir_n + vel_dir_e * target_dir_e
            dist_factor = np.clip(dist / 15.0, 0.0, 1.0)
            reward += alignment  * dist_factor
        '''
        # 4. Decelerate near target — reduces oscillation and overshooting
        if dist < 10.0:
            reward -= speed * 0.05
        '''

        # 5. Absolute distance penalty
        reward -= dist * 0.01

        return reward, False

    # ═════════════════════════════════════════════════════════════════════
    # Gymnasium interface
    # ═════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, **kwargs):
        """
        Start a fresh episode:
          1. Print cumulative episode reward for debugging.
          2. Teleport drone to home via Lua SIM_ERES_RESET trigger.
          3. Randomise target position based on curriculum stage.
          4. Switch to GUIDED mode, arm, and takeoff to TAKEOFF_ALT.
          5. Snapshot mAh for per-episode energy tracking.

        seed=None (SB3 default) produces different targets every episode.
        A fixed seed will always produce the same target — useful for eval.

        Returns (obs, {}).
        """
        print(f'[{self.env_id}] Episode steps: {self.steps}')
        print(f'[{self.env_id}] Episode reward: {self.ep_reward:.2f}')
        self.ep_reward = 0.0
        self.steps     = 0
        self.pre_dist  = None

        self.teleport_home()

        stage_config = {
            1: {'low': -20,  'high': 21,  'min_dist': 5,  'max_steps': 200},
            2: {'low': -35,  'high': 36,  'min_dist': 8,  'max_steps': 300},
            3: {'low': -50,  'high': 51,  'min_dist': 10, 'max_steps': 400},
            4: {'low': -70,  'high': 71,  'min_dist': 15, 'max_steps': 500},
            5: {'low': -100, 'high': 101, 'min_dist': 20, 'max_steps': 600},
        }

        rng = np.random.default_rng(seed)
        cfg = stage_config[self.curriculum_stage]
        self.MAX_STEPS = cfg['max_steps']

        while True:
            self.TARGET_NORTH = int(rng.integers(cfg['low'], cfg['high']))
            self.TARGET_EAST  = int(rng.integers(cfg['low'], cfg['high']))
            if abs(self.TARGET_NORTH) >= cfg['min_dist'] or \
               abs(self.TARGET_EAST)  >= cfg['min_dist']:
                break

        self.TARGET_DIST = float(np.sqrt(self.TARGET_NORTH**2 + self.TARGET_EAST**2))
        print(
            f"[{self.env_id}] Stage {self.curriculum_stage} | "
            f"Target: N={self.TARGET_NORTH:+4d} m  "
            f"E={self.TARGET_EAST:+4d} m  "
            f"Dist={self.TARGET_DIST:.1f} m"
        )

        self.set_guided()
        self.arm()
        time.sleep(0.2)
        self.takeoff(self.TAKEOFF_ALT)

        _, self.start_mah = self.get_battery()

        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        """
        Execute one control step with ACTION_REPEAT repetitions.

        The same velocity command is sent ACTION_REPEAT times with a 50ms
        sleep between each, allowing autopilot physics to settle before the
        next observation is read. Rewards are accumulated across repeats.

        Parameters
        ----------
        action : np.ndarray shape (2,)
            [vx, vy]  horizontal velocity command in m/s, clipped to ±3

        Returns
        -------
        obs    : np.ndarray (23,)
        reward : float   (sum across ACTION_REPEAT repeats)
        done   : bool
        False  : bool  — truncated (always False; SB3 compatibility)
        {}     : dict  — empty info
        """
        self.steps += 1
        step_start = time.time()

        # Clamp to action space bounds
        vx = float(np.clip(action[0], -3.0, 3.0))
        vy = float(np.clip(action[1], -3.0, 3.0))

        total_reward = 0.0
        done         = False
        obs          = None

        for _ in range(self.ACTION_REPEAT):
            self.send_velocity(vx, vy)
            time.sleep(0.05)
            obs          = self.get_obs()
            reward, done = self.get_reward(obs)
            total_reward += reward
            if done:
                break

        # Accumulate for episode debug print in reset()
        self.ep_reward += total_reward

        if done:
            self.force_disarm()

        elapsed    = time.time() - step_start
        sleep_time = 0.1 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        return obs, total_reward, done, False, {}

    def close(self):
        """
        Disarm the drone and close the MAVLink connection cleanly.
        Call this when training ends or on KeyboardInterrupt to avoid
        leaving the SITL drone armed and flying with no controller.
        """
        try:
            self.force_disarm()
        except Exception as e:
            print(f"[{self.env_id}] close() disarm error (ignored): {e}")
        self.master.close()
        print(f"[{self.env_id}] Environment closed.")


# ═════════════════════════════════════════════════════════════════════════
# Smoke test
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
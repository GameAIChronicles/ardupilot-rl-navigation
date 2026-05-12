"""
DroneWaypointTrain_Continue.py
────────────────────────────────────────────────────────────────────────────────
Resume PPO training from the latest checkpoint.

How it works:
1. Scans Data/models/ArbitraryPoint/ for model_PPO_ITER_<N>.zip files.
2. Picks the highest N as the resume point.
3. Loads matching vec_normalize_ITER_<N>.pkl if it exists.
4. Continues training with tighter hyperparameters for fine-tuning.
5. Saves model + VecNormalize stats at every checkpoint.

Curriculum stage is preserved from the checkpoint — do NOT override it.
The stage advances automatically via TrainSaveCallback thresholds.

Logs go to a new TensorBoard sub-run (PPO_Continue_<timestamp>).
────────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import glob
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from DroneEnv import DroneEnv


# ════════════════════════════════════════════════════════════════════════
# Helper — find latest checkpoint
# ════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(model_dir: str):
    """
    Scan model_dir for model_PPO_ITER_<N>.zip files.
    Returns (path_without_extension, iteration_number) for highest N.
    Raises FileNotFoundError if none found.
    """
    pattern = os.path.join(model_dir, "model_PPO_ITER_*.zip")
    files   = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No checkpoint files in '{model_dir}'.\n"
            "Run DroneWayPointTraining.py first."
        )

    def extract_iter(path):
        match = re.search(r"model_PPO_ITER_(\d+)\.zip$", path)
        return int(match.group(1)) if match else -1

    files.sort(key=extract_iter)
    latest      = files[-1]
    latest_iter = extract_iter(latest)

    return latest[:-4], latest_iter   # path without .zip, iter number


# ════════════════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════════════════

save_dir = os.path.join("Data", "models", "ArbitraryPoint")
log_dir  = os.path.join("Data", "logs",   "ArbitraryPoint")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# Find checkpoint
# ════════════════════════════════════════════════════════════════════════

checkpoint_path, resume_iter = find_latest_checkpoint(save_dir)
print(f"\n  Found checkpoint: {checkpoint_path}.zip  (iter {resume_iter})")


# ════════════════════════════════════════════════════════════════════════
# Environment — same stack as fresh training
# ════════════════════════════════════════════════════════════════════════

def make_env():
    return Monitor(DroneEnv(
        DEVICE='tcp:127.0.0.1:5762',
        env_id='Instance0'
    ))

env = DummyVecEnv([make_env])

# Load VecNormalize stats if available — preserves running mean/std
vec_path = os.path.join(save_dir, f"vec_normalize_ITER_{resume_iter}.pkl")

if os.path.exists(vec_path):
    print(f"  Loading VecNormalize stats: {vec_path}")
    env = VecNormalize.load(vec_path, env)
    env.training    = True    # keep updating stats during continued training
    env.norm_reward = True
    env.clip_obs    = 10.0
else:
    print("  WARNING: VecNormalize stats not found — creating fresh normalizer")
    print("  Training may be slightly unstable for first few iterations")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# ════════════════════════════════════════════════════════════════════════
# Training configuration
# ════════════════════════════════════════════════════════════════════════

Iteration_Step       =  5
Steps_Per_Iter       = 2048
Extra_Loops          = 10    # 10 × 5 = 50 more iterations

additional_timesteps = Extra_Loops * Iteration_Step * Steps_Per_Iter
save_freq            = Iteration_Step * Steps_Per_Iter

tb_run_name = f"PPO_Continue_{int(time.time())}"


# ════════════════════════════════════════════════════════════════════════
# Callback — saves model + VecNormalize at every checkpoint
# ════════════════════════════════════════════════════════════════════════

class TrainSaveCallback(BaseCallback):
    """
    Saves model and VecNormalize stats every `save_freq` steps.
    Also advances curriculum stage when performance thresholds are met.

    Curriculum thresholds (checked against raw Monitor rewards):
        Stage 1 → 2: avg_rew > 5,  avg_ep_len < 150
        Stage 2 → 3: avg_rew > 4,  avg_ep_len < 250
        Stage 3 → 4: avg_rew > 3,  avg_ep_len < 350
        Stage 4 → 5: avg_rew > 2,  avg_ep_len < 450
    """

    STAGE_THRESHOLDS = {
        1: {'min_rew': 5, 'max_ep_len': 150},
        2: {'min_rew': 4, 'max_ep_len': 250},
        3: {'min_rew': 3, 'max_ep_len': 350},
        4: {'min_rew': 2, 'max_ep_len': 450},
    }

    def __init__(self, save_freq: int, save_path: str,
                 start_iter: int, iter_step: int):
        super().__init__()
        self.save_freq    = save_freq
        self.save_path    = save_path
        self.current_iter = start_iter
        self.iter_step    = iter_step

        self.recent_rewards = []
        self.recent_ep_lens = []
        self.CHECK_FREQ     = 5000
        self.WINDOW         = 20

        os.makedirs(save_path, exist_ok=True)

    def _get_inner_env(self):
        """Unwrap VecNormalize → DummyVecEnv → Monitor → DroneEnv."""
        monitor_env = self.training_env.envs[0]   # type: ignore
        return monitor_env.env                      # DroneEnv

    def _on_step(self) -> bool:

        # ── Collect episode stats ─────────────────────────────────────
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.recent_rewards.append(info['episode']['r'])
                self.recent_ep_lens.append(info['episode']['l'])

        # ── Curriculum check ──────────────────────────────────────────
        if self.n_calls % self.CHECK_FREQ == 0 \
                and len(self.recent_rewards) >= self.WINDOW:

            avg_rew = float(np.mean(self.recent_rewards[-self.WINDOW:]))
            avg_len = float(np.mean(self.recent_ep_lens[-self.WINDOW:]))
            inner   = self._get_inner_env()
            stage   = inner.curriculum_stage

            print(
                f"  [Curriculum] Stage {stage} | "
                f"avg_rew={avg_rew:.1f} | avg_ep_len={avg_len:.1f}"
            )

            if stage in self.STAGE_THRESHOLDS:
                thresh = self.STAGE_THRESHOLDS[stage]
                if avg_rew > thresh['min_rew'] and avg_len < thresh['max_ep_len']:
                    inner.curriculum_stage += 1
                    print(
                        f"\n  🎓 Advancing to Stage {inner.curriculum_stage}! "
                        f"(avg_rew={avg_rew:.1f}, avg_ep_len={avg_len:.1f})\n"
                    )
                    self.recent_rewards = []
                    self.recent_ep_lens = []

                    name = f"model_PPO_ITER_{self.current_iter}_stage{inner.curriculum_stage}"
                    self.model.save(os.path.join(self.save_path, name))
                    self.training_env.save(
                        os.path.join(self.save_path, f"vec_normalize_{name}.pkl")
                    )
                    print(f"  ✓ Stage transition saved: {name}\n")

        # ── Regular checkpoint ────────────────────────────────────────
        if self.n_calls % self.save_freq == 0:
            self.current_iter += self.iter_step

            name = f"model_PPO_ITER_{self.current_iter}"
            self.model.save(os.path.join(self.save_path, name))
            self.training_env.save(
                os.path.join(self.save_path, f"vec_normalize_ITER_{self.current_iter}.pkl")
            )

            print(
                f"  ✓ [{self.num_timesteps:>8d} steps]  "
                f"Saved: {name}.zip + vec_normalize_ITER_{self.current_iter}.pkl"
            )

        return True


# ════════════════════════════════════════════════════════════════════════
# Load model with fine-tuning hyperparameters
# ════════════════════════════════════════════════════════════════════════

print("  Loading model...")

model = PPO.load(
    checkpoint_path,
    env             = env,
    tensorboard_log = log_dir,
    learning_rate   = 5e-5,    # reduced from 3e-4 for stable fine-tuning
    ent_coef        = 0.0001,    # keep exploration alive
    target_kl       = 0.01,    # prevent large policy updates
    n_epochs        = 10,
    clip_range      = 0.2,
)

print(f"\n{'─'*60}")
print(f"  Continuing from iter {resume_iter}")
print(f"  Training {Extra_Loops * Iteration_Step} more iters ({additional_timesteps:,} steps)")
print(f"  Final iter: {resume_iter + Extra_Loops * Iteration_Step}")
print(f"  Checkpoint every {save_freq:,} steps")
print(f"  TensorBoard run: {tb_run_name}")
print(f"  TensorBoard: tensorboard --logdir {log_dir}")
print(f"{'─'*60}\n")


# ════════════════════════════════════════════════════════════════════════
# Train
# ════════════════════════════════════════════════════════════════════════

callback = TrainSaveCallback(
    save_freq  = save_freq,
    save_path  = save_dir,
    start_iter = resume_iter,
    iter_step  = Iteration_Step,
)

model.learn(
    total_timesteps     = additional_timesteps,
    callback            = callback,
    tb_log_name         = tb_run_name,
    reset_num_timesteps = False,   # continue TensorBoard x-axis
    log_interval        = 1,
)

# Save final state
model.save(os.path.join(save_dir, "model_PPO_CONTINUE_FINAL"))
env.save(os.path.join(save_dir, "vec_normalize_CONTINUE_FINAL.pkl"))

print(f"\n✅ Continue training complete. Reached iter {resume_iter + Extra_Loops * Iteration_Step}.")
env.close()
"""
DroneWaypointTrain_Continue.py
────────────────────────────────────────────────────────────────────────────────
Resume PPO training from the LATEST saved checkpoint in the models directory.

How it works
────────────
1. Scans Data/models/ArbitraryPoint/ for all  model_PPO_ITER_<N>.zip  files.
2. Picks the file with the highest N as the resume point.
3. Loads that model into a NEW PPO instance (env + unchanged hyperparams).
4. Continues training for `Extra_Loops × Iteration_Step` more iterations,
   saving checkpoints with labels that continue from N upward.

Usage
─────
  # After a fresh run that reached iter 200:
  python DroneWaypointTrain_Continue.py
  # → scans dir, finds ITER_200, continues to ITER_300 (if Extra_Loops=20)

  # To add even more iters, just run it again — it will find the new latest.

Logs go to a NEW TensorBoard sub-run (PPO_Continue_<timestamp>) so graphs
stay separate from the fresh run but live in the same log directory.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import glob
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from DroneEnv import DroneEnv


# ════════════════════════════════════════════════════════════════════════
# Helper — find the latest checkpoint
# ════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(model_dir: str):
    """
    Scan `model_dir` for files matching  model_PPO_ITER_<N>.zip
    and return (path_without_extension, iteration_number) for the
    highest N found.  Raises FileNotFoundError if none exist.
    """
    pattern = os.path.join(model_dir, "model_PPO_ITER_*.zip")
    files   = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No checkpoint files found in '{model_dir}'.\n"
            "Run DroneWaypointTrain_Fresh.py first."
        )

    # Extract the iteration number from each filename and sort
    def extract_iter(path):
        match = re.search(r"model_PPO_ITER_(\d+)\.zip$", path)
        return int(match.group(1)) if match else -1

    files.sort(key=extract_iter)
    latest_path = files[-1]
    latest_iter = extract_iter(latest_path)

    # SB3 load() wants the path WITHOUT .zip
    path_no_ext = latest_path[:-4]
    return path_no_ext, latest_iter


# ════════════════════════════════════════════════════════════════════════
# Environment
# ════════════════════════════════════════════════════════════════════════

env = DroneEnv(
    DEVICE='tcp:127.0.0.1:5762',
    env_id='Instance0'
)


# ════════════════════════════════════════════════════════════════════════
# Custom Callback — same logic as the fresh script
# ════════════════════════════════════════════════════════════════════════

class TrainSaveCallback(BaseCallback):
    """
    Saves model checkpoints every `save_freq` steps.
    Iteration labels continue from `start_iter` upward.
    """

    def __init__(self, save_freq: int, save_path: str,
                 start_iter: int, iter_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq    = save_freq
        self.save_path    = save_path
        self.current_iter = start_iter
        self.iter_step    = iter_step

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.current_iter += self.iter_step

            save_name = f"model_PPO_ITER_{self.current_iter}"
            save_file = os.path.join(self.save_path, save_name)
            self.model.save(save_file)

            print(
                f"  ✓ [{self.num_timesteps:>8d} steps]  "
                f"Saved: {save_name}.zip"
            )

        return True


# ════════════════════════════════════════════════════════════════════════
# Continue-training configuration
# ════════════════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────────────────
save_dir = os.path.join("Data", "models", "ArbitraryPoint")
log_dir  = os.path.join("Data", "logs",   "ArbitraryPoint")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)

# ── Locate the latest checkpoint ─────────────────────────────────────────
checkpoint_path, resume_iter = find_latest_checkpoint(save_dir)
print(f"\n  Found checkpoint: {checkpoint_path}.zip  (iter {resume_iter})")

# ── How many MORE iterations to train ────────────────────────────────────
Iteration_Step  =  5    # must match the fresh-run setting
Steps_Per_Iter  = 2048  # must match model n_steps
Extra_Loops     = 20    # 20 loops × 5 iters = 100 more iters
                        # → if you resumed at 200, you'll reach 300

additional_timesteps = Extra_Loops * Iteration_Step * Steps_Per_Iter
# = 20 × 5 × 2 048 = 204 800 extra steps

save_freq = Iteration_Step * Steps_Per_Iter
# checkpoints every 10 240 steps, same as fresh run

# ── TensorBoard sub-run name includes timestamp so runs don't collide ────
tb_run_name = f"PPO_Continue_{int(time.time())}"


# ════════════════════════════════════════════════════════════════════════
# Load model and continue
# ════════════════════════════════════════════════════════════════════════

print(f"  Loading model…")
model = PPO.load(
    checkpoint_path,
    env=env,                # re-attach the live environment
    tensorboard_log=log_dir,
    # ── Hyperparameters can be overridden here if you want to fine-tune ──
    # clip_range=0.1,       # tighter clip for refinement
    # learning_rate=1e-4,   # lower LR for fine-tuning
    # ent_coef=0.005,       # less exploration if policy is converging
)

print(f"\n{'─'*60}")
print(f"  Continuing from iter {resume_iter}")
print(f"  Training {Extra_Loops * Iteration_Step} more iters "
      f"({additional_timesteps:,} steps)")
print(f"  Final iter label: {resume_iter + Extra_Loops * Iteration_Step}")
print(f"  Checkpoint every {save_freq:,} steps")
print(f"  Models → {save_dir}")
print(f"  TensorBoard run: {tb_run_name}")
print(f"  TensorBoard: tensorboard --logdir {log_dir}")
print(f"{'─'*60}\n")

callback = TrainSaveCallback(
    save_freq  = save_freq,
    save_path  = save_dir,
    start_iter = resume_iter,   # labels continue from where we left off
    iter_step  = Iteration_Step,
)

model.learn(
    total_timesteps     = additional_timesteps,
    callback            = callback,
    tb_log_name         = tb_run_name,
    reset_num_timesteps = False,   # False = TensorBoard x-axis continues
    log_interval        = 1,
)

print(f"\n✅ Continue training complete.  "
      f"Reached iter {resume_iter + Extra_Loops * Iteration_Step}.")
env.close()
"""
DroneWaypointTrain_Fresh.py
────────────────────────────────────────────────────────────────────────────────
Start a BRAND-NEW PPO training run from scratch.

Configuration summary
─────────────────────
  Total_Loops    = 40
  Iteration_Step =  5   ← iterations saved per loop
  ──────────────────────
  Total iters    = 40 × 5 = 200  (each "iter" = 5 × 2 048 = 10 240 steps)
  Total steps    = 200 × 2 048 = 409 600

  To push beyond 200 iters, increase Total_Loops or Iteration_Step.

Checkpoints are saved every Iteration_Step × Steps_Per_Iteration steps
under  Data/models/ArbitraryPoint/model_PPO_ITER_<N>.zip

TensorBoard logs → Data/logs/ArbitraryPoint/
────────────────────────────────────────────────────────────────────────────────
DroneWaypointTrain_Fresh.py — Phase 1 (Curriculum: short-range targets)
────────────────────────────────────────────────────────────────────────
Start a BRAND-NEW PPO training run from scratch.

Phase 1 curriculum:
  Target range  : ±40 m (north/east), 10 m dead-zone at origin
  MAX_STEPS     : 400   (40 s at 10 Hz — tighter deadline for close targets)
  Goal          : drone consistently reaches targets before extending range

Configuration summary
─────────────────────
  Total_Loops    = 40
  Iteration_Step =  5
  Total iters    = 200  (each iter = 5 × 2048 = 10,240 steps)
  Total steps    = 409,600

PPO hyperparameter notes
────────────────────────
  learning_rate = 0.0002   reduced from 0.0003 to prevent overshooting
  n_epochs      = 6        reduced from 10 to avoid overfitting each rollout
  target_kl     = 0.01     early-stops update if policy changes too fast;
                            prevents the collapse seen in the previous run
  ent_coef      = 0.01     small entropy bonus keeps early exploration alive

Phase 2 readiness (switch when ALL three are stable):
  ep_rew_mean       > +50  consistently
  ep_len_mean       < 300  (drone reaching targets, not timing out)
  explained_variance > 0.6
────────────────────────────────────────────────────────────────────────
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from DroneEnv import DroneEnv


# ════════════════════════════════════════════════════════════════════════
# Environment
# ════════════════════════════════════════════════════════════════════════

env = DroneEnv(
    DEVICE='tcp:127.0.0.1:5762',
    env_id='Instance0'
)


# ════════════════════════════════════════════════════════════════════════
# Custom Callback — checkpoint + console logging
# ════════════════════════════════════════════════════════════════════════

class TrainSaveCallback(BaseCallback):
    """
    Saves a model checkpoint every `save_freq` steps.

    Parameters
    ----------
    save_freq    : int   — how many env steps between saves
    save_path    : str   — directory to write .zip files
    start_iter   : int   — iteration label to start counting from
    iter_step    : int   — how many iteration units each save represents
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
        # n_calls is incremented every step (single env → no division needed)
        if self.n_calls % self.save_freq == 0:
            self.current_iter += self.iter_step

            save_name = f"model_PPO_ITER_{self.current_iter}"
            save_file = os.path.join(self.save_path, save_name)
            self.model.save(save_file)

            print(
                f"  ✓ [{self.num_timesteps:>8d} steps]  "
                f"Saved: {save_name}.zip"
            )

        return True   # returning False would halt training early


# ════════════════════════════════════════════════════════════════════════
# Training configuration
# ════════════════════════════════════════════════════════════════════════

# ── Iteration counts ─────────────────────────────────────────────────────
Pre_Iteration     =   0   # start labelling from this iteration (0 = fresh)
Iteration_Step    =   5   # iters worth of data between checkpoints
Steps_Per_Iter    = 2048  # must match model n_steps
Total_Loops       =  40   # Total_Loops × Iteration_Step = 200 total iters
                           # increase Total_Loops for 250, 300, … iters

total_timesteps = Total_Loops * Iteration_Step * Steps_Per_Iter
# = 40 × 5 × 2 048 = 409 600 steps

save_freq = Iteration_Step * Steps_Per_Iter
# = 5 × 2 048 = 10 240 steps per checkpoint

# ── Save / log paths ─────────────────────────────────────────────────────
save_dir = os.path.join("Data", "models", "ArbitraryPoint")
log_dir  = os.path.join("Data", "logs",   "ArbitraryPoint")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# PPO model (fresh)
# ════════════════════════════════════════════════════════════════════════

model = PPO(
    policy        = "MlpPolicy",
    env           = env,
    # ── Rollout buffer ─────────────────────────────────────────────────
    n_steps       = Steps_Per_Iter,   # steps collected per update cycle
    batch_size    = 512,              # mini-batch size inside each epoch
    n_epochs      = 6,               # PPO update epochs per rollout
    # ── Discount & advantage ────────────────────────────────────────────
    gamma         = 0.99,             # long-horizon discount
    gae_lambda    = 0.95,             # GAE smoothing factor
    target_kl     = 0.01,
    learning_rate = 0.0002,
    # ── Clipping & entropy ──────────────────────────────────────────────
    clip_range    = 0.2,              # standard PPO clip
    ent_coef      = 0.01,             # small entropy bonus → exploration
    # ── Network ─────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=[256, 256]),  # two hidden layers, 256 units
    # ── Misc ────────────────────────────────────────────────────────────
    verbose       = 1,
    tensorboard_log = log_dir,
)

print(f"\n{'─'*60}")
print(f"  Fresh training: 0 → {total_timesteps:,} steps  ({Total_Loops * Iteration_Step} iters)")
print(f"  Checkpoint every {save_freq:,} steps  (iter step = {Iteration_Step})")
print(f"  Models → {save_dir}")
print(f"  Logs   → {log_dir}")
print(f"  TensorBoard: tensorboard --logdir {log_dir}")
print(f"{'─'*60}\n")

# ════════════════════════════════════════════════════════════════════════
# Train
# ════════════════════════════════════════════════════════════════════════

callback = TrainSaveCallback(
    save_freq  = save_freq,
    save_path  = save_dir,
    start_iter = Pre_Iteration,
    iter_step  = Iteration_Step,
)

model.learn(
    total_timesteps    = total_timesteps,
    callback           = callback,
    tb_log_name        = "PPO_Fresh",
    reset_num_timesteps = True,       # True = fresh TensorBoard run
    log_interval       = 1,
)

print("\n✅ Fresh training complete.")
env.close()
"""
DroneWaypointTrain_Fresh.py
────────────────────────────────────────────────────────────────────────────────
Start a BRAND-NEW PPO training run from scratch with curriculum learning,
VecNormalize, and automatic stage advancement.

Environment stack:
    DummyVecEnv → Monitor → DroneEnv
    VecNormalize wraps the full stack (norm_obs=True, norm_reward=True)

Curriculum stages (auto-advanced by TrainSaveCallback):
    1: ±20m  targets, MAX_STEPS=200  → advances when avg_rew>5,  ep_len<150
    2: ±35m  targets, MAX_STEPS=300  → advances when avg_rew>4,  ep_len<250
    3: ±50m  targets, MAX_STEPS=400  → advances when avg_rew>3,  ep_len<350
    4: ±70m  targets, MAX_STEPS=500  → advances when avg_rew>2,  ep_len<450
    5: ±100m targets, MAX_STEPS=600  → final stage

Configuration summary
─────────────────────
  Total_Loops    = 40
  Iteration_Step =  5
  Total iters    = 200  (each iter = 5 × 2048 = 10,240 steps)
  Total steps    = 409,600

PPO hyperparameter notes
────────────────────────
  learning_rate = 0.0003   standard — VecNormalize stabilizes gradients
  n_epochs      = 5        reduced from 6 to avoid overfitting each rollout
  target_kl     = 0.01     early-stops update if policy changes too fast
  ent_coef      = 0.01     small entropy bonus keeps early exploration alive
  ACTION_REPEAT = 4        in DroneEnv — same action applied 4× per step

Checkpoints saved every Iteration_Step × Steps_Per_Iter steps:
    Data/models/ArbitraryPoint/model_PPO_ITER_<N>.zip
    Data/models/ArbitraryPoint/vec_normalize_ITER_<N>.pkl  ← save with model

TensorBoard logs → Data/logs/ArbitraryPoint/
────────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from DroneEnv import DroneEnv


# ════════════════════════════════════════════════════════════════════════
# Environment — DummyVecEnv → Monitor → DroneEnv, wrapped with VecNormalize
# ════════════════════════════════════════════════════════════════════════

env = DummyVecEnv([lambda: Monitor(DroneEnv(
    DEVICE='tcp:127.0.0.1:5762',
    env_id='Instance0'
))])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# ════════════════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════════════════

save_dir = os.path.join("Data", "models", "ArbitraryPoint")
log_dir  = os.path.join("Data", "logs",   "ArbitraryPoint")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# Custom Callback — checkpoint + curriculum advancement
# ════════════════════════════════════════════════════════════════════════

class TrainSaveCallback(BaseCallback):
    """
    Saves model checkpoints every `save_freq` steps and automatically
    advances the curriculum stage when performance thresholds are met.

    Curriculum stage is read/written on the inner DroneEnv instance,
    accessed by unwrapping: VecNormalize → DummyVecEnv → Monitor → DroneEnv.

    Episode stats come from Monitor's info['episode'] dict, which contains
    RAW (un-normalized) rewards. Thresholds are set against raw rewards.

    Parameters
    ----------
    save_freq    : int — steps between checkpoint saves
    save_path    : str — directory to write .zip and .pkl files
    start_iter   : int — iteration label to start from
    iter_step    : int — iteration units per save
    """

    STAGE_THRESHOLDS = {
        1: {'min_rew': 5,  'max_ep_len': 150},
        2: {'min_rew': 4,  'max_ep_len': 250},
        3: {'min_rew': 3,  'max_ep_len': 350},
        4: {'min_rew': 2,  'max_ep_len': 450},
        # Stage 5 is final — no threshold needed
    }

    def __init__(self, save_freq: int, save_path: str,
                 start_iter: int, iter_step: int, verbose: int = 0):
        super().__init__(verbose)
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
        """
        Unwrap the environment stack to reach the DroneEnv instance.
        Stack: VecNormalize → DummyVecEnv → Monitor → DroneEnv
        training_env is VecNormalize, so:
            training_env.envs[0]      = Monitor instance
            training_env.envs[0].env  = DroneEnv instance
        """
        monitor_env = self.training_env.envs[0]   # type: ignore
        return monitor_env.env                      # DroneEnv

    def _on_step(self) -> bool:

        # ── Collect episode stats from Monitor ────────────────────────
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.recent_rewards.append(info['episode']['r'])
                self.recent_ep_lens.append(info['episode']['l'])

        # ── Curriculum check every CHECK_FREQ steps ───────────────────
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

                    # Save checkpoint at transition point
                    save_name = f"model_PPO_ITER_{self.current_iter}_stage{inner.curriculum_stage}"
                    self.model.save(os.path.join(self.save_path, save_name))
                    self.training_env.save(
                        os.path.join(self.save_path, f"vec_normalize_ITER_{self.current_iter}_stage{inner.curriculum_stage}.pkl")
                    )
                    print(f"  ✓ Stage transition checkpoint: {save_name}.zip\n")

        # ── Regular checkpoint save ───────────────────────────────────
        if self.n_calls % self.save_freq == 0:
            self.current_iter += self.iter_step

            save_name = f"model_PPO_ITER_{self.current_iter}"
            self.model.save(os.path.join(self.save_path, save_name))
            self.training_env.save(
                os.path.join(self.save_path, f"vec_normalize_ITER_{self.current_iter}.pkl")
            )

            print(
                f"  ✓ [{self.num_timesteps:>8d} steps]  "
                f"Saved: {save_name}.zip + vec_normalize_ITER_{self.current_iter}.pkl"
            )

        return True


# ════════════════════════════════════════════════════════════════════════
# Training configuration
# ════════════════════════════════════════════════════════════════════════

Pre_Iteration  =   0
Iteration_Step =   5
Steps_Per_Iter = 2048
Total_Loops    =  40

total_timesteps = Total_Loops * Iteration_Step * Steps_Per_Iter
save_freq       = Iteration_Step * Steps_Per_Iter


# ════════════════════════════════════════════════════════════════════════
# PPO model
# ════════════════════════════════════════════════════════════════════════

model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    n_steps         = Steps_Per_Iter,
    batch_size      = 512,
    n_epochs        = 5,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    target_kl       = 0.01,
    learning_rate   = 0.0003,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    policy_kwargs   = dict(net_arch=[256, 256]),
    verbose         = 1,
    tensorboard_log = log_dir,
)

print(f"\n{'─'*60}")
print(f"  Fresh training: 0 → {total_timesteps:,} steps  ({Total_Loops * Iteration_Step} iters)")
print(f"  Checkpoint every {save_freq:,} steps")
print(f"  Models → {save_dir}")
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
    total_timesteps     = total_timesteps,
    callback            = callback,
    tb_log_name         = "PPO_Fresh",
    reset_num_timesteps = True,
    log_interval        = 1,
)

# Save final model and normalizer stats
model.save(os.path.join(save_dir, "model_PPO_FINAL"))
env.save(os.path.join(save_dir, "vec_normalize_FINAL.pkl"))

print("\n✅ Fresh training complete.")
env.close()
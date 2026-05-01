

# ArduPilot RL Navigation

> Built during 12th grade as part of an ongoing robotics + RL project.

Teaching a drone to navigate to waypoints using PPO (Proximal Policy Optimization).

## Stack
- ArduPilot SITL (simulated drone)
- pymavlink (MAVLink communication)
- Stable Baselines3 (PPO)
- Lua scripting (simulation reset)
- Gymnasium (RL environment)

## What it does
Custom Gymnasium environment connecting to ArduPilot SITL via MAVLink.
PPO agent learns to fly to randomised waypoints using velocity commands.

## Current status
🟡 Training in progress — Phase 1 curriculum (±40m targets)

## What I learned
- MAVLink sysid conflicts in multi-instance SITL
- PPO policy collapse diagnosis from TensorBoard
- Reward shaping for sparse navigation tasks
- Stale observation bug in pymavlink buffering
- Curriculum learning for RL

## Training History
- Run 1: 130 episodes — stale observation bug corrupting training data
- Run 2: 200 iterations — policy collapse at 200k steps, large KL updates
- Run 3: current — fixed obs pipeline, asymmetric reward, target_kl=0.01

## Roadmap
- [x] Custom MAVLink Gymnasium environment
- [x] Lua sim reset scripting
- [x] PPO training pipeline
- [x] Curriculum learning setup
- [ ] Consistent target reaching
- [ ] Obstacle avoidance (rangefinder slots already reserved in obs)
- [ ] Gazebo integration

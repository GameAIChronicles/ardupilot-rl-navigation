from stable_baselines3.common.env_checker import check_env
from DroneEnv import DroneEnv

env = DroneEnv()
check_env(env=env)
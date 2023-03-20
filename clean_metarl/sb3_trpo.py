import gym
import metaworld
import random
from gym.wrappers import TimeLimit
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env

print(f"MetaWorld Tasks: {metaworld.ML1.ENV_NAMES}")

t_name = "window-open-v2"

ml1 = metaworld.ML1(t_name)
env = ml1.train_classes[t_name]()

task = random.choice(ml1.train_tasks)

print("task", task)

env.set_task(task)
env = TimeLimit(env, max_episode_steps=500)

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

model.save("window_trpo_25k")

del model 

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

model.save("window_trpo_50k")

del model 

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("window_trpo_100k")

del model 


model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)

model.save("window_trpo_200k")

del model


model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=400000)

model.save("window_trpo_400k")

del model 

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

model.save("window_trpo_1M")

del model 


model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000000)

model.save("window_trpo_5M")

del model 

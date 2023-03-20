import gym
import metaworld
import random
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

print(f"MetaWorld Tasks: {metaworld.ML1.ENV_NAMES}")

t_name = "window-open-v2"

ml1 = metaworld.ML1(t_name)
env = ml1.train_classes[t_name]()

task = random.choice(ml1.train_tasks)

print("task", task)

env.set_task(task)
env = TimeLimit(env, max_episode_steps=500)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=25000)

model.save("window_ppo_25k")

del model 

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=50000)

model.save("window_ppo_50k")

del model 

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=100000)

model.save("window_ppo_100k")

del model 


model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=200000)

model.save("window_ppo_200k")

del model


model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=400000)

model.save("window_ppo_400k")

del model 

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=1000000)

model.save("window_ppo_1M")

del model 


model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=5000000)

model.save("window_ppo_5M")

del model 

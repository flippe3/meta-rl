import gym
import metaworld
import random
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

print(f"MetaWorld Tasks: {metaworld.ML1.ENV_NAMES}")
ml1 = metaworld.ML1('box-close-v2')
env = ml1.train_classes['box-close-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)
env = TimeLimit(env, max_episode_steps=500)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

# model.save("metaworld_ppo")
model.load("metaworld_ppo")



# obs = env.reset()
# for i in range(50):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
    
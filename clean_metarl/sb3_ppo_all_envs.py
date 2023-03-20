import gym
import metaworld
import random
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

print(f"MetaWorld Tasks: {metaworld.ML1.ENV_NAMES}")

for t_name in metaworld.ML1.ENV_NAMES:
    ml1 = metaworld.ML1(t_name)
    env = ml1.train_classes[t_name]()

    task = random.choice(ml1.train_tasks)

    print("task", task)

    env.set_task(task)
    env = TimeLimit(env, max_episode_steps=500)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=1000000)

    model.save(t_name+"_ppo_1M")

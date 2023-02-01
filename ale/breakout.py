from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=64)
env = VecFrameStack(env, n_stack=4)

# model = A2C("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=int(5e6), progress_bar=True)
obs = env.reset()
model = A2C.load("breakout_a2c") 
#model.save("breakout_a2c")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

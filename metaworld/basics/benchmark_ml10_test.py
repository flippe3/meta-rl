import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

testing_envs = []
for name, env_cls in ml10.test_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.test_tasks
                        if task.env_name == name])
  env.set_task(task)
  testing_envs.append(env)

for env in testing_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

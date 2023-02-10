# Might have to run unset LD_PRELOAD to get this to run.
from garage.experiment import Snapshotter
import metaworld
import random
import time
import gym

snapshotter = Snapshotter()

data = snapshotter.load("/home/fleip/Downloads/maml_trpo_17/maml_trpo_17/")
policy = data['algo'].policy

ml1 = metaworld.ML1('drawer-close-v2')
env = ml1.train_classes['drawer-close-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

obs = env.reset()
steps, max_steps = 0, 500
done = False

while steps < max_steps or done:
    a, agent_info = policy.get_action(obs)
    obs, reward, done, info = env.step(a)
    env.render()  # Render the environment to see what's going on (optional)
    time.sleep(0.025)
    steps += 1

env.close()
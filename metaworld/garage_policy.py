# Might have to run unset LD_PRELOAD to get this to run.
from garage.experiment import Snapshotter
import metaworld
import random
import time

snapshotter = Snapshotter()

data = snapshotter.load("/home/fleip/research/meta-rl/metaworld/experiment/maml_trpo_metaworld_ml10_seed=1_epochs=2000_rollouts_per_task=10_meta_batch_size=20_inner_lr=0.0001_6")
policy = data['algo'].policy

ml1 = metaworld.ML1('pick-place-v2')
env = ml1.train_classes['pick-place-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

obs = env.reset()
steps, max_steps = 0, 2000
done = False

while steps < max_steps or done:
    a, agent_info = policy.get_action(obs)
    obs, reward, done, info = env.step(a)
    env.render()  # Render the environment to see what's going on (optional)

    time.sleep(0.025)
    steps += 1

# env.close()

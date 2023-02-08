# Load the policy
from garage.experiment import Snapshotter

snapshotter = Snapshotter()
data = snapshotter.load("/home/fleip/research/meta-rl/metaworld/experiment/maml_trpo_metaworld_ml10_seed=1_epochs=2000_rollouts_per_task=10_meta_batch_size=20_inner_lr=0.0001_6")
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

# See what the trained policy can accomplish
from garage import rollout
path = rollout(env, policy,pause_per_frame=0.025, animated=True)
print(path)
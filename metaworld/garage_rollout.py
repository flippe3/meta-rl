# Load the policy
from garage.experiment import Snapshotter

snapshotter = Snapshotter()
data = snapshotter.load("/home/fleip/research/meta-rl/metaworld/experiment/maml_trpo_metaworld_ml10_seed=1_epochs=3_rollouts_per_task=10_meta_batch_size=20_inner_lr=0.0001")
policy = data['algo'].policy

# You can also access other components of the experiment
env = data['env']

# See what the trained policy can accomplish
from garage import rollout
path = rollout(env, policy, animated=True)
print(path)
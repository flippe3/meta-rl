import metaworld
import random
from utils import sample_random_tasks

seed = 0
# Load metaworld envs and tasks 
ml10 = metaworld.ML10(seed=seed)

# Load envs
training_envs = sample_random_tasks(ml10, 'train')
test_envs = sample_random_tasks(ml10, 'test')

# Create model 
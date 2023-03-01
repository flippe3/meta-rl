# Simple implementation of maml_trpo for metaworld
# inspired from garage github.

import torch
import torch.nn as nn

# policy: (Gaussian) Two headed policy

# value_function: linear feature baseline (not nn)

# policy_optimizer: simple optimizer

# MAML (which does the meta-optimization of an inner algo)

# MAML Meta Optimizer Conjugate Optim

# Inner algo: VPG (TRPO), takes in policy, value function
   
   
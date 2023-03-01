import metaworld
import wandb
import random
import torch

class MAML:
    def __init__(self,
                 inner_algo,
                 inner_optimizer,
                 env,
                 policy,
                 sampler,
                 task_sampler,
                 meta_batch_size=40,
                 inner_lr=0.1,
                 outer_lr=1e-3,
                 num_grad_updates=1,
                 meta_evaluator=None):
        self.inner_algo = inner_algo
        self.inner_optimizer = inner_optimizer
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.task_sampler = task_sampler
        self.meta_batch_size = meta_batch_size
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_grad_updates = num_grad_updates
        self.meta_evaluator=meta_evaluator
        
        # Initialize wandb?
        print("Initialized MAML")

    def train(self, epochs=2000):
        """Training loop that runs the entire training process"""
        for epoch in epochs:
            all_samples, all_params = self.obtain_samples()
            last_return = self.train_once(all_samples, all_params)
        return last_return
    
    def train_once(self, all_samples, all_params):
        # Before meta-optimization
        old_theta = dict(self.policy.named_parameters())
        kl_before = self.compute_kl_constraint(all_samples, all_params, set_grad=False)
        meta_objective = self.compute_meta_loss(all_samples, all_params)

        # Set all gradients to zero in meta_optimizer
        meta_objective.backwards()
        self.meta_optimize(all_samples, all_params)

        # After meta-optimization
        loss_after = self.compute_meta_loss(all_samples, all_params, set_grad=False)
        kl_after = self.compute_kl_constraint(all_samples, all_params, set_grad=False)



        

        
        return
        

    
    def obtain_samples(self):
        """Obtain samples for each task before and after the fast-adaption"""
        
        # Sample tasks (meta_batch_size)
        all_samples = []
        all_params = []
        theta = dict(self.policy.named_parameters())
        
        # Iterate through all training tasks
            # Iterate through num_grad_updates (2)
                # get episodes
                # compute all the rewards, obs, etc. in batches
                # store all samples
                # complete one gradient update
            
            # Store parameters after update
            # Restore policy to old policy

        return all_samples, all_params

    def adapt(self, samples, set_grad=True):
        """Runs one SGD step on the inner policy"""
        loss = self.inner_algo.compute_loss(samples)
        
        # Update policy params with one SGD step
        self.inner_optimizer.set_grads_none()
        loss.backward()
        
        with torch.set_grad_enabled(set_grad):
            self.inner_optimizer.step()

    def compute_meta_loss():
        return
    
    def compute_kl_constraint():
        return

    def meta_optimize(self, all_samples, all_params):
        self.meta_optimizer.step(loss=self.compute_meta_loss(), constraint=self.compute_kl_constraint())

            
            
        


import garage
from garage.experiment.deterministic import set_seed
from garage.experiment import Snapshotter
from garage.trainer import Trainer, TFTrainer
from garage import wrap_experiment

ctxt = Snapshotter()
ctxt.load("/home/fleip/research/garage/metaworld_examples/data/local/experiment/rl2_ppo_metaworld_ml10_seed=1_entropy_coefficient=5e-06_12")

@wrap_experiment
def resume_training(ctxt):
    trainer = Trainer(ctxt)
    trainer.restore("/home/fleip/research/garage/metaworld_examples/data/local/experiment/rl2_ppo_metaworld_ml10_seed=1_entropy_coefficient=5e-06_12")
    trainer.resume(store_episodes=50)

resume_training()

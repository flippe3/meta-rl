import garage
from garage.experiment.deterministic import set_seed
from garage.experiment import Snapshotter
from garage.trainer import Trainer
from garage import wrap_experiment

ctxt = Snapshotter()
ctxt.load("/home/fleip/Downloads/maml_trpo_17/maml_trpo_17")

@wrap_experiment
def resume_training(ctxt):
    trainer = Trainer(ctxt)
    trainer.restore("/home/fleip/Downloads/maml_trpo_17/maml_trpo_17")
    trainer.resume()

resume_training()

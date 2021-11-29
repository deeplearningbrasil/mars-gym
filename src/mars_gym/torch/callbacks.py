import torchbearer
from torchbearer import Trial
from torchbearer.callbacks import on_end_epoch
import wandb

# Example callback running each epoch
@on_end_epoch
def wandb_callback_logger(state):
    fields = {'epoch': state[torchbearer.EPOCH]}
    fields.update(state[torchbearer.METRICS]) 
    wandb.log(fields)
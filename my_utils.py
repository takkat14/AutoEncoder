import torch
import wandb
import numpy as np

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_wandb(notes, params):
    run = wandb.init(notes=notes, entity="takkat14", project="gan-hse-hw1")
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config  # Initialize config
    config.model_type = params['MODEL_TYPE']
    config.batch_size = params['BATCH_SIZE']  # input batch size for training (default: 64)
    config.epochs = params['EPOCHS']  # number of epochs to train (default: 10)
    config.lr = params['LEARNING_RATE']  # learning rate (default: 0.01)
    config.no_cuda = False  # disables CUDA training
    config.seed = params['SEED']  # random seed (default: 42)
    config.log_interval = params['LOG_INTERVAL']  # how many batches to wait before logging training status
    return config, run

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils   import spectral_norm 
# from torchsummary import summary
import os
import random

def seed_everything(seed=3407):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True

def save_checkpoint(model, optimizer, path="my_checkpoint.pth.tar"):
    """
    Saving training checkpoints
    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        WGAN model
    optimizer : :obj:`torch.optim`
        Optimizer
    path : :obj:`str`
        Path and file name
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Loading training checkpoints
    Parameters
    ----------
    checkpoint_file : :obj:`str`
        Path and file name
    model : :obj:`torch.nn.Module`
        WGAN model
    optimizer : :obj:`torch.optim`
        Optimizer
    lr : :obj:`float`
        Learning rate
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        

class MyLoader(Dataset):
    """
    Prepare dataset for data loader
    Attributes:
    data: dataset
    """

    def __init__(self, data, condition):
        super(MyLoader, self).__init__()
        self.data = data
        self.condition = condition

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item, ...].squeeze(), self.condition[item, ...].squeeze()

def loss_function(recon_data,real,mean,logvar):
    MSEloss = torch.nn.MSELoss()
    MSE = MSEloss(recon_data, real)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return MSE + KLD
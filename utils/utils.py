import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, roc_auc_score
import logging

logger = logging.getLogger(__name__)


def time_since(start_time):
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def get_path(base_dir, base_name, suffix=''):
    return os.path.join(base_dir, base_name + suffix)


def set_random_seed(seed, device):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = EasyDict(config)
    return config


def save_model(model, base_dir, base_name):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), get_path(base_dir, base_name, '.pt'))


def load_model(model, model_weights_path, device, copy_to_cpu=True):
    raw_model = model.module if hasattr(model, "module") else model
    map_location = lambda storage, loc: storage if copy_to_cpu else None
    raw_model.load_state_dict(torch.load(model_weights_path, map_location))
    return raw_model.to(device)

def log_GPU_info():
    logger.info('GPU INFO:')
    logger.info(f'Available devices: {torch.cuda.device_count()}')
    logger.info(f'GPU name: {torch.cuda.get_device_name(0)}')
    logger.info(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')

def get_metrics(y_hat, y_test, print_metrics=True):
    # logger.debug(f'y_hat: {y_hat}, y_test: {y_test}')
    acc = accuracy_score(y_test, y_hat)
    sn = recall_score(y_test, y_hat)
    sp = recall_score(y_test, y_hat, pos_label=0)
    mcc = matthews_corrcoef(y_test, y_hat)
    auroc = roc_auc_score(y_test, y_hat)
    
    if print_metrics:
        print(f'Acc(%) \t Sn(%) \t Sp(%) \t MCC \t AUROC')
        print(f'{acc*100:.2f}\t{sn*100:.2f}\t{sp*100:.2f}\t{mcc:.3f}\t{auroc:.3f}')
    return acc, sn, sp, mcc, auroc

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1_embd, x2_embd, label):
        dist = F.pairwise_distance(x1_embd, x2_embd)
        loss = torch.mean((1 - label) * torch.pow(dist, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

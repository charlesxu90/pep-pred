import pandas as pd
from torch.utils.data import Dataset
from utils.utils import get_path
import logging
import torch

logger = logging.getLogger(__name__)


def load_data(data_path, col_name='SMILES'):
    col_names = col_name.split(',') # for cross modal dataset
    
    train_data = pd.read_csv(get_path(data_path, 'train.csv'))[col_names].values
    valid_data = pd.read_csv(get_path(data_path, 'test.csv'))[col_names].values
    return train_data, valid_data


class UniDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx][0]
        # logger.debug(f'UniDataset: {item}')
        return item


class CrossDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        [item1, item2] = item
        # logger.debug(f'CrossDataset: {item1}, {item2}')
        return item1, item2
    

class TaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        # print(f'rank {torch.distributed.get_rank()}, fetch sample {idx}')

        item = self.dataset[idx]
        [item, prop] = item
        # logger.debug(f'TaskDataset: {item}, {prop}')
        return item, prop
    

def cl_collate(batch):
    # return pair of sequences by split data into two halves (seq1, seq2, label1, label2, label)

    batch_size = len(batch)
    # logger.debug(f"batch_size: {batch_size}")
    # logger.debug(f"batch type: {type(batch)}")

    seq1_batch = [batch[i] for i in range(int(batch_size / 2))]
    seq2_batch = [batch[i + int(batch_size/2)] for i in range(int(batch_size / 2))]
    # logger.debug(f"seq1_batch: {seq1_batch}")
    seq1, label1 = zip(*seq1_batch)
    seq2, label2 = zip(*seq2_batch)
    # logger.debug(f"seq1: {seq1}, label1: {label1}")
    label = [label1[i] ^ label2[i] for i in range(int(batch_size / 2))]

    # covert label to tensor
    label1 = torch.tensor(label1)
    label2 = torch.tensor(label2)
    label = torch.tensor(label)
    
    return list(seq1), list(seq2), label1, label2, label

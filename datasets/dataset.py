import pandas as pd
from torch.utils.data import Dataset
from utils.utils import get_path
import logging

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
        item = self.dataset[idx]
        [item, prop] = item
        # logger.debug(f'TaskDataset: {item}, {prop}')
        return item, prop
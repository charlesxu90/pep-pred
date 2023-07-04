import pandas as pd
from torch.utils.data import Dataset

from utils.utils import get_path
import logging
import torch
from torch.utils.data import TensorDataset
import numpy as np

logger = logging.getLogger(__name__)


def load_data(data_path, col_name='SMILES'):
    if col_name == 'SMILES' or col_name == 'helm':
        train_data = pd.read_csv(get_path(data_path, 'train.csv'))[col_name].values
        valid_data = pd.read_csv(get_path(data_path, 'test.csv'))[col_name].values
    else:
        col_names = col_name.split(',')
        train_data = pd.read_csv(get_path(data_path, 'train.csv'))[col_names].values
        valid_data = pd.read_csv(get_path(data_path, 'test.csv'))[col_names].values
    return train_data, valid_data

class SmilesTokenizer(object):
    PAD = ' '
    BEGIN = 'Q'
    END = '\n'
    MASK = '!'

    def __init__(self) -> None:
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
                         '.': 30,
                         '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
                         '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
                         'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
                         'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28,
                         "@": 42, "R": 43, '/': 44, "\\": 45, 'E': 46, self.MASK: 47,
                         }

        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def allowed(self, smiles) -> bool:
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                logger.info('Forbidden symbol {:<2}  in  {}'.format(symbol, smiles))
                return False
        return True

    def encode_atoms(self, smiles: str) -> str:
        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decode_atoms(self, smiles):
        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def get_vocab_size(self) -> int:
        return len(self.idx_char)

    @property
    def begin_token_id(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_token_id(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_token_id(self) -> int:
        return self.char_idx[self.PAD]

    @property
    def mask_token_id(self) -> int:
        return self.char_idx[self.MASK]

    def batch_decode(self, array):
        """ Decode a SMILES string from an encoded matrix or array
        """
        smiles_strings = []

        for row in array:
            predicted_chars = []

            for j in row:
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)

            smi = ''.join(predicted_chars)
            smi = self.decode_atoms(smi)
            smiles_strings.append(smi)

        return smiles_strings

    def batch_encode(self, smiles, max_len=100):
        """ Encode a batch of smiles strings into a padded torch tensor
        """
        batch_size = len(smiles)
        idx_matrix = torch.zeros((batch_size, max_len))
        for i, smi in enumerate(smiles):
            idx_matrix[i, :] = self.encode(smi, max_len=max_len)

        return idx_matrix.to(torch.int64)
    
    def encode(self, smiles, max_len=100):
        """ Encode a smiles string into a padded torch tensor
        """
        # logger.debug(f'Encoding smiles: {smiles}')
        idx_matrix = torch.zeros((1, max_len))
        enc_smi = self.BEGIN + self.encode_atoms(smiles) + self.END
        for j in range(max_len):
            if j >= len(enc_smi):
                break
            idx_matrix[0, j] = self.char_idx[enc_smi[j]]

        return idx_matrix.to(torch.int64)


class SmilesDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        smiles = self.dataset[idx]
        input_ids = self.tokenizer.encode(smiles, max_len=self.max_len)
        return input_ids
    

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
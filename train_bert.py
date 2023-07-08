import argparse
import os
import logging
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader

from utils.utils import parse_config, load_model, log_GPU_info
from datasets.dataset import load_data, UniDataset
from datasets.tokenizer import SmilesTokenizer, AATokenizer
from models.bert import BERT
from models.bert_trainer import BertTrainer

    
def main(args, config):    
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    
    log_GPU_info(logger)
    
    logger.info(f"Create dataset")
    train_data, valid_data = load_data(config.data.input_path, col_name=config.data.col_name,)
    train_dataloader = DataLoader(UniDataset(train_data), batch_size=config.data.batch_size, shuffle=True, num_workers=4, 
                                  pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(UniDataset(valid_data), batch_size=config.data.batch_size, shuffle=False, num_workers=4, 
                                 pin_memory=True, persistent_workers=True)

    logger.info(f"Initialize model")
    if config.data.type == 'smiles':
        tokenizer = SmilesTokenizer()
    elif config.data.type == 'aa_seq':
        tokenizer = AATokenizer()
    else:
        raise Exception(f"Unknown data type: {config.data.type}")
    
    model = BERT(tokenizer=tokenizer, **config.model).to(device)
    if args.ckpt is not None:
        model = load_model(model, args.ckpt, device)
    
    logger.info(f"Start training")
    trainer = BertTrainer(model, args.output_dir)
    trainer.fit(train_dataloader, test_dataloader, n_epochs=config.train.max_epochs)
    logger.info(f"Training finished")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='train_smi_bert.yaml')
    parser.add_argument('--output_dir', default='checkpoints/pretrain/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', default=None, type=str, help='path to checkpoint to load')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)

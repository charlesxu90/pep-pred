import argparse
import os
import logging
import torch
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from utils.utils import parse_config, log_GPU_info
from datasets.dataset import load_data, CrossDataset
from datasets.tokenizer import SmilesTokenizer
from models.molclip import MolCLIP
from models.molclip_trainer import CrossTrainer

    
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
    train_dataloader = DataLoader(CrossDataset(train_data), batch_size=config.data.batch_size, shuffle=True, num_workers=4, 
                                  pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(CrossDataset(valid_data), batch_size=config.data.batch_size, shuffle=False, num_workers=4, 
                                 pin_memory=True, persistent_workers=True)

    logger.info(f"Initialize model")
    tokenizer = SmilesTokenizer()
    model = MolCLIP(tokenizer=tokenizer, device=device, config=config.model)
    
    logger.info(f"Start training")
    trainer = CrossTrainer(model, args.output_dir)
    trainer.fit(train_dataloader, test_dataloader, n_epochs=config.train.max_epochs)
    logger.info(f"Training finished")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_molclip.yaml')
    parser.add_argument('--output_dir', default='results/train_molclip/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)

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
import lightning.pytorch as pl

from utils.utils import parse_config
from datasets.smiles_dataset import load_data, SmilesDataset, SmilesTokenizer
from models.smi_bert import SmilesBERT

    
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
    
    logger.info(f"Create dataset")
    train_data, valid_data = load_data(config.data.input_path)
    tokenizer = SmilesTokenizer()
    train_data = SmilesDataset(tokenizer=tokenizer, dataset=train_data, max_len=config.data.max_len)
    valid_data = SmilesDataset(tokenizer=tokenizer, dataset=valid_data, max_len=config.data.max_len)
    train_dataloader = DataLoader(train_data, batch_size=config.data.train_batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(valid_data, batch_size=config.data.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    logger.info(f"Initialize model")
    model = SmilesBERT(tokenizer=tokenizer, **config.model).to(device)
    logger.info(f"Start training")
    trainer = pl.Trainer(accelerator=args.device, **config.train)
    # trainer.fit(model, train_dataloader, test_dataloader, )
    trainer.fit(model,test_dataloader, )
    trainer.save_checkpoint(os.path.join(args.output_dir, 'model.ckpt'))
    logger.info(f"Save model to {args.output_dir}")
    
    logger.info(f"Training finished")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='checkpoints/pretrain/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    config = parse_config(args.config)
    main(args, config)

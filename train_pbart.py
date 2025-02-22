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

from utils.utils import parse_config, log_GPU_info, load_model
from datasets.dataset import load_data, CrossDataset
from models.pbart import PepBART
from models.trainer import Trainer
from torch.utils.data.distributed import DistributedSampler
from utils.dist import init_distributed, get_rank, is_main_process
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main(args, config):
    init_distributed()
    global_rank = get_rank()

    device = torch.device(args.device)
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    if is_main_process():
        log_GPU_info()
    
    train_data, valid_data = load_data(config.data.input_path, col_name=config.data.col_name,)
    train_set, test_set = CrossDataset(train_data), CrossDataset(valid_data)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank)
    train_dataloader = DataLoader(train_set, batch_size=config.data.batch_size, sampler=train_sampler, num_workers=config.data.num_workers, pin_memory=True)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank)
    test_dataloader = DataLoader(test_set, batch_size=config.data.batch_size, sampler=test_sampler, shuffle=False, num_workers=config.data.num_workers, pin_memory=True)

    model = PepBART(device=device, config=config.model).to(device)
    if args.ckpt is not None:
        model = load_model(model, args.ckpt, device)
    if args.aa_ckpt is not None:
        model.load_pretrained_encoder(args.aa_ckpt)
    
    logger.info(f"Start training")
    trainer = Trainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_pep_bart.yaml')
    parser.add_argument('--output_dir', default='results/train_pep_bart/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', default=None, type=str, help='path to checkpoint to load')
    parser.add_argument('--aa_ckpt', default=None, type=str, help='path to aa_bert checkpoint to load')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)

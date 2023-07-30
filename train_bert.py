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
from datasets.tokenizer import SmilesTokenizer, AATokenizer, HELMTokenizer
from models.bert import BERT
from models.bert_trainer import BertTrainer
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

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    
    if is_main_process():
        log_GPU_info()
    
    train_data, valid_data = load_data(config.data.input_path, col_name=config.data.col_name,)
    train_set, test_set = UniDataset(train_data), UniDataset(valid_data)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank)
    train_dataloader = DataLoader(train_set, batch_size=config.data.batch_size, sampler=train_sampler, num_workers=config.data.num_workers, pin_memory=True)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank)
    test_dataloader = DataLoader(test_set, batch_size=config.data.batch_size, sampler=test_sampler, shuffle=False, num_workers=config.data.num_workers, pin_memory=True)

    if config.data.type == 'smiles':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif config.data.type == 'aa_seq':
        tokenizer = AATokenizer(max_len=config.data.max_len)
    elif config.data.type == 'helm':
        tokenizer = HELMTokenizer(max_len=config.data.max_len)
    else:
        raise Exception(f"Unknown data type: {config.data.type}")
    
    model = BERT(tokenizer=tokenizer, **config.model).to(device)
    if args.ckpt is not None:
        model = load_model(model, args.ckpt, device)
    
    logger.info(f"Start training")
    trainer = BertTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
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

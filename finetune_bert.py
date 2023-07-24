import argparse
import os
import logging
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader

from utils.utils import parse_config, load_model, log_GPU_info
from datasets.dataset import load_data, TaskDataset, cl_collate
from datasets.tokenizer import SmilesTokenizer, AATokenizer
from models.bert import BERT, TaskPred
from models.task_trainer import TaskTrainer
from torch.utils.data.distributed import DistributedSampler
from utils.dist import init_distributed, get_rank, is_main_process
from torch.distributed.elastic.multiprocessing.errors import record
import warnings
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")
import copy

@record
def main(args, config):
    device = torch.device(config.train.device)
    if config.train.distributed:
        init_distributed()
        global_rank = get_rank()
        seed = args.seed + global_rank
    else:
        seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    
    if is_main_process():
        log_GPU_info()
    
    collate_fn = cl_collate if config.model.model_type == 'siamese' else None
    batch_size, num_workers = config.data.batch_size, config.data.num_workers
    
    train_data, valid_data = load_data(config.data.input_path, col_name=config.data.col_name,)
    train_set, test_set = TaskDataset(train_data), TaskDataset(valid_data)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank) if config.train.distributed else None
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank) if config.train.distributed else None
    test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    # logger.debug(f"train_sampler: {len(train_sampler)}, test_sampler: {len(test_sampler)}")
    val_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)

    if config.data.type == 'smiles':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif config.data.type == 'aa_seq':
        tokenizer = AATokenizer(max_len=config.data.max_len)
    else:
        raise Exception(f"Unknown data type: {config.data.type}")
    
    model = BERT(tokenizer=tokenizer, **config.model.bert).to(device)
    if args.ckpt is not None:
        model = load_model(model, args.ckpt, device)
    
    pred_model = TaskPred(model, model_type=config.model.model_type, device=device)

    logger.info(f"Start training")
    trainer = TaskTrainer(pred_model, args.output_dir, model_type=config.model.model_type, **config.train)
    trainer.fit(train_dataloader, test_dataloader, val_dataloader)
    logger.info(f"Training finished")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='train_smi_bert.yaml')
    parser.add_argument('--output_dir', default='checkpoints/pretrain/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', default=None, type=str, help='path to checkpoint to load')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)

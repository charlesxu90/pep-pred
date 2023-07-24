import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd

from utils.utils import parse_config, load_model, get_metrics
from models.bert import TaskPred, BERT
from datasets.tokenizer import SmilesTokenizer, AATokenizer

def load_task_model(ckpt, config, device='cuda', model_type='smi_bert'):
    if model_type == 'smi_bert':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif model_type == 'aa_bert':
        tokenizer = AATokenizer(max_len=config.data.max_len)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')

    model = BERT(tokenizer, **config.model.bert)
    pred_model = TaskPred(model, model_type=config.model.model_type, device=device)
    model = load_model(pred_model, ckpt, device)
    model.eval()
    return model

def eval_task_model(model, X_test, y_test):
    with torch.set_grad_enabled(False):
        output, _ = model.forward(X_test)
        y_hat = output.squeeze()
        y_hat = y_hat.argmax(axis=1)

    y_test_hat = y_hat.cpu().numpy()
    _, _, _, mcc, _ = get_metrics(y_test_hat, y_test, print_metrics=True)

    return mcc


def main(args, config):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)

    df_test_cpp924 = pd.read_csv('data/CPP924/test.csv')
    y_test = df_test_cpp924.is_cpp.values

    if args.model_type == 'aa_bert':
        X_test = df_test_cpp924.aa_seq.values

        ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
        best_mcc = 0
        for ckpt in ckpt_files:
            ckpt = os.path.join(args.ckpt_dir, ckpt)
            print(ckpt)
            model = load_task_model(ckpt, config, device=args.device, model_type=args.model_type)
            mcc = eval_task_model(model, X_test, y_test)

            if mcc > best_mcc:
                best_mcc = mcc
                best_ckpt = ckpt
                logger.info(f'best_mcc: {best_mcc}, best_ckpt: {best_ckpt}')
        
        logger.info(f'best_mcc: {best_mcc}, best_ckpt: {best_ckpt}')
        model = load_task_model(best_ckpt, config, device=args.device, model_type=args.model_type)
        eval_task_model(model, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='aa_bert', type=str, help='model type: aa_bert, smi_bert, molclip, pep_bart')
    parser.add_argument('--ckpt_dir', type=str, help='path to checkpoint directory')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()  
    config = parse_config(args.config)
    main(args, config)

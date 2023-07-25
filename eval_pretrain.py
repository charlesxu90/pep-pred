import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd

from utils.utils import parse_config, load_model, get_metrics
from models.bert import BERT
from models.molclip import MolCLIP
from models.pbart import PepBART
from datasets.tokenizer import SmilesTokenizer, AATokenizer

def load_bert_model(ckpt, config, device='cuda', model_type='smi_bert'):
    if model_type == 'smi_bert':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif model_type == 'aa_bert':
        tokenizer = AATokenizer(max_len=config.data.max_len)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')

    model = BERT(tokenizer, **config.model)
    model = load_model(model, ckpt, device)
    model.eval()
    return model, device

def get_bert_embd(encoder, inputs, device='cuda', cls_token=False):
    tokens = encoder.tokenize_inputs(inputs).to(device)
    batch_lens = (tokens != encoder.tokenizer.pad_token_id).sum(1)
    embd = encoder.embed(tokens)
    if cls_token:
        return embd[:, 0, :].squeeze()
    else:
        reps = []
        for i, tokens_len in enumerate(batch_lens):
            reps.append(embd[i, 1 : tokens_len - 1].mean(0))
        
        return torch.stack(reps)

def encode_with_bert(list, model, device='cuda', cls_token=False):
    with torch.no_grad():
        output= get_bert_embd(model, list, device=device, cls_token=cls_token)
        embd = output.cpu().numpy()
    return embd

def get_metric_by_clf(X_train, y_train, X_test, y_test, clf='xgb'):
    if clf == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        
        best_mcc = 0
        for i in range(10):
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)

            acc, sn, sp, mcc, auroc = get_metrics(y_hat, y_test, print_metrics=False)
            if mcc > best_mcc:
                accs, sns, sps, mccs, aurocs = [acc], [sn], [sp], [mcc], [auroc]
                best_mcc = mcc
        print(f'Best results: Acc(%) \t Sn(%) \t Sp(%) \t MCC \t AUROC')
        print(f'{np.mean(accs)*100:.2f}\t{np.mean(sns)*100:.2f}\t{np.mean(sps)*100:.2f}\t{np.mean(mccs):.3f}\t{np.mean(aurocs):.3f}')
        return best_mcc

    elif clf == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(eta=0.1, max_depth=6, n_estimators=1000, n_jobs=8, random_state=1)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        _, _, _, mcc, _ = get_metrics(y_hat, y_test)
        return mcc

    elif clf == 'svm':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', C=1, gamma=0.1, random_state=1)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        _, _, _, mcc, _ = get_metrics(y_hat, y_test)
        return mcc

def load_molclip_model(ckpt, config, device='cuda'):
    model = MolCLIP(device=device, config=config.model)
    model = load_model(model, ckpt, device)
    model.eval()
    return model

def get_molclip_embd(model, smiles, aa_seq):
    with torch.no_grad():
        smi_output, _ = model.get_smi_embd(smiles)
        smi_feat = model.smi_proj(smi_output)
        aa_output, _ = model.get_aa_embd(aa_seq)
        aa_feat = model.aa_proj(aa_output)
        feat = torch.cat([smi_feat, aa_feat], dim=1)
        embd = feat.cpu().numpy()
    return embd

def load_pep_bart_model(ckpt, config, device='cuda'):

    model = PepBART(device=device, config=config.model).to(device)
    model = load_model(model, ckpt, device)
    model.eval()
    return model, device

def get_pep_bart_embd(model, inputs, device='cuda'):
    tokens = model.aa_encoder.tokenize_inputs(inputs).to(device)
    embd = model.aa_encoder.embed(tokens)
    batch_lens = (tokens != model.aa_encoder.tokenizer.pad_token_id).sum(1)
    reps = []
    for i, tokens_len in enumerate(batch_lens):
        reps.append(embd[i, 1 : tokens_len - 1].mean(0))
    del tokens, batch_lens, embd
    return torch.stack(reps)

def encode_with_pep_bart(list, model, device='cuda'):
    with torch.no_grad():
        output= get_pep_bart_embd(model, list, device=device)
        embd = output.cpu().numpy()
    return embd

def main(args, config):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)

    df_train_cpp924 = pd.read_csv('data/CPP924/train.csv')
    df_test_cpp924 = pd.read_csv('data/CPP924/test.csv')

    y_train = df_train_cpp924.is_cpp.values
    y_test = df_test_cpp924.is_cpp.values
    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
    best_mcc = 0
    for ckpt in ckpt_files:
        ckpt = os.path.join(args.ckpt_dir, ckpt)
        print(ckpt)

        if args.model_type == 'smi_bert':
            model, device = load_bert_model(ckpt=ckpt, config=config, device=args.device, model_type=args.model_type)
            X_train = encode_with_bert(df_train_cpp924.smi, model, device=device, cls_token=args.cls_token_embd)
            X_test = encode_with_bert(df_test_cpp924.smi, model, device=device, cls_token=args.cls_token_embd)
        elif args.model_type == 'molclip':
            model = load_molclip_model(ckpt=ckpt, config=config, device=args.device)
            X_train = get_molclip_embd(model, df_train_cpp924.smi, df_train_cpp924.aa_seq)
            X_test = get_molclip_embd(model, df_test_cpp924.smi, df_test_cpp924.aa_seq)
        elif args.model_type == 'aa_bert':
            model, device = load_bert_model(ckpt=ckpt, config=config, device=args.device, model_type=args.model_type)
            X_train = encode_with_bert(df_train_cpp924.aa_seq, model, device=device, cls_token=args.cls_token_embd)
            X_test = encode_with_bert(df_test_cpp924.aa_seq, model, device=device, cls_token=args.cls_token_embd)
        elif args.model_type == 'pep_bart':
            model, device = load_pep_bart_model(ckpt=ckpt, config=config, device=args.device)
            X_train = encode_with_pep_bart(df_train_cpp924.aa_seq, model)
            X_test = encode_with_pep_bart(df_test_cpp924.aa_seq, model)

        logger.debug(f"data shape X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        mcc = get_metric_by_clf(X_train, y_train, X_test, y_test, clf=args.clf)
        if mcc > best_mcc:
            best_mcc = mcc
            best_ckpt = ckpt
            logger.info(f'best_mcc: {best_mcc}, best_ckpt: {best_ckpt}')
    logger.info(f'best_mcc: {best_mcc}, best_ckpt: {best_ckpt}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='smi_bert', type=str, help='model type: smi_bert, molclip, pep_bart')
    parser.add_argument('--ckpt_dir', type=str, help='path to checkpoint directory, containing .pt files')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clf', default='rf', type=str, help='classifier: rf, svm, xgb')
    parser.add_argument('--cls_token_embd', action='store_true', help='use cls token embedding')

    args = parser.parse_args()  
    config = parse_config(args.config)
    main(args, config)

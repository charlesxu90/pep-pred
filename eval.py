import argparse
import logging
import torch
import numpy as np
import pandas as pd

from utils.utils import parse_config, load_model, get_metrics
from models.bert import BERT
from models.molclip import MolCLIP
from datasets.tokenizer import SmilesTokenizer

def load_smi_bert_model(ckpt, config, device='cuda'):
    tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    model = BERT(tokenizer, **config.model)
    model = load_model(model, ckpt, device)
    model.eval()
    return model, device

def get_smi_embd(smi_encoder, smiles, device='cuda', cls_token=False):
    smi_tokens = smi_encoder.tokenize_inputs(smiles).to(device)
    batch_lens = (smi_tokens != smi_encoder.tokenizer.pad_token_id).sum(1)
    smi_embd = smi_encoder.embed(smi_tokens)
    if cls_token:
        return smi_embd[:, 0, :].squeeze()
    else:
        smi_reps = []
        for i, tokens_len in enumerate(batch_lens):
            smi_reps.append(smi_embd[i, 1 : tokens_len - 1].mean(0))
        
        return torch.stack(smi_reps)

def encode_smi(smi_list, model, device='cuda', cls_token=False):
    with torch.no_grad():
        output= get_smi_embd(model, smi_list, device=device, cls_token=cls_token)
        embd = output.cpu().numpy()
    return embd

def get_metric_by_clf(X_train, y_train, X_test, y_test, clf='xgb'):
    if clf == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        
        accs, sns, sps, mccs, aurocs = [], [], [], [], []
        for i in range(10):
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)

            acc, sn, sp, mcc, auroc = get_metrics(y_hat, y_test)
            accs.append(acc), sns.append(sn), sps.append(sp), mccs.append(mcc), aurocs.append(auroc)
        print(f'Mean results: Acc(%) \t Sn(%) \t Sp(%) \t MCC \t AUROC')
        print(f'{np.mean(accs)*100:.2f}\t{np.mean(sns)*100:.2f}\t{np.mean(sps)*100:.2f}\t{np.mean(mccs):.3f}\t{np.mean(aurocs):.3f}')

    elif clf == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(eta=0.1, max_depth=6, n_estimators=1000, n_jobs=8, random_state=1)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        get_metrics(y_hat, y_test)

    elif clf == 'svm':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', C=1, gamma=0.1, random_state=1)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        get_metrics(y_hat, y_test)

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

def main(args, config):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)

    df_train_cpp924 = pd.read_csv('data/CPP924/train.csv')
    df_test_cpp924 = pd.read_csv('data/CPP924/test.csv')

    y_train = df_train_cpp924.is_cpp.values
    y_test = df_test_cpp924.is_cpp.values

    if args.model_type == 'smi_bert':
        model, device = load_smi_bert_model(ckpt=args.ckpt, config=config, device=args.device)
        X_train = encode_smi(df_train_cpp924.smi, model, device=device, cls_token=args.cls_token_embd)
        X_test = encode_smi(df_test_cpp924.smi, model, device=device, cls_token=args.cls_token_embd)
    elif args.model_type == 'molclip':
        model = load_molclip_model(ckpt=args.ckpt, config=config, device=args.device)
        X_train = get_molclip_embd(model, df_train_cpp924.smi, df_train_cpp924.aa_seq)
        X_test = get_molclip_embd(model, df_test_cpp924.smi, df_test_cpp924.aa_seq)

    logger.debug(f"data shape X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    get_metric_by_clf(X_train, y_train, X_test, y_test, clf=args.clf)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='smi_bert', type=str, help='model type: smi_bert, molclip')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint to load')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clf', default='rf', type=str, help='classifier: rf, svm, xgb')
    parser.add_argument('--cls_token_embd', action='store_true', help='use cls token embedding')

    args = parser.parse_args()  
    config = parse_config(args.config)
    main(args, config)

import argparse
import logging
import torch
import pandas as pd

from transformers import RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from utils.utils import parse_config
from datasets.smiles_tokenizer import SmilesTokenizer
from datasets.smiles_dataset import load_data, SmilesDataset

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    config = parse_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running with device: {device}')

    train_data, valid_data = load_data(config.data.input_path)

    max_len = config.tokenizer.max_len
    tokenizer = SmilesTokenizer.from_pretrained("roberta-base", max_len=max_len)
    data_train = SmilesDataset(tokenizer=tokenizer, dataset=train_data, max_len=max_len)
    data_valid = SmilesDataset(tokenizer=tokenizer, dataset=valid_data, max_len=max_len)
    logger.info(f'Train dataset size: {len(data_train)}, Valid dataset size: {len(data_valid)}')
 

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=config.dataloader.mlm_probability)

    model_config = RobertaConfig(**config.model)
    model = RobertaForMaskedLM(config=model_config).to(device)

    training_args = TrainingArguments(**config.train)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data_train,
        eval_dataset=data_valid
    )

    trainer.train()
    trainer.save_model(config.train.output_dir)
    exit()

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain SMILES BERT model')
    parser.add_argument('--config', required=True, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

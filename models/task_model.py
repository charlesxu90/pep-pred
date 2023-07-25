import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_transformer import QuickGELU

logger = logging.getLogger(__name__)

class TaskPred(nn.Module):
    def __init__(self, bert_model, model_type='bert', device='cuda', mlp1_embd=512, mlp2_hid_size=None):
        super().__init__()
        self.model_type = model_type
        self.bert = bert_model
        self.mlp1_embed = mlp1_embd
        self.mlp2_hid_size = [256, 128, 64] if mlp2_hid_size is None else [int(i) for i in mlp2_hid_size.split(',')]
        # logger.debug(f'mlp2_hid_size: {self.mlp2_hid_size}')
        self.mlp1 = nn.Sequential(
            nn.Linear(self.bert.transformer.width, mlp1_embd),
            nn.BatchNorm1d(mlp1_embd),
            QuickGELU(),
            nn.Linear(mlp1_embd, self.bert.transformer.width),
        ) if self.model_type == 'siamese' else None
        mlp2_layers = [[nn.Linear(self.bert.transformer.width, self.mlp2_hid_size[0]), nn.BatchNorm1d(self.mlp2_hid_size[0]), QuickGELU()]] + \
            [[nn.Linear(self.mlp2_hid_size[i], self.mlp2_hid_size[i+1]), nn.BatchNorm1d(self.mlp2_hid_size[i+1]), QuickGELU()]
             for i in range(len(self.mlp2_hid_size) - 1)] + [[nn.Linear(self.mlp2_hid_size[-1], 2)]]
        self.mlp2 = nn.Sequential(*[layer for layers in mlp2_layers for layer in layers])
        self.device = device
        self.to(self.device)
        
    def get_bert_embd(self, tokens):
        outputs = self.bert.embed(tokens)
        batch_lens = (tokens != self.bert.tokenizer.pad_token_id).sum(1)
        reps = []
        for i, tokens_len in enumerate(batch_lens):
            reps.append(outputs[i, 1 : tokens_len - 1].mean(0))
            
        reps = torch.stack(reps)
        return reps

    def forward(self, inputs):
        tokens = self.bert.tokenize_inputs(inputs).to(self.device)
        reps = self.get_bert_embd(tokens)

        mlp1_embd = self.mlp1(reps) + reps if self.model_type == 'siamese' else reps
        mlp2_output = self.mlp2(mlp1_embd)
        return mlp2_output, mlp1_embd

    def configure_optimizers(self, learning_rate=1e-4):
        return self.bert.configure_optimizers(learning_rate)
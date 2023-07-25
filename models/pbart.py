import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_transformer import LayerNorm, TransformerDecoder
from .bert import BERT
from datasets.tokenizer import SmilesTokenizer, AATokenizer
from utils.utils import load_model

logger = logging.getLogger(__name__)


class BARTDecoder(nn.Module):
    def __init__(self,
                 tokenizer,
                 context_length: int,
                 width: int,
                 n_heads: int,
                 n_layers: int,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.token_embedding = nn.Embedding(self.vocab_size, width)
        causal_attn_mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1).bool()
        self.decoder = TransformerDecoder(
            width=width,
            n_layers=n_layers,
            n_heads=n_heads,
            attn_mask=causal_attn_mask,
            source_attn_mask=None,
        )
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = LayerNorm(width)
        self.proj = nn.Linear(width, self.vocab_size)  # decodes into vocab_size

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.decoder.width ** -0.5) * ((2 * self.decoder.n_layers) ** -0.5)
        attn_std = self.decoder.width ** -0.5
        fc_std = (2 * self.decoder.width) ** -0.5
        for block in self.decoder.resblocks:
            nn.init.normal_(block.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.cross_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def tokenize_inputs(self, inputs):
        """ remove useless dimension, and fill to the context_length
        """
        inputs = self.tokenizer.tokenize(inputs)
        inputs = nn.functional.pad(inputs, (0, self.context_length - inputs.shape[1]+1), value=self.tokenizer.pad_token_id)  # padding inputs length to n_ctx
        return inputs

    def forward(self, inputs, encoder_hidden_states=None):
        x = self.token_embedding(inputs).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # BLD -> LBD
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)  # BLD -> LBD

        x = self.decoder(x, encoder_hidden_states)
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.proj(self.ln_final(x))  # x.shape = [batch_size, n_ctx, vocab_size]
        return x
    

class PepBART(nn.Module):
    def __init__(self, device, config,):
        super().__init__()

        self.device = device
        self.aa_encoder = BERT(tokenizer=AATokenizer(max_len=config.aa_max_len), **config.aa_bert)        
        self.smi_decoder = BARTDecoder(tokenizer=SmilesTokenizer(max_len=config.smi_max_len), **config.smi_decoder)
        self.mlm = config.mlm

    def forward(self, x, y):
        aa_seqs, smiles  = x, y

        aa_tokens = self.aa_encoder.tokenize_inputs(aa_seqs).to(self.device)
        if self.mlm:
            mlm_loss = self.aa_encoder.mlm(aa_tokens)

        aa_embd = self.aa_encoder.embed(aa_tokens)  # [batch_size, n_ctx, d_model]
        smi_tokens = self.smi_decoder.tokenize_inputs(smiles).to(self.device)
        smi_inputs = smi_tokens[:, :-1]
        smi_target = smi_tokens[:, 1:]
        smi_pred = self.smi_decoder(smi_inputs, aa_embd)

        lm_loss = F.cross_entropy(smi_pred.view(-1, smi_pred.size(-1)), smi_target.contiguous().view(-1), ignore_index=self.smi_decoder.tokenizer.pad_token_id)
        logger.debug(f'mlm_loss: {mlm_loss}, lm_loss: {lm_loss}')
        loss = mlm_loss + lm_loss if self.mlm else lm_loss
        return loss

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer
    
    def load_pretrained_encoder(self, aa_ckpt=None):
        if aa_ckpt is not None:
            self.aa_encoder = load_model(self.aa_encoder, aa_ckpt, self.device)

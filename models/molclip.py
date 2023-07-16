import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import logging

from .base_transformer import LayerNorm, Transformer
from .bert import BERT
from datasets.tokenizer import SmilesTokenizer, AATokenizer

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


class FusionEncoder(nn.Module):
    def __init__(self,
                 width: int,
                 n_heads: int,
                 n_layers: int,
                 ):
        super().__init__()

        self.transformer = Transformer(
            width=width,
            layers=n_layers,
            heads=n_heads,
            attn_mask=None,
        )
        self.ln_final = LayerNorm(width)
        self.initialize_parameters()

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # BLD -> LBD
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.ln_final(x)   # x.shape = [batch_size, n_ctx, transformer.width]
        return x

def replace_nan_with_zero(loss):
    loss = loss.clone()
    loss[loss != loss] = 0 # replace nan with 0
    return loss

class MolCLIP(nn.Module):
    def __init__(self, 
                 device,
                 config,
                 ):
        super().__init__()
        self.device = device

        self.temp = nn.Parameter(torch.tensor(1.0)) * config.temp_scale
        self.smi_encoder = BERT(tokenizer=SmilesTokenizer(max_len=config.smi_max_len),**config.smi_bert)
        self.smi_proj = nn.Linear(config.smi_bert.width, config.proj_dim)

        self.aa_encoder = BERT(tokenizer=AATokenizer(max_len=config.aa_max_len), **config.aa_bert)
        self.aa_proj = nn.Linear(config.aa_bert.width, config.proj_dim)
        self.smi_fuse_proj = nn.Linear(config.smi_bert.context_length, config.aa_bert.context_length)
        self.fusion_encoder = FusionEncoder(width=config.smi_bert.width + config.aa_bert.width, **config.fusion_encoder)

    def forward(self, x, y):
        smiles, aa_seqs = x, y

        aa_embd, aa_feat, aa_tokens = self._get_embd(aa_seqs, input_type='aa')
        smi_embd, smi_feat, smi_tokens = self._get_embd(smiles, input_type='smi')

        smi_feat = F.normalize(self.smi_proj(smi_feat), dim=-1)
        aa_feat = F.normalize(self.aa_proj(aa_feat), dim=-1)

        #======= Smiles-AA contrastive (SAC) loss =======#
        sim_s2a = torch.mm(smi_feat, aa_feat.T) / self.temp
        sim_a2s = torch.mm(aa_feat, smi_feat.T) / self.temp

        targets = torch.zeros(sim_s2a.size()).to(self.device)
        targets.fill_diagonal_(1)

        loss_s2a = -torch.sum(F.log_softmax(sim_s2a, dim=-1) * targets, dim=-1).mean()
        loss_a2s = -torch.sum(F.log_softmax(sim_a2s, dim=-1) * targets, dim=-1).mean()
        
        loss_sac = (loss_s2a + loss_a2s) / 2
        # loss_sac = replace_nan_with_zero(loss_sac)

        #======= MLM loss (MAM ++ MSM)=======#
        loss_msm = self.smi_encoder.mlm(smi_tokens.clone())
        loss_mam = self.aa_encoder.mlm(aa_tokens.clone())
        loss_mlm = loss_msm + loss_mam
        
        # loss = loss_sac + loss_mlm
        # return loss
        #======= Fusion contrastive (FC) loss =======#
        smi_embd = smi_embd.permute(0, 2, 1)  # BLD -> BDL
        smi_embd = self.smi_fuse_proj(smi_embd)
        smi_embd = smi_embd.permute(0, 2, 1)  # BDL -> BLD

        all_embd = torch.cat([smi_embd, aa_embd], dim=-1)
        all_embd = self.fusion_encoder(all_embd)
        all_feat = F.normalize(all_embd[:, 0, :], dim=-1)
        sim_mol = torch.mm(all_feat, all_feat.T) / self.temp

        loss_sam = -torch.sum(F.log_softmax(sim_mol, dim=-1) * targets, dim=-1).mean()
        # loss_sam = replace_nan_with_zero(loss_sam)
        
        loss = loss_sac + loss_mlm + loss_sam
        return loss
    
    def _get_embd(self, inputs, input_type='smi'):
        encoder = self.smi_encoder if input_type == 'smi' else self.aa_encoder
        tokens = encoder.tokenize_inputs(inputs).to(self.device)
        batch_lens = (tokens != encoder.tokenizer.pad_token_id).sum(1)
        embd = encoder.embed(tokens)
        reps = []
        for i, tokens_len in enumerate(batch_lens):  # Get average of input tokens
            reps.append(embd[i, 1 : tokens_len - 1].mean(0))
        reps = torch.stack(reps)
        return embd, reps, tokens

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer

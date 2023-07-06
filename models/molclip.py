import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from .base_transformer import LayerNorm, Transformer
from .bert import BERT
import esm

torch.autograd.set_detect_anomaly(True)

class MolCLIP(nn.Module):
    def __init__(self, 
                 tokenizer, 
                 device,
                 config
                 ):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(1.0)) * config.temp_scale
        self.smi_encoder = BERT(tokenizer=tokenizer,**config.smi_bert)
        self.aa_encoder, self.aa_tokenizer, self.aa_alphabet = self.load_esm2_model()
        self.device = device

        self.smi_proj = nn.Linear(config.smi_bert.width, config.proj_dim)
        self.aa_proj = nn.Linear(1280, config.proj_dim)
    
    @staticmethod
    def load_esm2_model():  # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        return model, batch_converter, alphabet

    def forward(self, x, y):
        smiles, aa_seqs = x, y

        aa_embd = self.get_esm_embd(aa_seqs)
        smi_embd, smi_tokens = self.get_smi_embd(smiles)

        smi_feat = F.normalize(self.smi_proj(smi_embd), dim=-1)
        aa_feat = F.normalize(self.aa_proj(aa_embd), dim=-1)

        #======= Smiles-AA contrastive loss =======#
        sim_s2a = torch.mm(smi_feat, aa_feat.T) / self.temp
        sim_a2s = torch.mm(aa_feat, smi_feat.T) / self.temp

        targets = torch.zeros(sim_s2a.size()).to(self.device)
        targets.fill_diagonal_(1)

        loss_s2a = -torch.sum(F.log_softmax(sim_s2a, dim=-1) * targets, dim=-1).mean()
        loss_a2s = -torch.sum(F.log_softmax(sim_a2s, dim=-1) * targets, dim=-1).mean()

        loss_sac = (loss_s2a + loss_a2s) / 2

        #======= Smiles MLM loss =======#
        smi_tokens_mlm = smi_tokens.clone()
        loss_mlm = self.smi_encoder.mlm(smi_tokens_mlm)

        #======= Smiles-AA match loss =======#
        # TODO: Implement this later as fusion needed
        # loss_sam = F.cross_entropy(smi_feat, aa_feat.argmax(dim=-1))

        loss = loss_sac + loss_mlm

        return loss
    
    @torch.no_grad()
    def get_esm_embd(self, aa_seqs):
        data = [(f"seq{id}", seq) for id, seq in enumerate(aa_seqs)]
        _, _, batch_tokens = self.aa_tokenizer(data)
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.aa_alphabet.padding_idx).sum(1)

        with torch.no_grad():  # Disable gradient calculation on ESM-2 model
            results = self.aa_encoder(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        sequence_rep = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_rep.append(token_representations[i, 1 : tokens_len - 1].mean(0)) # Take mean of non-pad tokens
        
        return torch.stack(sequence_rep)
    
    def get_smi_embd(self, smiles):
        smi_tokens_raw = self.smi_encoder.tokenizer.batch_encode(smiles).to(self.device)
        smi_tokens = self.smi_encoder.process_inputs(smi_tokens_raw)
        batch_lens = (smi_tokens != self.smi_encoder.tokenizer.pad_token_id).sum(1)
        smi_embd = self.smi_encoder.embed(smi_tokens)
        smi_reps = []
        for i, tokens_len in enumerate(batch_lens):
            smi_reps.append(smi_embd[i, 1 : tokens_len - 1].mean(0))
        
        return torch.stack(smi_reps), smi_tokens

    
    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ) # use simple optimzer first
        return optimizer

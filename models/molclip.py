import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base_transformer import LayerNorm, Transformer
from .smi_bert import SmilesBERT
import esm


class MolCLIP(nn.Module):
    def __init__(self, 
                 tokenizer,
                 config
                 ):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(1.0)) * config.temp_scale
        self.smi_encoder = SmilesBERT(tokenizer=tokenizer,**config.smi_bert)
        self.aa_encoder, self.aa_tokenizer, self.aa_alphabet = self.load_esm2_model()

        self.smi_proj = nn.Linear(config.smi_bert.width, config.proj_dim)
        self.aa_proj = nn.Linear(1280, config.proj_dim)
    
    @staticmethod
    def load_esm2_model():  # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        return model, batch_converter, alphabet

    def forward(self, batch):
        smiles, aa_seqs = batch

        aa_embd = self.get_esm_embd(aa_seqs)
        smi_embd, smi_tokens = self.get_esm_embd(smiles)

        smi_feat = F.normalize(self.smi_proj(smi_embd), dim=-1)
        aa_feat = F.normalize(self.aa_proj(aa_embd), dim=-1)

        #======= Smiles-AA contrastive loss =======#
        sim_s2a = torch.mm(smi_feat, aa_feat.T) / self.temp
        sim_a2s = torch.mm(aa_feat, smi_feat.T) / self.temp

        loss_s2a = -torch.sum(F.log_softmax(sim_s2a, dim=-1) * F.softmax(sim_a2s, dim=-1), dim=-1).mean()
        loss_a2s = -torch.sum(F.log_softmax(sim_a2s, dim=-1) * F.softmax(sim_s2a, dim=-1), dim=-1).mean()

        loss_sac = (loss_s2a + loss_a2s) / 2

        #======= Smiles MLM loss =======#
        loss_mlm = self.smi_encoder.mlm(smi_tokens)

        #======= Smiles-AA match loss =======#
        # TODO: Implement this later as fusion needed
        # loss_sam = F.cross_entropy(smi_feat, aa_feat.argmax(dim=-1))

        return loss_sac, loss_mlm
    

    def get_esm_embd(self, aa_seqs):
        data = [(f"seq{id}", seq) for id, seq in enumerate(aa_seqs)]
        _, _, batch_tokens = self.aa_tokenizer(data)
        batch_lens = (batch_tokens != self.aa_alphabet.padding_idx).sum(1)

        with torch.no_grad():  # Disable gradient calculation on ESM-2 model
            results = self.aa_encoder(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0)) # Take mean of non-pad tokens
        
        return torch.stack(sequence_representations).numpy()
    
    def get_smi_embd(self, smiles):
        smi_tokens = self.smi_encoder.tokenizer.batch_encode(smiles)
        batch_lens = (smi_tokens != self.smi_encoder.tokenizer.pad_token_id).sum(1)
        smi_embd = self.smi_encoder.embed(smi_tokens)
        smi_reps = []
        for i, tokens_len in enumerate(batch_lens):
            smi_reps.append(smi_embd[i, 1 : tokens_len - 1].mean(0))
        
        return torch.stack(smi_reps).numpy(), smi_tokens

        
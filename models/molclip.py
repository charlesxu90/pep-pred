import numpy as np
import torch
from torch import nn

from .base_transformer import LayerNorm, Transformer

class SmiEncoder(nn.Module):
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 width: int,
                 n_heads: int,
                 n_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.transformer = Transformer(
            width=width,
            layers=n_layers,
            heads=n_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = LayerNorm(width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask; fill with -inf for masked positions
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def encode_smiles(self, smiles):
        x = self.token_embedding(smiles).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)   # x.shape = [batch_size, n_ctx, transformer.width]
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), smiles.argmax(dim=-1)]

        return x

    def forward(self, smiles):
        smi_features = self.encode_smiles(smiles)
        # masked_lm_loss, logits, hidden_states, attentions
        return smi_features


class MolCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 width: int,
                 n_heads: int,
                 n_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        self.smi_encoder = SmiEncoder(context_length, vocab_size, width, n_heads, n_layers)
        self.smi_projection = nn.Parameter(torch.empty(width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self.smi_projection is not None:
            nn.init.normal_(self.smi_projection, std=self.smi_encoder.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.smi_encoder.token_embedding.weight.dtype

    def forward(self, input_ids):
        exit()

        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, self.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        mlm_output = self.text_encoder(input_ids, 
                                       labels = labels,  
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )

        loss_mlm = mlm_output.loss  
        return loss_mlm
    
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

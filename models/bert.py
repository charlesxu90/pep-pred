import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_transformer import LayerNorm, Transformer

logger = logging.getLogger(__name__)

class BERT(nn.Module):
    def __init__(self,
                 tokenizer,
                 context_length: int,
                 width: int,
                 n_heads: int,
                 n_layers: int,
                 mlm_probability: float = 0.15,
                 grad_ckpt: bool = False,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.transformer = Transformer(
            width=width,
            layers=n_layers,
            heads=n_heads,
            attn_mask=None,
            grad_ckpt=grad_ckpt,
        )

        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = LayerNorm(width)
        self.mlm_probability = mlm_probability
        self.proj = nn.Linear(width, self.vocab_size)  # decodes into vocab_size

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

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def embed(self, inputs):
        x = self.token_embedding(inputs).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # logger.debug(f'x.shape: {x.shape}')
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # BLD -> LBD
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.ln_final(x).type(self.dtype)   # x.shape = [batch_size, n_ctx, transformer.width]
        
        return x
        
    def mask(self, input_ids, targets):
        probability_matrix = torch.full(targets.shape, self.mlm_probability)  # mask with 0.15 probability

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.begin_token_id] = False
        masked_indices[input_ids == self.tokenizer.end_token_id] = False
        targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10%), we keep the masked input tokens unchanged   
        
        return input_ids, targets, masked_indices

    def mlm(self, inputs):
        targets = inputs.clone()
        mask_inputs, targets, masked_indices = self.mask(inputs, targets=targets)  # mlm

        out_embd = self.embed(mask_inputs)
        outputs = self.proj(out_embd)

        targets_masked = targets[masked_indices]
        outputs_masked = outputs[masked_indices].view(-1, self.vocab_size)

        masked_lm_loss = F.cross_entropy(outputs_masked, targets_masked)
        return masked_lm_loss
    
    def tokenize_inputs(self, inputs):
        """ remove useless dimension, and fill to the context_length
        """
        inputs = self.tokenizer.tokenize(inputs)
        inputs = nn.functional.pad(inputs, (0, self.context_length - inputs.shape[1]), value=self.tokenizer.pad_token_id)  # padding inputs length to n_ctx
        return inputs

    def forward(self, inputs):
        masked_lm_loss = self.mlm(inputs)
        return masked_lm_loss
    
    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer

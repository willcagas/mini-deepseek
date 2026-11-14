import torch
import torch.nn as nn
from src.layers import RMSNorm

# Multi-Token Prediction (MTP)
class MTP(nn.Module):
    def __init__(self, config, block):
        super().__init__()

        model_conf = config["model"]
        rms_conf = config["rms"]

        self.mtp_block = block

        self.hidden_norm = RMSNorm(hidden_size=model_conf["hidden_size"], epsilon=rms_conf["epsilon"])
        self.next_tok_norm = RMSNorm(hidden_size=model_conf["hidden_size"], epsilon=rms_conf["epsilon"])

        # combine hidden features and next token embedding
        self.combine_proj = nn.Linear(in_features=2*model_conf["hidden_size"], out_features=model_conf["hidden_size"], bias=False)

    def forward(self, x, next_tok):
        # normalize each input
        x = self.hidden_norm(x)
        next_tok = self.next_tok_norm(next_tok)

        combined = torch.cat((x, next_tok), dim=-1) # concat features, shape (batch, sequence, 2 * hidden size)
        combined = self.combine_proj(combined) # project back to the model's hidden size

        mtp_hidden, aux_loss = self.mtp_block(combined, use_cache=False) # process through the mtp block w/o caching

        return mtp_hidden, aux_loss
import torch
import torch.nn as nn
from layers import RMSNorm, SwiGLU, RoPE
from attention import MLA
from moe import MoE

class DeepSeekBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        model_conf = config["model"]
        rms_conf = config["rms"]

        self.num_layers = model_conf["num_layers"] 

        self.is_dense = layer_idx < model_conf["num_dense_layers"]

        self.layer_norm_1 = RMSNorm(hidden_size=model_conf["hidden_size"], epsilon=rms_conf["epsilon"])
        self.mha = MLA(config)

        self.layer_norm_2 = RMSNorm(hidden_size=model_conf["hidden_size"], epsilon=rms_conf["epsilon"])

        if self.is_dense:
            self.mlp = SwiGLU(hidden_size=model_conf["hidden_size"], inter_size=model_conf["dense_inter_size"])
        else:
            self.mlp = MoE(config)

    def forward(self, x):
        x = x + self.mha(self.layer_norm_1(x)) # residual skip connection

        if self.is_dense:
            output = self.mlp(self.layer_norm_2(x)) # use dense swiglu mlp
            aux_loss = x.new_zeros(())
        else:
            output, aux_loss = self.mlp(self.layer_norm_2(x)) # use moe

        x = x + output # residual skip connection

        return x, aux_loss

class MiniDeepSeekV3(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_conf = config["model"]

        self.tok_embed_table = nn.Embedding(num_embeddings=model_conf["vocab_size"], embedding_dim=model_conf["hidden_size"])

        # learned weight matrix used to turn final hidden features into logit predictions for each token
        self.lm_head = nn.Linear(
            in_features=model_conf["hidden_size"],
            out_features=model_conf["vocab_size"],
            bias=False,
        )

    # input of token IDs as values, shape (batch size, sequence len)
    def forward(self, x):
        tok_embeddings = self.tok_embed_table(x) # lookup token IDs and return embeddings, shape (batch size, sequence len, hidden size)


        return self.lm_head(tok_embeddings) # temporary of tok embeddings, shape (batch size, sequence len, vocab size)
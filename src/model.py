import torch
import torch.nn as nn

class MiniDeepSeekV3(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tok_embed_table = nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["hidden_size"])

    # input of token IDs as values, shape (batch size, sequence len)
    def forward(self, x):
        tok_embeddings = self.tok_embed_table[x] # lookup embeddings and return shape (batch size, sequence len, hidden features)
    


 
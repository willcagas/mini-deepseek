import torch
import torch.nn as nn
from layers import RMSNorm, RoPE
import torch.nn.functional as F

# Multi-Head Latent Attention (MLA)
class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_conf = config["model"]
        mla_conf = config["mla"]
        rms_conf = config["rms"]

        self.num_heads = mla_conf["num_heads"]
        self.rope_dim = mla_conf["qk_rope_head_dim"]
        self.nope_dim = mla_conf["qk_nope_head_dim"]
        self.value_dim = mla_conf["value_head_dim"]

        # kv compression
        self.kv_proj = nn.Linear(in_features=model_conf["hidden_size"], out_features=mla_conf["kv_latent_size"], bias=False)
        self.kv_norm = RMSNorm(mla_conf["kv_latent_size"], epsilon=rms_conf["epsilon"])

        # query compression
        self.q_proj = nn.Linear(in_features=model_conf["hidden_size"], out_features=mla_conf["query_latent_size"], bias=False)
        self.q_norm = RMSNorm(hidden_size=mla_conf["query_latent_size"], epsilon=rms_conf["epsilon"])

        # positional projections
        self.k_pos_proj = nn.Linear(
            in_features=model_conf["hidden_size"], 
            out_features=mla_conf["qk_rope_head_dim"], 
            bias=False
        ) 
        self.q_pos_proj = nn.Linear(
            in_features=mla_conf["query_latent_size"], 
            out_features=mla_conf["num_heads"]*mla_conf["qk_rope_head_dim"], 
            bias=False
        )

        # rope rotations
        self.rope = RoPE(sequence_len=rms_conf["max_sequence_len"], feature_size=mla_conf["qk_rope_head_dim"])

        # content projections
        self.q_con_proj = nn.Linear(
            in_features=mla_conf["query_latent_size"],
            out_features=mla_conf["num_heads"]*mla_conf["qk_nope_head_dim"],
            bias=False
        )
        self.k_con_proj = nn.Linear(
            in_features=mla_conf["kv_latent_size"],
            out_features=mla_conf["num_heads"]*mla_conf["qk_nope_head_dim"],
            bias=False
        )

        # value projection
        self.v_proj = nn.Linear(
            in_features=mla_conf["kv_latent_size"],
            out_features=mla_conf["value_head_dim"]*mla_conf["num_heads"],
            bias=False
        )

        # output layer
        self.out_proj = nn.Linear(
            in_features=mla_conf["value_head_dim"]*mla_conf["num_heads"],
            out_features=model_conf["hidden_size"],
            bias=False,
        )

        # caches
        self.kv_cache = None
        self.k_pos_cache = None

    def forward(self, x):
        batch_size, sequence_len, hidden_size = x.shape

        kv_compressed = self.kv_norm(self.kv_proj(x)) # compute prefill's kv latents on first pass and each token's latents after

        if self.kv_cache is None:
            self.kv_cache = kv_compressed
        else:
            self.kv_cache = torch.cat((self.kv_cache, kv_compressed), dim=1) # concat existing cached kv latents with new latents from single token

        kv_compressed = self.kv_cache # contains all cached token's kv latents, shape (batch, key sequence, kv latent size)
        q_compressed = self.q_norm(self.q_proj(x))

        k_pos = self.k_pos_proj(x) # shared positional key, shape (batch, key sequence, rope dim)
        q_pos = self.q_pos_proj(q_compressed)
        q_pos = q_pos.reshape(batch_size, sequence_len, self.num_heads, self.rope_dim)

        # count previously cached tokens
        if self.k_pos_cache is None:
            start_pos = 0
        else: 
            start_pos = self.k_pos_cache.size(1) # set token pos for correct rotation angle for rope

        rope_q, rope_k = self.rope(q_pos, k_pos, start_pos=start_pos) # rotate positional features

        if self.k_pos_cache is None:
            self.k_pos_cache = rope_k
        else:
            self.k_pos_cache = torch.cat((self.k_pos_cache, rope_k), dim=1) # concat existing cached rotated pos keys with new rope pos key from single token

        rope_k = self.k_pos_cache # contains all cached rotated pos keys, shape (batch, key sequence, rope dim)

        q_con = self.q_con_proj(q_compressed) 
        q_con = q_con.reshape(batch_size, sequence_len, self.num_heads, self.nope_dim) # shape (batch, sequence, heads, nope dim)

        k_con_weights = self.k_con_proj.weight
        k_con_weights = k_con_weights.reshape(self.num_heads, self.nope_dim, k_con_weights.shape[-1]) # shape (heads, nope dim, kv latent size)

        q_absorbed = (q_con.unsqueeze(-2) @ k_con_weights).squeeze(-2) # shape (batch, q sequence, heads, kv latent size)
        
        q_con = q_absorbed.transpose(1, 2) # shape (batch, heads, q sequence, kv latent size)
        k_con_t = kv_compressed.transpose(1, 2)  # shape (batch, kv latent size, k sequence)
        k_con_t = k_con_t.unsqueeze(1) # shape (batch, 1, kv latent size, k sequence)

        con_scores = q_con @ k_con_t # absorbed MLA content scores for Q @ K^T, shape (batch, heads, q sequence, k sequence)

        q_pos = rope_q.transpose(1, 2)
        k_pos_t = rope_k.transpose(1, 2).unsqueeze(1)

        pos_scores = q_pos @ k_pos_t # positional scores for each query key pairs

        # combine content and positional scores
        scores = con_scores + pos_scores
        scores = scores / ((self.nope_dim + self.rope_dim) ** 0.5) # scale by the root of full query/key size per head, shape (batch, heads, q sequence, k sequence)

        mask = torch.triu(torch.ones(sequence_len, kv_compressed.size(1), device=x.device, dtype=torch.bool), diagonal=start_pos+1) 
        scores = scores.masked_fill(mask, float("-inf")) # causal mask so no attention to future tokens

        # normalize key positions for every query and head
        att_weights = F.softmax(scores, dim=-1)

        # combine shared KV latents using every head's attention weights
        weighted_latents = att_weights @ kv_compressed.unsqueeze(1)

        # give each head a value proj matrix
        v_weights = self.v_proj.weight.reshape(self.num_heads, self.value_dim, -1)
        v_weights = v_weights.transpose(-2, -1) # shape (heads, kv latent size, value head dim)

        head_outputs = weighted_latents @ v_weights # shape (batch, heads, q sequence, value head dim)

        # put heads beside their features and concatenate outputs
        combined = head_outputs.transpose(1, 2).reshape(
            batch_size, sequence_len, self.num_heads * self.value_dim
        )

        return self.out_proj(combined)
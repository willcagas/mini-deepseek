import torch
import torch.nn as nn
import torch.nn.functional as F

# Root Mean Square Layer Normalization (RMSNorm)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, epsilon):
        super().__init__()

        self.epsilon = epsilon # fixed hyperparameter
        self.gamma = nn.Parameter(torch.ones(hidden_size)) # learned parameter, 1D tensor with size = number of features

    def forward(self, x):
        original_dtype = x.dtype
        x = x.float() # convert to fp32

        # calculate rms
        squares = x.pow(2)
        mean = squares.mean(dim=-1, keepdim=True) # tensor shape goes from (x,y,z) to (x,y,1)
        root = (mean + self.epsilon).sqrt() # add epsilon to prevent div by 0 when calculating norm
        norm = x.div(root) # divide each token's features by its RMS using broadcasting
        
        rms_norm = norm * self.gamma # one learned "scale" per feature and each shared across tokens

        return rms_norm.to(original_dtype)


# Swish Gated Linear Unit (SwiGLU)
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, inter_size):
        super().__init__()

        self.gate_linear = nn.Linear(in_features=hidden_size, out_features=inter_size, bias=False) # linear transformation using "W" parameter
        self.up_linear = nn.Linear(in_features=hidden_size, out_features=inter_size, bias=False) # linear transformation using "V" parameter
        self.down_linear = nn.Linear(in_features=inter_size, out_features=hidden_size, bias=False) # technically not SwiGLU, but part of the FF block as a whole

    def forward(self, x):
        gate_proj = self.gate_linear(x) # matrix multiplication --> x @ W
        swish_gate = gate_proj * F.silu(gate_proj) # apply swish (sigmoid linear unit)

        up_proj = self.up_linear(x) # matrix multiplication --> x @ V

        swi_glu = self.down_linear(swish_gate * up_proj) # final linear layer, combined SwiGLU @ W_out 

        return swi_glu


class RoPE(nn.Module):
    def __init__(self, sequence_len, feature_size):
        super().__init__()  

        pair_indices = torch.arange(start=0,end=feature_size//2,step=1) # feature-pair indices
        self.rotation_rates = 1 / (10000 ** (2*pair_indices / feature_size)) # constant rotation rate for each feature-pair, shared across every query and key vector

        token_pos = torch.arange(start=0,end=sequence_len,step=1).view(sequence_len, 1) # create 1D tensor then reshape to 2D tensor with dim 0 size of sequence len
        self.rotation_angles = token_pos * self.rotation_rates # angle = token pos x rotation rate

    def rotate(self, x):
        original_dtype = x.dtype
        x = x.float() # convert to fp32
        
        cos = self.rotation_angles.cos()
        sin = self.rotation_angles.sin()

        x = x.view(x.size(0), x.size(1), x.size(2) // 2, 2) # turn into 4D tensor with shape (batch_size, sequence_len, num_pairs, 2)
        
        a = x[:, :, :, 0] # first value of every pair
        b = x[:, :, :, 1] # second value of every pair

        # apply rotation linear transformation
        rotated_a = a*cos - b*sin 
        rotated_b = a*sin + b*cos

        x = torch.stack((rotated_a, rotated_b), dim=-1) # combine rotated values from a and b (respecitively) into pairs along the last dimension
        x = x.view(x.size(0), x.size(1), x.size(2) * 2)

        return x.to(original_dtype)

    def forward(self, query, key):
        return self.rotate(query), self.rotate(key)
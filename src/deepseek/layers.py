import torch
import torch.nn as nn

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
        sigmoid = nn.Sigmoid()

        gate_proj = self.gate_linear(x) # matrix multiplication --> x @ W
        swish_gate = gate_proj * sigmoid(gate_proj) # apply swish (sigmoid linear unit)

        up_proj = self.up_linear(x) # matrix multiplication --> x @ V

        swi_glu = self.down_linear(swish_gate * up_proj) # final linear layer, combined SwiGLU @ W_out 

        return swi_glu


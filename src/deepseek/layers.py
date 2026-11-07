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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import RMSNorm, SwiGLU

# Mixture of Experts (MoE)
class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_conf = config["model"]
        moe_conf = config["moe"]

        self.num_routed_experts = moe_conf["routed_experts"]
        self.experts_per_token = moe_conf["experts_per_token"]
        self.gamma = moe_conf["bias_update_gamma"]
        self.alpha = moe_conf["aux_loss_alpha"]
        
        self.router = nn.Linear(in_features=model_conf["hidden_size"], out_features=moe_conf["routed_experts"], bias=False)

        router_bias = torch.zeros(moe_conf["routed_experts"]) # create router bias heuristic, not a learned parameter updated by optimizer
        self.register_buffer("router_bias", router_bias) # register as buffer

        # for auxiliary loss-free load balancing
        batch_expert_counts = torch.zeros(moe_conf["routed_experts"])
        self.register_buffer("batch_expert_counts", batch_expert_counts, persistent=False) # register as buffer

        # list of routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLU(hidden_size=model_conf["hidden_size"], inter_size=moe_conf["routed_expert_inter_size"]) 
            for _ in range(moe_conf["routed_experts"])
        ])

        self.shared_expert = SwiGLU(hidden_size=model_conf["hidden_size"], inter_size=moe_conf["shared_expert_inter_size"]) 

    @torch.no_grad()
    def update_router_bias(self):
        target_load = self.batch_expert_counts.sum() / self.num_routed_experts # calc target load to balance expert routing of tokens

        self.router_bias.add_(self.gamma * torch.sign(target_load - self.batch_expert_counts))
        self.batch_expert_counts.zero_()

    def forward(self, x):
        batch, tokens, hidden_size = x.shape

        seq_expert_counts = torch.zeros(batch, self.num_routed_experts, device=x.device)

        # put router projection and sigmoid in fp32 for numerical stability
        with torch.autocast(device_type=x.device.type, enabled=False):
            routed_logits = self.router(x.float())
            routed_scores = F.sigmoid(routed_logits) # compute affinity score for each token and expert pair and go through squishification function

        selection_scores = routed_scores + self.router_bias

        indices = torch.topk(selection_scores, k=self.experts_per_token, dim=-1).indices
        aux_indices = torch.topk(routed_scores, k=self.experts_per_token, dim=-1).indices # indices for sequence wise aux loss

        flat_x = x.reshape(-1, x.size(-1)) # reshape input tensor to have every token in each row and have features in columns, shape (batch_size x sequence len, hidden size)
        flat_indices = indices.reshape(-1, self.experts_per_token) # reshape indices so that every row is a token and each column is a top expert of k slots
        
        expert_weights = routed_scores.gather(dim=-1, index=indices) # for each token, retrieve each routed score and corresponding top k experts, shape (batch, sequence, top k)
        norm_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True) # for each token, normalize by dividing each score by sum of scores of top k experts
        flat_weights = norm_weights.reshape(-1, self.experts_per_token) # flatten so its only rows of tokens and columns of experts and values are weights

        accumulated_outputs = torch.zeros_like(flat_x) # create a tensor of zeros with flattened shape of flat_x for rows of every token (not batched)

        for expert_id, routed_expert in enumerate(self.routed_experts):
            tok_indices, selection_slots = torch.where(flat_indices == expert_id) # for each expert, find each token that routed through current indexed expert and which slot was picked

            expert_inputs = flat_x[tok_indices] # index the specific tokens corresponding to this current expert
            expert_outputs = routed_expert(expert_inputs) # shape (tokens for specific expert, hidden size)

            weights = flat_weights[tok_indices, selection_slots] # grab every weight corresponding to the token (row) and top k expert slot (column), shape 1D tensor (tokens,)
            weighted_outputs = expert_outputs * weights.unsqueeze(-1) # broadcast, for each token, multiply each feature in that token's row by the weight for that row

            accumulated_outputs[tok_indices] += weighted_outputs # accumulate outputs for every routed expert

            if self.training:
                self.batch_expert_counts[expert_id] += tok_indices.numel() # update expert counts for load balancing using router bias only if in training

            expert_selected = aux_indices == expert_id 
            counts_per_sequence = expert_selected.sum(dim=(1, 2)) # for each sequence, count how many times expert selected
            seq_expert_counts[:, expert_id] = counts_per_sequence # update separate expert counts for sequence wise aux loss

        routed_output = accumulated_outputs.reshape(batch, tokens, hidden_size)
        shared_output = self.shared_expert(x) 

        full_output = routed_output + shared_output # combine routed experts and shared expert(s) output

        relative_usage = seq_expert_counts * self.num_routed_experts / (tokens * self.experts_per_token)
        normalized_mean_scores = (routed_scores / routed_scores.sum(dim=-1, keepdim=True)).mean(dim=1)
        aux_loss = self.alpha * (relative_usage * normalized_mean_scores).sum(dim=-1).mean() # alpha controls how much the penalty contributes to training, higher aux loss means imbalance in expert contributors
        
        return full_output, aux_loss








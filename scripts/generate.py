import torch

from transformers import AutoTokenizer
from src.config import config
from src.model import MiniDeepSeekV3

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

def generate_text(prompt, max_tokens=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniDeepSeekV3(config=config).to(device)
    model.eval()

    tok_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        logits = model(tok_ids) # prefill call, shape (1, sequence, vocab size)

        # autoregressive decoding
        for step in range(max_tokens):
            last_logits = logits[:, -1, :] # take the last token's (in the sequence) vocab logits to predict the next token
            next_tok = last_logits.argmax(dim=-1, keepdim=True) # greedy token decoding, shape (batch, 1)

            # check for EOS token
            if next_tok.item() == tokenizer.eos_token_id:
                break

            tok_ids = torch.cat((tok_ids, next_tok), dim=1) # concatenate to add next token to tensor of token ids

            if step + 1 < max_tokens:
                logits = model(tok_ids) # recompute with the new token included
 
    output = tokenizer.decode(tok_ids[0].tolist(), skip_special_tokens=True) # detokenize IDs into final text output

    return output

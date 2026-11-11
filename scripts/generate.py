import torch

from transformers import AutoTokenizer
from src.config import config
from src.model import DeepSeekV3

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

def generate_text(prompt, max_tokens=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepSeekV3(config=config).to(device)
    model.eval()

    tok_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        # prefill, select the last prompt token's scores
        logits = model(tok_ids)
        next_logits = logits[:, -1, :] # shape (batch, vocab size)

        # decode, pass next token until EOS token or max tokens
        for step in range(max_tokens):
            next_tok = next_logits.argmax(dim=-1, keepdim=True)

            if next_tok.item() == tokenizer.eos_token_id:
                break

            tok_ids = torch.cat((tok_ids, next_tok), dim=1)

            if step + 1 < max_tokens:
                logits = model(next_tok) # cached decode, output is one token with scores, shape (batch, 1, vocab size)
                next_logits = logits.squeeze(1) # shape (batch, vocab_size)
 
    output = tokenizer.decode(tok_ids[0].tolist(), skip_special_tokens=True) # detokenize IDs into final text output

    return output
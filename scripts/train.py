import torch
import torch.nn.functional as F

from src.config import config
from pathlib import Path
from datatrove.utils.dataset import DatatroveFolderDataset
from torch.utils.data import DataLoader
from src.model import DeepSeekV3

def create_data_loader(config, split):
    data_conf = config["data"]
    training_conf = config["training"]

    data_dir = Path(data_conf["output_dir"]) / split

    # read bytes per token from shard's metadata
    metadata_path = next(data_dir.glob("*_doc_shuffled.ds.metadata"))
    metadata = metadata_path.read_text().splitlines()

    token_size = int(metadata[0].rsplit("|", 1)[1])

    dataset = DatatroveFolderDataset(
        data_folder=str(Path(data_conf["output_dir"]) / split),
        filename_pattern="*_doc_shuffled.ds",
        seq_len=training_conf["sequence_len"],
        token_size=token_size,
        return_positions=False,
    )

    return DataLoader(
        dataset,
        batch_size=training_conf["batch_size"],
        shuffle=False, # false because docs were already shuffled during data preprocessing
        num_workers=0,
    )


def train(max_steps=None):
    training_conf = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_data_loader(config=config, split="train")
    val_loader = create_data_loader(config=config, split="validation")

    model = DeepSeekV3(config=config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=training_conf["learning_rate"])

    for step, batch in enumerate(train_loader):
        tokens = batch["input_ids"].to(device)

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer.zero_grad()

        logits, aux_loss = model(inputs, use_cache=False) # do not use kv cache during training

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) + aux_loss # calculate full loss by sequence wise aux loss 
        loss.backward()

        optimizer.step()

        # update aux loss biases and reset expert counts
        for block in model.deepseek_blocks:
            if not block.is_dense:
                block.mlp.update_router_bias()

        print(
            f"prediction loss: {loss.item():.4f}, "
            f"aux loss: {aux_loss.item():.6f}"
        )

        if max_steps is not None and step + 1 >= max_steps:
            break
    
if __name__ == "__main__":
    train()
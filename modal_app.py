import modal

from pathlib import Path

app = modal.App("mini-deepseek-v3")

# python env on modal
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "datatrove[processing]",
        "datasets",
        "transformers",
        "pyyaml",
        "torch"
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("configs", remote_path="/root/configs")
)

# persistent storage for prepared data and checkpoints
volume = modal.Volume.from_name("mini-deepseek-v3-data", create_if_missing=True, version=2)

@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=3600)
def prepare_data():
    from src.config import config
    from scripts.prepare_data import main

    # save preprocessing output inside persistent storage
    config["data"]["output_dir"] = str(Path("/data") / Path(config["data"]["output_dir"]).name)

    main()

    volume.commit()

@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192)
def test_data_loaders():
    import torch
    from src.config import config
    from scripts.train import create_data_loader

    config["data"]["output_dir"] = str(
        Path("/data") / Path(config["data"]["output_dir"]).name
    )

    sequence_len = config["training"]["sequence_len"]
    vocab_size = config["model"]["vocab_size"]

    for split in ("train", "validation"):
        loader = create_data_loader(config, split)
        assert len(loader.dataset) > 0, f"No full sequences in {split}"

        batches = 0

        for batch in loader:
            tokens = batch["input_ids"]

            assert tokens.dtype == torch.long
            assert tokens.ndim == 2
            assert tokens.size(1) == sequence_len + 1
            assert tokens.min().item() >= 0
            assert tokens.max().item() < vocab_size

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            assert inputs.shape == targets.shape
            batches += 1

        print(f"{split}: {batches} batches checked successfully")

@app.function(image=image, volumes={"/data": volume}, gpu="A100-80GB", cpu=4, memory=16384, timeout=1800)
def test_training():
    import torch
    
    from src.config import config
    from scripts.train import train

    assert torch.cuda.is_available(), "Training test requires a GPU."

    # point to the existing tokenized data on the Volume
    config["data"]["output_dir"] = str(
        Path("/data") / Path(config["data"]["output_dir"]).name
    )

    train(max_steps=3)

@app.local_entrypoint()
def main():
    prepare_data.remote()
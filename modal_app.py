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
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("configs", remote_path="/root/configs")
)

# persistent storage for prepared data and checkpoints
volume = modal.Volume.from_name(
    "mini-deepseek-v3-data",
    create_if_missing=True,
    version=2,
)

@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=8192, timeout=3600)
def prepare_data():
    from src.config import config
    from scripts.prepare_data import main

    # save preprocessing output inside persistent storage
    config["data"]["output_dir"] = str(Path("/data") / Path(config["data"]["output_dir"]).name)

    main()

    volume.commit()


@app.local_entrypoint()
def main():
    prepare_data.remote()
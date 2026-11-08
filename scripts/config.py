from pathlib import Path
import yaml

config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

with config_path.open() as file:
    config = yaml.safe_load(file)
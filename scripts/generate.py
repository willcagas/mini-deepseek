from transformers import AutoTokenizer
from src.config import config

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
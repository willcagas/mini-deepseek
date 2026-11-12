import hashlib

from pathlib import Path
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from transformers import AutoTokenizer
from src.config import config

class PrepareData():
    def __init__(self, config, split, output_dir, eos_token):
        data_conf = config["data"]

        self.split = split
        self.seed = data_conf["seed"]
        self.validation_fraction = data_conf["validation_fraction"]

        # turn dataset rows and turn into document objects w/ id, text, and metadata
        self.reader = HuggingFaceDatasetReader(
            dataset=data_conf["dataset"],
            dataset_options={
                "name": data_conf["subset"],
                "split": "train",
                "columns": ["id", "text"],
            },
            text_key="text",
            id_key="id",
            streaming=True,
            limit=data_conf["documents_per_task"],
        )

        # prep documents to preprocessing batches (not training batch) and convert each into token IDs and adds EOS tokens
        self.tokenizer = DocumentTokenizer(
            output_folder=str(output_dir / split),
            tokenizer_name_or_path=config["tokenizer_name"],
            eos_token=eos_token,
            batch_size=data_conf["tokenization_batch_size"],
            max_tokens_per_file=data_conf["tokens_per_shard"],
            shuffle_documents=True,
            seed=self.seed,
        )

        self.num_tasks = data_conf["num_tasks"]
        self.num_workers = data_conf["num_workers"]
        self.logging_dir = str(output_dir / "logs" / split)

    def select_documents(self, data, rank=0, world_size=1):
        # split documents into either train or val splits, before going thruogh tokenizer

        for document in data:
            # assign identical text to the same split every time
            document_text = f"{self.seed}:{document.text}"
            document_hash = hashlib.sha256(document_text.encode("utf-8"))

            # convert part of the hash into a number between 0 and 1
            hash_value = int.from_bytes(document_hash.digest()[:8], "big")
            split_value = hash_value / (2**64)

            is_validation = split_value < self.validation_fraction

            if self.split == "validation":
                if is_validation:
                    yield document
            else:
                if not is_validation:
                    yield document

    def run(self):
        executor = LocalPipelineExecutor(
            pipeline=[
                self.reader,
                self.select_documents,
                self.tokenizer,
            ],
            logging_dir=self.logging_dir,
            tasks=self.num_tasks,
            workers=self.num_workers,
        )

        executor.run()


def main():
    data_conf = config["data"]

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must have an EOS token.")

    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / data_conf["output_dir"]

    # avoid mixing a new preparation run with existing files
    output_dir.mkdir(parents=True, exist_ok=False)

    for split in ("train", "validation"):
        preparation = PrepareData(
            config=config,
            split=split,
            output_dir=output_dir,
            eos_token=tokenizer.eos_token,
        )
        preparation.run()


if __name__ == "__main__":
    main()
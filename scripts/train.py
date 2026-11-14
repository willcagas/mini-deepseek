import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import csv

from src.config import config
from pathlib import Path
from datatrove.utils.dataset import DatatroveFolderDataset
from torch.utils.data import DataLoader
from src.model import DeepSeekV3
from transformers import get_wsd_schedule


def create_data_loader(config, split):
    data_conf = config["data"]
    training_conf = config["training"]
    data_dir = Path(data_conf["output_dir"]) / split

    # read bytes per token from shard's metadata
    metadata_path = next(data_dir.glob("*_doc_shuffled.ds.metadata"))
    metadata = metadata_path.read_text().splitlines()
    token_size = int(metadata[0].rsplit("|", 1)[1])

    dataset = DatatroveFolderDataset(
        data_folder=str(data_dir),
        filename_pattern="*_doc_shuffled.ds",
        seq_len=training_conf["sequence_len"],
        token_size=token_size,
        return_positions=False
    )

    return DataLoader(
        dataset,
        batch_size=training_conf["batch_size"],
        shuffle=False, # false because docs were already shuffled during data preprocessing
        num_workers=0
    )


@torch.no_grad()
def evaluate(model, val_loader, device, use_bf16=False, max_batches=None):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(val_loader):
        tokens = batch["input_ids"].to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits, _ = model(inputs, use_cache=False)

            # sum token losses so smaller batches receive the correct weight
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")

        total_loss += loss.item()
        total_tokens += targets.numel()

        if max_batches is not None and batch_idx + 1 == max_batches:
            break

    model.train(was_training)

    # report pred loss w/o aux balancing loss
    return total_loss / total_tokens


def save_checkpoint(path, model, optimizer, scheduler, step, epoch, next_batch, best_val_loss, config, on_checkpoint=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "next_batch": next_batch,
        "best_val_loss": best_val_loss,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )
    }

    # finish writing before replacing previous checkpoint
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)

    # modal can use this callback to commit the volume
    if on_checkpoint is not None:
        on_checkpoint()


def train(max_steps=None, resume_from=None, on_checkpoint=None):
    training_conf = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")

        memory = torch.cuda.get_device_properties(device).total_memory

        print(f"gpu memory: {memory / 1024**3:.1f} GiB")

    use_bf16 = False

    # use bf16 mixed precision
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        use_bf16 = True

    print(f"device: {device}")
    print(f"precision: {'BF16 mixed precision' if use_bf16 else 'FP32'}")

    torch.manual_seed(config["data"]["seed"])

    train_loader = create_data_loader(config=config, split="train")
    val_loader = create_data_loader(config=config, split="validation")

    accumulation_steps = training_conf["gradient_accumulation_steps"]
    total_steps = training_conf["total_steps"]

    stop_step = total_steps

    if max_steps is not None:
        stop_step = min(max_steps, total_steps)

    model = DeepSeekV3(config=config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_conf["learning_rate"],
        betas=(training_conf["beta1"], training_conf["beta2"]),
        weight_decay=training_conf["weight_decay"]
    )

    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=training_conf["warmup_steps"],
        num_stable_steps=(
            training_conf["decay_start_step"]
            - training_conf["warmup_steps"]
        ),
        num_decay_steps=(
            training_conf["total_steps"]
            - training_conf["decay_start_step"]
        ),
        warmup_type="linear",
        decay_type="cosine",
        min_lr_ratio=(
            training_conf["min_learning_rate"]
            / training_conf["learning_rate"]
        )
    )

    checkpoint_dir = Path(training_conf["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    epoch = 0
    next_batch = 0
    best_val_loss = float("inf")
    checkpoint = None

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        next_batch = checkpoint["next_batch"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming after optimizer step {step}")

    # restore next batch to read within this epoch
    train_iterator = iter(train_loader)

    for _ in range(next_batch):
        next(train_iterator)

    if checkpoint is not None:
        torch.set_rng_state(checkpoint["torch_rng"])

        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        del checkpoint

    metrics_path = checkpoint_dir / "metrics.csv"

    # start fresh history for new run, keep existing history when resuming
    if resume_from is None or not metrics_path.exists():
        with metrics_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "step",
                "prediction_loss",
                "aux_loss",
                "mtp_loss",
                "validation_loss",
                "learning_rate",
                "gradient_norm",
                "tokens_per_update"
            ])

    while step < stop_step:
        optimizer.zero_grad(set_to_none=True)

        update_tokens = 0
        prediction_sum = 0.0
        auxiliary_sum = 0.0
        mtp_sum = 0.0

        # accumulate gradients before making optimizer update
        for _ in range(accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                epoch += 1
                next_batch = 0
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            next_batch += 1

            tokens = batch["input_ids"].to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16
            ):
                logits, aux_loss, mtp_logits = model(inputs, use_cache=False) # do not use kv cache during training

                prediction_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, mtp_logits.size(-1)), targets[:, 1:].reshape(-1))

                loss = prediction_loss + aux_loss + training_conf["mtp_loss_weight"]*mtp_loss # calculate full loss by sequence wise aux loss 

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite training loss before step {step + 1}"
                )

            # weight each batch by its token count, including smaller batches
            num_tokens = targets.numel()
            (loss * num_tokens).backward()

            update_tokens += num_tokens
            prediction_sum += prediction_loss.item() * num_tokens
            auxiliary_sum += aux_loss.item() * num_tokens
            mtp_sum += mtp_loss.item() * num_tokens

            del logits, mtp_logits, prediction_loss, mtp_loss, aux_loss, loss 

        # avg accumulated gradients over all tokens in update
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(update_tokens)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_conf["grad_clip"], error_if_nonfinite=True)

        # record learning rate used for optimizer update
        learning_rate = optimizer.param_groups[0]["lr"]

        optimizer.step()
        scheduler.step()
        step += 1

        # update aux loss biases and reset expert counts
        for block in model.deepseek_blocks:
            if not block.is_dense:
                block.mlp.update_router_bias()

        model.mtp.mtp_block.mlp.update_router_bias() # update mtp block's router bias too

        print(
            f"Step {step} | "
            f"prediction loss: {prediction_sum / update_tokens:.4f}, "
            f"aux loss: {auxiliary_sum / update_tokens:.6f}, "
            f"mtp loss: {mtp_sum / update_tokens:.4f}, "
            f"learning rate: {learning_rate:.2e}, "
            f"gradient norm: {grad_norm.item():.4f}"
        )

        is_final_step = step == stop_step

        val_loss = None # no validation measurement unless we evaluate this step

        if step % training_conf["eval_interval"] == 0 or is_final_step:
            if step == training_conf["total_steps"]:
                max_batches = None

            val_loss = evaluate(model, val_loader, device, use_bf16=use_bf16, max_batches=training_conf["eval_batches"])

            print(f"Validation loss: {val_loss:.4f}")

            if not math.isfinite(val_loss):
                raise RuntimeError("Non-finite validation loss.")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, step, epoch, next_batch, best_val_loss, config, on_checkpoint)

        # save one row per optimizer update
        with metrics_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                step,
                prediction_sum / update_tokens,
                auxiliary_sum / update_tokens,
                mtp_sum / update_tokens,
                val_loss,
                learning_rate,
                grad_norm.item(),
                update_tokens
            ])

        if step % training_conf["checkpoint_interval"] == 0 or is_final_step:
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scheduler, step, epoch, next_batch, best_val_loss, config, on_checkpoint)

if __name__ == "__main__":
    train()
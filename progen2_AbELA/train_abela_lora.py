import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from progen.abela import LINKER, load_abela_records
from progen.lora import (
    apply_qv_lora,
    mark_only_lora_trainable,
    save_lora_adapter,
    trainable_parameter_count,
)
from progen.runtime import autocast_for, create_grad_scaler, load_progen_model, select_device, str_to_bool
from progen.tokenization import load_tokenizer


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed():
    return "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    if not is_distributed():
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


class SerializedDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=1024, drop_long=False, private_log=True):
        self.examples = []
        dropped = 0
        for record in records:
            token_ids = tokenizer.encode(record.serialized).ids
            if len(token_ids) > max_length:
                if drop_long:
                    dropped += 1
                    continue
                message = f"record exceeds --max-length {max_length}; use --drop-long or increase --max-length"
                if not private_log:
                    message = f"{record.record_id} has {len(token_ids)} tokens, exceeding --max-length {max_length}"
                raise ValueError(message)
            self.examples.append(
                {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "length": len(token_ids),
                }
            )
        self.dropped = dropped

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id):
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for row, item in enumerate(batch):
        ids = item["input_ids"]
        length = ids.numel()
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1
        labels[row, :length] = ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def split_indices(size, validation_fraction, seed):
    indices = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(round(size * validation_fraction))
    if size > 1 and validation_fraction > 0:
        val_size = max(1, min(size - 1, val_size))
    else:
        val_size = 0
    return indices[val_size:], indices[:val_size]


def reduce_sums(loss_sum, token_sum, device):
    if not (dist.is_available() and dist.is_initialized()):
        return loss_sum, token_sum
    values = torch.tensor([loss_sum, token_sum], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values[0].item(), values[1].item()


def evaluate(model, loader, device, fp16):
    model.eval()
    loss_sum = 0.0
    token_sum = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_for(device, fp16):
                loss = model(**batch).loss
            tokens = (batch["labels"][:, 1:] != -100).sum().item()
            loss_sum += float(loss.item()) * tokens
            token_sum += tokens

    loss_sum, token_sum = reduce_sums(loss_sum, token_sum, device)
    model.train()
    if token_sum == 0:
        return None
    loss = loss_sum / token_sum
    return loss, math.exp(min(20.0, loss))


def sanitized_metadata(args, checkpoint_dir=None, output_dir=None, extra=None):
    metadata = vars(args).copy()
    for key, placeholder in (
        ("dataset", "<dataset>"),
        ("checkpoint_dir", "<checkpoint-dir>"),
        ("tokenizer", "<tokenizer>"),
        ("output_dir", "<output-dir>"),
    ):
        if key in metadata and metadata[key] is not None:
            metadata[key] = placeholder
    if checkpoint_dir is not None:
        metadata["checkpoint_dir"] = "<checkpoint-dir>"
    if output_dir is not None:
        metadata["output_dir"] = "<output-dir>"
    if extra:
        metadata.update(extra)
    return metadata


def dataset_metadata(args, records, dataset, train_dataset, val_dataset):
    if args.private_log:
        return {"data_summary": "redacted"}
    return {
        "records": len(records),
        "used_records": len(dataset),
        "dropped_long": dataset.dropped,
        "train_records": len(train_dataset),
        "val_records": len(val_dataset) if val_dataset else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ProGen2-OAS on epitope-conditioned AbELA records.")
    parser.add_argument("--dataset", required=True, help="AbELA CSV/TSV/JSON/JSONL file.")
    parser.add_argument("--model", default="progen2-oas")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epitope-column", default=None)
    parser.add_argument("--vh-column", default=None)
    parser.add_argument("--vl-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--active-column", default=None)
    parser.add_argument("--active-only", type=str_to_bool, default=True)
    parser.add_argument("--active-values", default="AbELA-Q,active,positive,binder,true,1,yes")
    parser.add_argument("--ec50-column", default=None)
    parser.add_argument("--max-ec50", type=float, default=None)
    parser.add_argument("--cdr-index-base", type=int, default=1)
    parser.add_argument("--skip-invalid-records", type=str_to_bool, default=True)
    parser.add_argument(
        "--private-log",
        type=str_to_bool,
        default=True,
        help="Redact dataset sizes, sequence-derived statistics, and private metrics in logs and adapter metadata.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--drop-long", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda device. Ignored under torchrun DDP.")
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument(
        "--save-split-indices",
        type=str_to_bool,
        default=False,
        help="Write train/validation row indices. Defaults off because indices reveal dataset size.",
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)
    local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    try:
        device = select_device(args.device, local_rank=local_rank, world_size=world_size)
        if device.type != "cuda":
            args.fp16 = False

        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required")
        checkpoint_dir = args.checkpoint_dir
        tokenizer = load_tokenizer(args.tokenizer)
        pad_token_id = tokenizer.encode("<|pad|>").ids[0]

        records = load_abela_records(
            args.dataset,
            epitope_column=args.epitope_column,
            vh_column=args.vh_column,
            vl_column=args.vl_column,
            id_column=args.id_column,
            target_column=args.target_column,
            active_column=args.active_column,
            active_only=args.active_only,
            active_values=args.active_values,
            ec50_column=args.ec50_column,
            max_ec50=args.max_ec50,
            cdr_index_base=args.cdr_index_base,
            skip_invalid=args.skip_invalid_records,
        )
        if args.limit_records is not None:
            records = records[: args.limit_records]
        if not records:
            raise ValueError("no AbELA records were loaded")

        dataset = SerializedDataset(
            records,
            tokenizer,
            max_length=args.max_length,
            drop_long=args.drop_long,
            private_log=args.private_log,
        )
        if len(dataset) == 0:
            raise ValueError("all records were dropped by --max-length")

        train_indices, val_indices = split_indices(len(dataset), args.validation_fraction, args.seed)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices) if val_indices else None

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 and val_dataset else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: collate_batch(batch, pad_token_id),
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=lambda batch: collate_batch(batch, pad_token_id),
            )
            if val_dataset
            else None
        )

        model = load_progen_model(checkpoint_dir, fp16=args.fp16)
        model.config.use_cache = False
        model.config.gradient_checkpointing = args.gradient_checkpointing
        apply_qv_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        mark_only_lora_trainable(model)
        model.to(device)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        trainable, total = trainable_parameter_count(unwrap(model))
        if is_main:
            if args.private_log:
                print("data_summary=redacted")
            else:
                lengths = sorted(example["length"] for example in dataset.examples)
                mid = lengths[len(lengths) // 2]
                print(f"loaded_records={len(records)} used_records={len(dataset)} dropped_long={dataset.dropped}")
                print(f"serialized_length_min={lengths[0]} median={mid} max={lengths[-1]}")
                print(f"train_records={len(train_dataset)} val_records={len(val_dataset) if val_dataset else 0}")
            print(f"trainable_parameters={trainable} total_parameters={total}")

        optimizer = torch.optim.AdamW(
            [parameter for parameter in unwrap(model).parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scaler = create_grad_scaler(device, args.fp16)

        best_val = float("inf")
        bad_epochs = 0
        global_step = 0
        output_dir = Path(args.output_dir)

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_token_sum = 0
            started = time.time()

            for step, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast_for(device, args.fp16):
                    loss = model(**batch).loss / args.grad_accum_steps

                scaler.scale(loss).backward()
                tokens = (batch["labels"][:, 1:] != -100).sum().item()
                epoch_loss_sum += float(loss.item()) * args.grad_accum_steps * tokens
                epoch_token_sum += tokens

                if step % args.grad_accum_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            train_loss_sum, train_token_sum = reduce_sums(epoch_loss_sum, epoch_token_sum, device)
            train_loss = train_loss_sum / max(1, train_token_sum)

            val_result = evaluate(model, val_loader, device, args.fp16) if val_loader is not None else None
            if is_main:
                elapsed = time.time() - started
                if args.private_log:
                    print(f"epoch={epoch} completed elapsed_sec={elapsed:.1f}")
                elif val_result is None:
                    print(f"epoch={epoch} step={global_step} train_loss={train_loss:.4f} elapsed_sec={elapsed:.1f}")
                else:
                    val_loss, val_ppl = val_result
                    print(
                        f"epoch={epoch} step={global_step} train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} val_ppl={val_ppl:.4f} elapsed_sec={elapsed:.1f}"
                    )

            score = val_result[0] if val_result is not None else train_loss
            improved = score < best_val
            if improved:
                best_val = score
                bad_epochs = 0
                if is_main:
                    metadata = sanitized_metadata(
                        args,
                        checkpoint_dir=checkpoint_dir,
                        output_dir=output_dir,
                        extra={
                            "model": args.model,
                            "linker": LINKER,
                            "world_size": world_size,
                            **dataset_metadata(args, records, dataset, train_dataset, val_dataset),
                            **({} if args.private_log else {"best_loss": best_val}),
                        },
                    )
                    save_lora_adapter(unwrap(model), output_dir / "best", metadata=metadata)
                    if args.save_split_indices:
                        with (output_dir / "split.json").open("w") as handle:
                            json.dump({"train_indices": train_indices, "val_indices": val_indices}, handle, indent=2)
            else:
                bad_epochs += 1

            if val_loader is not None and bad_epochs >= args.early_stop_patience:
                if is_main:
                    if args.private_log:
                        print(f"early_stop epoch={epoch}")
                    else:
                        print(f"early_stop epoch={epoch} best_loss={best_val:.4f}")
                break

        if is_main:
            save_lora_adapter(
                unwrap(model),
                output_dir / "last",
                metadata=sanitized_metadata(
                    args,
                    checkpoint_dir=checkpoint_dir,
                    output_dir=output_dir,
                    extra={
                        "model": args.model,
                        "linker": LINKER,
                        **dataset_metadata(args, records, dataset, train_dataset, val_dataset),
                    },
                ),
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

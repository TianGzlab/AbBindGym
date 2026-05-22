import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class QVLoraLinear(nn.Module):
    """LoRA adapter for ProGen2's merged qkv projection.

    ProGen2 stores query, value, and key projections in one Linear layer. The
    output layout is split into 8 model-parallel shards, each with
    [query, value, key]. This wrapper leaves the frozen base projection intact
    and adds trainable low-rank deltas only to query and value slices.
    """

    def __init__(self, base, rank=8, alpha=16, dropout=0.0, mp_num=8):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("QVLoraLinear expects an nn.Linear base module")
        if base.out_features % (3 * mp_num) != 0:
            raise ValueError("qkv projection output size is not compatible with the ProGen2 shard layout")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.mp_num = int(mp_num)
        self.local_dim = base.out_features // (3 * self.mp_num)
        self.proj_dim = self.local_dim * self.mp_num
        self.lora_dropout = nn.Dropout(dropout)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_q_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_q_B = nn.Linear(self.rank, self.proj_dim, bias=False)
        self.lora_v_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_v_B = nn.Linear(self.rank, self.proj_dim, bias=False)

        self.reset_lora_parameters()
        self.lora_q_A.float()
        self.lora_q_B.float()
        self.lora_v_A.float()
        self.lora_v_B.float()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x):
        output = self.base(x)

        lora_x = self.lora_dropout(x).to(self.lora_q_A.weight.dtype)
        q_delta = self.lora_q_B(self.lora_q_A(lora_x)) * self.scaling
        v_delta = self.lora_v_B(self.lora_v_A(lora_x)) * self.scaling

        q_delta = q_delta.to(output.dtype).reshape(*output.shape[:-1], self.mp_num, self.local_dim)
        v_delta = v_delta.to(output.dtype).reshape(*output.shape[:-1], self.mp_num, self.local_dim)

        delta = output.new_zeros(output.shape)
        delta = delta.view(*output.shape[:-1], self.mp_num, 3 * self.local_dim)
        delta[..., : self.local_dim] = q_delta
        delta[..., self.local_dim : 2 * self.local_dim] = v_delta
        return output + delta.view_as(output)


def apply_qv_lora(model, rank=8, alpha=16, dropout=0.0, mp_num=8):
    replaced = 0
    for block in model.transformer.h:
        qkv_proj = block.attn.qkv_proj
        if isinstance(qkv_proj, QVLoraLinear):
            continue
        block.attn.qkv_proj = QVLoraLinear(
            qkv_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            mp_num=mp_num,
        )
        replaced += 1
    return replaced


def mark_only_lora_trainable(model):
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, QVLoraLinear):
            for name, parameter in module.named_parameters():
                if name.startswith("lora_"):
                    parameter.requires_grad = True


def trainable_parameter_count(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def lora_state_dict(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_" in name
    }


def save_lora_adapter(model, output_dir, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter_model.bin")
    with (output_dir / "adapter_config.json").open("w") as handle:
        json.dump(metadata or {}, handle, indent=2, sort_keys=True)


def load_lora_adapter(model, adapter_dir, map_location="cpu"):
    adapter_dir = Path(adapter_dir)
    state = torch.load(adapter_dir / "adapter_model.bin", map_location=map_location, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [name for name in unexpected if ".lora_" in name]
    if unexpected_lora:
        raise RuntimeError(f"unexpected LoRA parameters: {unexpected_lora}")
    return missing, unexpected


def lora_parameters(model):
    for module in model.modules():
        if isinstance(module, QVLoraLinear):
            yield from module.lora_q_A.parameters()
            yield from module.lora_q_B.parameters()
            yield from module.lora_v_A.parameters()
            yield from module.lora_v_B.parameters()

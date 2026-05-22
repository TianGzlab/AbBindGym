import torch


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def select_device(requested, local_rank=0, world_size=1):
    if world_size > 1:
        return torch.device(f"cuda:{local_rank}")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_for(device, fp16):
    enabled = fp16 and device.type == "cuda"
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def create_grad_scaler(device, fp16):
    enabled = fp16 and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_progen_model(checkpoint_dir, fp16):
    from progen.modeling_progen import ProGenForCausalLM

    if fp16:
        return ProGenForCausalLM.from_pretrained(
            checkpoint_dir,
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    return ProGenForCausalLM.from_pretrained(checkpoint_dir)

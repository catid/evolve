from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool
    is_main_process: bool
    autocast_dtype: torch.dtype | None


def detect_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "fp32":
        return None
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def init_distributed(device_arg: str = "auto", precision: str = "auto") -> DistributedContext:
    torch.set_float32_matmul_precision("high")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    device = detect_device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank if device_arg != "cpu" else 0)
        device = torch.device("cuda", local_rank)
    if is_distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    autocast_dtype = choose_autocast_dtype(device, precision)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
        is_main_process=rank == 0,
        autocast_dtype=autocast_dtype,
    )


def maybe_wrap_ddp(module: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.is_distributed:
        return module
    if ctx.device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            module,
            device_ids=[ctx.device.index],
            output_device=ctx.device.index,
            find_unused_parameters=True,
        )
    return torch.nn.parallel.DistributedDataParallel(module, find_unused_parameters=True)


def unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def barrier(ctx: DistributedContext) -> None:
    if ctx.is_distributed:
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.device.index])
        else:
            dist.barrier()


def reduce_scalar_dict(metrics: dict[str, float], ctx: DistributedContext, average: bool = True) -> dict[str, float]:
    if not ctx.is_distributed:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= ctx.world_size
    return {key: float(value.item()) for key, value in zip(keys, values)}


def reduce_tensor_sum(values: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if not ctx.is_distributed:
        return values
    reduced = values.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def broadcast_scalar_dict(metrics: dict[str, float], ctx: DistributedContext) -> dict[str, float]:
    if not ctx.is_distributed:
        return metrics
    payload = [metrics if ctx.is_main_process else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0] or {}


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()

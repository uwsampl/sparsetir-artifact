import tvm
import torch
import os
from torch.profiler import profile, ProfilerActivity, schedule
from typing import List, Callable, Any

__all__ = ["profile_pytorch_ms", "profile_tvm_ms"]


def profile_tvm_ms(f: tvm.runtime.Module, args: List[Any]) -> float:
    flush_l2 = os.getenv("FLUSH_L2", "OFF") == "ON"
    if flush_l2:
        evaluator = f.time_evaluator(
            f.entry_name,
            tvm.cuda(0),
            number=1,
            repeat=100,
            f_preproc="l2_cache_flush_cuda",
        )
    else:
        evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=100)
    return evaluator(*args).mean * 1000


def profile_pytorch_ms(f: Callable[[], None]) -> float:
    flush_l2 = os.getenv("FLUSH_L2", "OFF") == "ON"
    n_warmup = 10
    n_repeat = 100
    """The following code copied from Triton profiler."""
    cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        f()
    # Benchmark
    for i in range(n_repeat):
        # we clear the L2 cache before each run
        if flush_l2:
            cache.zero_()
        # record time of `fn`
        start_event[i].record()
        f()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    )
    dur = torch.mean(times).item()
    return dur

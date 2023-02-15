import tvm
import torch
import os
from torch.profiler import profile, ProfilerActivity, schedule
from typing import List, Callable, Any, Tuple, Union
import subprocess

__all__ = ["profile_pytorch_ms", "profile_tvm_ms", "plot"]


def profile_tvm_ms(f: tvm.runtime.Module, args: List[Any]) -> float:
    flush_l2 = os.getenv("FLUSH_L2", "OFF") == "ON"
    if flush_l2:
        r"""
        NOTE(Zihao): TVM's native profiler has some constant extra 
        overhead when number=1, turn to use Triton's profiler instead.
        """
        # evaluator = f.time_evaluator(
        #     f.entry_name,
        #     tvm.cuda(0),
        #     number=1,
        #     repeat=100,
        #     f_preproc="l2_cache_flush_cuda",
        # )
        return profile_pytorch_ms(lambda: f(*args))
    else:
        evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=100)
    return evaluator(*args).mean * 1000


def profile_pytorch_ms(f: Callable[[], None]) -> float:
    r"""
    Use Triton's profiler when FLUSH_L2 is set to True.
    Use PyTorch's native profiler when FLUSH_L2 is set to False.
    """
    flush_l2 = os.getenv("FLUSH_L2", "OFF") == "ON"
    n_wait = 1
    n_warmup = 10
    n_repeat = 100
    if flush_l2:
        """The following code copied from Triton profiler."""
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
        start_event = [
            torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
        ]
        end_event = [
            torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
        ]
        # Warm-up
        for _ in range(n_warmup):
            f()
        # Benchmark
        for i in range(n_repeat):
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            start_event[i].record()
            f()
            end_event[i].record()
        # Record clocks
        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)])
        dur = torch.mean(times).item()
    else:
        with profile(activities=[ProfilerActivity.CUDA],
                     schedule=schedule(wait=n_wait,
                                       warmup=n_warmup,
                                       active=n_repeat)) as prof:
            for _ in range(n_wait + n_warmup + n_repeat):
                f()
                prof.step()
        dur = sum([e.cuda_time for e in prof.events()]) / 1000 / n_repeat
    return dur


plt_header = """
set output "{}.ps"
"""


def plot(filename: str, prelude: str, subplots: List[Tuple[str, str]],
         label_str_list: Union[str, List[str]], label_x_offset_func: Callable,
         p_list: List, ls_list: List, **extra_args):
    if isinstance(label_str_list, str):
        label_str_list = [label_str_list for _ in subplots]
    with open(filename + ".plt", "w") as f_out:
        f_out.write(plt_header.format(filename) + "\n")
        f_out.write(prelude + "\n")
        for plot_id, subplot in enumerate(subplots):
            label_str = label_str_list[plot_id]
            axis_suffix = ""
            if "axes" in extra_args:
                axis_suffix = " axes x1y{}".format(extra_args["axes"][plot_id])
            name, text = subplot
            f_out.write(text + "\n")
            with open(name + ".dat", "r") as f_in:
                lines = f_in.readlines()
                num_rows = len(lines)
                num_cols = len(lines[0].split())
            for i in range(num_cols - 1):
                fmt_str = "fs {} fc ls {} lw 3 ti col".format(
                    p_list[i], ls_list[i]) + axis_suffix
                if i == 0:
                    f_out.write(
                        """plot "{}" u (y_val($2)):xtic(1) {},\\\n""".format(
                            name + ".dat", fmt_str))
                else:
                    f_out.write("""'' u (y_val(${})) {},\\\n""".format(
                        i + 2, fmt_str))
                f_out.write("""'' u ($0+({})):(y_pos(${})):(to_str(${})) {} """
                            .format(label_x_offset_func(i), i + 2, i +
                                    2, label_str) + axis_suffix)
                if i != num_cols - 2:
                    f_out.write(",\\\n")
                else:
                    f_out.write("\n")

    subprocess.call(["gnuplot", filename + ".plt"])
    subprocess.call(["epstopdf", filename + ".ps"])
    subprocess.call(["rm", filename + ".plt"])
    subprocess.call(["rm", filename + ".ps"])

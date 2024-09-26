import functools
import time
import torch
import os
from datetime import datetime
import argparse

class Profiler:
    enabled = False
    snapshot_dir = 'profiler_snapshots'
    run_id = None

    @classmethod
    def initialize(cls, output_dir='profiler_snapshots'):
        cls.enabled = True
        cls.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cls.snapshot_dir = os.path.join(output_dir, f"run_{cls.run_id}")
        os.makedirs(cls.snapshot_dir, exist_ok=True)
        torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='all')

    @classmethod
    def cuda(cls):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not cls.enabled:
                    return func(*args, **kwargs)

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                start_memory = torch.cuda.memory_allocated()

                result = func(*args, **kwargs)

                torch.cuda.synchronize()
                end_time = time.perf_counter()
                end_memory = torch.cuda.memory_allocated()

                elapsed_time = end_time - start_time
                memory_diff = (end_memory - start_memory) / 1024**2  # Convert to MB

                print(f"Function: {func.__name__}")
                print(f"Elapsed time: {elapsed_time:.4f} seconds")
                print(f"Memory change: {memory_diff:.2f} MB")

                return result
            return wrapper
        return decorator


    @classmethod
    def take_snapshot(cls, label=''):
        if not cls.enabled:
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{label}_snapshot_{timestamp}.pickle"
            filepath = os.path.join(cls.snapshot_dir, filename)
            torch.cuda.memory._dump_snapshot(filepath)
            print(f"CUDA memory snapshot saved: {filepath}")
        except Exception as e:
            print(f"Failed to take snapshot: {e}")

    @classmethod
    def print_memory_stats(cls):
        if not cls.enabled:
            return

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e6:.2f}MB")
            print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1e6:.2f}MB")

    @classmethod
    def reset_peak_memory_stats(cls):
        if cls.enabled:
            torch.cuda.reset_peak_memory_stats()

    @classmethod
    def get_peak_memory_stats(cls):
        if not cls.enabled:
            return {}
        return {i: torch.cuda.max_memory_allocated(i) / 1e6 for i in range(torch.cuda.device_count())}

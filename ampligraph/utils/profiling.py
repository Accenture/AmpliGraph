# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tracemalloc
from functools import wraps
from time import time


def get_memory_size():
    """Get memory size.

    Returns
    -------
    Total: float
        Memory size used in total.
    """
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno", cumulative=True)
    total = sum(stat.size for stat in stats)
    return total


def get_human_readable_size(size_in_bytes):
    """Convert size from bytes to human readable units.

    Parameters
    ----------
    size_in_bytes: int
        Original size given in bytes

    Returns
    -------
    readable_size: tuple
        Tuple of new size and unit, size in units GB/MB/KB/Bytes according
        to thresholds.
    """
    if size_in_bytes >= 1024 * 1024 * 1024:
        return float(size_in_bytes / (1024 * 1024 * 1024)), "GB"
    if size_in_bytes >= 1024 * 1024:
        return float(size_in_bytes / (1024 * 1024)), "MB"
    if size_in_bytes >= 1024:
        return float(size_in_bytes / 1024), "KB"  # return in KB
    return float(size_in_bytes), "Bytes"


def timing_and_memory(f):
    """Decorator to register time and memory used by a function f.

    Parameters
    ----------
    f: function
        Function for which the time and memory will be measured.

    It logs the time and the memory in the dictionary passed inside `'log'`
    parameter if provided. Time is logged in seconds, memory in bytes.
    Example dictionary entry looks like that:
    {'SPLIT': {'time': 1.62, 'memory-bytes': 789.097}},
    where keys are names of functions that were called to get
    the time measured in uppercase.

    Requires
    --------
    passing **kwargs in function parameters
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        mem_before = get_memory_size()
        start = time()
        result = f(*args, **kwargs)
        end = time()
        mem_after = get_memory_size()
        mem_diff = mem_after - mem_before
        print(
            "{}: memory before: {:.5}{}, after: {:.5}{},\
              consumed: {:.5}{}; exec time: {:.5}s".format(
                f.__name__,
                *get_human_readable_size(mem_before),
                *get_human_readable_size(mem_after),
                *get_human_readable_size(mem_diff),
                end - start
            )
        )

        if "log" in kwargs:
            name = kwargs.get("log_name", f.__name__.upper())
            kwargs["log"][name] = {
                "time": end - start,
                "memory-bytes": mem_diff,
            }
        return result

    return wrapper

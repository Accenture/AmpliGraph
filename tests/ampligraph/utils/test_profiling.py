# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.utils.profiling import get_memory_size, get_human_readable_size, timing_and_memory 
import time
import pytest
import numpy as np

@pytest.mark.skip(reason="may not be a reliable way for measuring memory used...")
def test_get_memory_size():
    pre_size = get_memory_size()
    post_size = None
    # create table of a certain size and make sure difference in 
    # occupied memory is visable with get_memory_size function
    size = int(0.1 * 1024 * 1024 * 1024)  # 0.1 of 1 Gb 
    tab = b'0' * size
    post_size = get_memory_size()

    assert(post_size - pre_size >= size)


def test_get_human_readable_size():
    size_in_gb = 1.210720
    result1, result2 = get_human_readable_size(1300000000)
    assert(np.round(result1,5) == size_in_gb and result2 == "GB")


def test_timing_and_memory_logging():
    @timing_and_memory
    def mock_fcn(**kwargs):
        time.sleep(1.0)
    test_logs = {"MOCK_FCN": None}
    logs = {}
    mock_fcn(log=logs)
    assert(logs.keys() == test_logs.keys())


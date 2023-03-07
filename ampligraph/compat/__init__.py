# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Provides backward compatibility to AmpliGraph 1 APIs."""
from .evaluate import evaluate_performance
from .models import ComplEx, DistMult, HolE, TransE

__all__ = ["evaluate_performance", "TransE", "ComplEx", "DistMult", "HolE"]

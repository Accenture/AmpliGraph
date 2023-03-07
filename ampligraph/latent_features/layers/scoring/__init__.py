# Copyright 2019-20213The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from .TransE import TransE
from .DistMult import DistMult
from .HolE import HolE
from .ComplEx import ComplEx
from .Random import Random

__all__ = ["TransE", "DistMult", "HolE", "ComplEx", "Random"]

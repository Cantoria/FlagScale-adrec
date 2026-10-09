# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native recommendation models (SASRec, DSSM, LightGCN) used by examples/rec."""

from .base import RecModel, unwrap_model
from .dssm import DSSM
from .lightgcn import LightGCN
from .registry import REC_MODEL_REGISTRY, build_rec_model, register_rec_model
from .sasrec import SASRec

__all__ = [
    "DSSM",
    "LightGCN",
    "REC_MODEL_REGISTRY",
    "RecModel",
    "SASRec",
    "build_rec_model",
    "register_rec_model",
    "unwrap_model",
]

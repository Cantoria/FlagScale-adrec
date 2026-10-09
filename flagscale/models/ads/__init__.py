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

"""Native advertising (CTR) models (DIN, DeepFM, DCN-Mix) used by examples/ads."""

from .base import AdsModel, unwrap_model
from .dcn_mix import DCNMix
from .deepfm import DeepFM
from .din import DIN
from .registry import ADS_MODEL_REGISTRY, build_ads_model, register_ads_model

__all__ = [
    "ADS_MODEL_REGISTRY",
    "AdsModel",
    "DCNMix",
    "DIN",
    "DeepFM",
    "build_ads_model",
    "register_ads_model",
    "unwrap_model",
]

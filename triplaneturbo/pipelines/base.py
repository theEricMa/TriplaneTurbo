from typing import *

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline


class Pipeline(DiffusionPipeline):
    """Base class for all pipelines."""

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def enable_xformers_memory_efficient_attention(self):
        pass

    def enable_model_cpu_offload(self):
        pass

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, "device"):
                return model.device
        for model in self.models.values():
            if hasattr(model, "parameters"):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

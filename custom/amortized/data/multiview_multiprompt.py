import math
import json
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from torch.utils.data import DataLoader, Dataset, IterableDataset
from threestudio.data.uncond_multiview import RandomMultiviewCameraIterableDataset
from .multiprompt import (
    MultipromptRandomCameraDataset4FixPrompt,
    MultipromptRandomCameraDataset4Test
)

@dataclass
class MultiviewMultipromptRandomCameraDataModuleConfig(RandomCameraDataModuleConfig):
    # new config for generative model optimization
    dim_gaussian: int = 512
    prompt_library: str = "magic3d_prompt_library"
    prompt_library_dir: str = "load"
    prompt_library_format: str = "json"
    eval_prompt: Optional[str] = None
    target_prompt: Optional[str] = None
    eval_fix_camera: Optional[int] = None # can be int, then fix the camera to the specified view
    # new config for multi-view in mvdream
    relative_radius: bool = True
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)


class MultiviewMultipromptRandomCameraIterableDataset(RandomMultiviewCameraIterableDataset):
    def __init__(self, cfg: Any, prompt_library: Dict) -> None:
        # just follow original initialization
        super().__init__(cfg)
        assert "train" in prompt_library, "prompt library must contain train split"
        self.prompt_library = prompt_library["train"] # make it a ordered list
        self.n_view = self.cfg.n_view
        
    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        batch_size = self.batch_size // self.n_view

        # first follow original collate
        batch_dict = super().collate(batch)
        # then add gaussian noise
        batch_dict["noise"] = torch.randn(batch_size, self.cfg.dim_gaussian)
        # randomly select prompt from a list
        if len(self.prompt_library) < batch_size:
            batch_dict["prompt"] = random.choices(self.prompt_library, k=batch_size)
        else:
            batch_dict["prompt"] = random.sample(self.prompt_library, k=batch_size)

        return batch_dict
    
@register("multiprompt-multiview-camera-datamodule")
class MultiviewRandomCameraDataModule(pl.LightningDataModule):
    cfg: MultiviewMultipromptRandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewMultipromptRandomCameraDataModuleConfig, cfg)
        path = os.path.join(
            self.cfg.prompt_library_dir, 
            self.cfg.prompt_library) \
                + "." + self.cfg.prompt_library_format
        with open(path, "r") as f:
            self.prompt_library = json.load(f)

            # each process only has a subset of the prompt library!
            rank = get_rank()
            num_gpu = torch.cuda.device_count()
            for key in self.prompt_library:
                self.prompt_library[key] = self.prompt_library[key][
                    rank::num_gpu
                ]
                
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = MultiviewMultipromptRandomCameraIterableDataset(self.cfg, prompt_library=self.prompt_library)
        if stage in (None, "fit", "validate"):
            self.val_dataset = MultipromptRandomCameraDataset4Test(self.cfg, "val", prompt_library=self.prompt_library)
        if stage in (None, "test", "predict"):
            if self.cfg.eval_prompt is not None:
                # fix the prompt during evaluation
                self.test_dataset = MultipromptRandomCameraDataset4FixPrompt(self.cfg, "test", )
            else:
                self.test_dataset = MultipromptRandomCameraDataset4Test(self.cfg, "test", prompt_library=self.prompt_library)
                # todo, is it ok to use test_dataset for prediction?

    def prepare_data(self) -> None:
        pass

    def general_loader(self, dataset, batch_size, shuffle = False, sampler = None, collate_fn: Optional[Callable] = None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
        )
    
    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )
    
    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=None, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )
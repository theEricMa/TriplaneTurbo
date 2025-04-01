import os
import json
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, SequentialSampler

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured

from threestudio.utils.typing import *
from threestudio.data.uncond import RandomCameraIterableDataset, RandomCameraDataset

from threestudio.utils.misc import get_rank

@dataclass
class MultipromptRandomCameraDataModuleConfig:
    # original config from threestudio/data/uncond.py, for single 3D representation optimization
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    # new config for generative model optimization
    dim_gaussian: int = 512
    prompt_library: str = "magic3d_prompt_library"
    prompt_library_dir: str = "load"
    prompt_library_format: str = "json"
    eval_prompt: Optional[str] = None
    target_prompt: Optional[str] = None
    eval_fix_camera: Optional[int] = None # can be int, then fix the camera to the specified view
    
class MultipromptRandomCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, cfg: Any, prompt_library: Dict) -> None:
        # just follow original initialization
        super().__init__(cfg)
        assert "train" in prompt_library, "prompt library must contain train split"
        self.prompt_library = prompt_library["train"] # make it a ordered list
        
    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # first follow original collate
        batch_dict = super().collate(batch)
        # then add gaussian noise
        batch_dict["noise"] = torch.randn(self.batch_size, self.cfg.dim_gaussian)
        # randomly select prompt from a list
        if len(self.prompt_library) < self.batch_size:
            batch_dict["prompt"] = random.choices(self.prompt_library, k=self.batch_size)
        else:
            batch_dict["prompt"] = random.sample(self.prompt_library, k=self.batch_size)

        return batch_dict
    
class MultipromptRandomCameraDataset4Test(IterableDataset):
    def __init__(self, cfg: Any, split: str, prompt_library: Dict) -> None:
        # just follow original initialization
        self.dataset = RandomCameraDataset(cfg = cfg, split = split)
        self.cfg =  self.dataset.cfg
        self.n_views = self.dataset.n_views
        # then add gaussian noise, set up the start and end point for interpolation
        start_point = torch.randn(self.cfg.dim_gaussian)  # TODO, is this the right way to initialize, when using DDP?
        end_point = torch.randn(self.cfg.dim_gaussian)
        self.noises = torch.stack(
                [
                    start_point + (end_point - start_point) * i / self.n_views \
                        for i in range(self.n_views)
                ]
            )
        self.prompt_library = prompt_library[split] if split in prompt_library else prompt_library["val"]

    def __iter__(self):
        for prompt in self.prompt_library:
            yield {
                "prompt": [prompt]
            }

    def collate(self, batch):
        if not hasattr(self, "n_views_cache"):
            batch_dict_list = []
            for i in range(self.n_views):
                batch_dict = self.dataset.__getitem__(i)
                batch_dict_list.append(batch_dict)
            # collate the batch_dict_list
            self.n_views_cache = torch.utils.data._utils.collate.default_collate(batch_dict_list)
    
        batch_dict = self.n_views_cache.copy()
        # then add gaussian noise
        batch_dict["noise"] = self.noises[0][None, :]
        # then add prompt
        batch_dict.update(batch)
        # override the height and width
        batch_dict['height'] = self.cfg.eval_height
        batch_dict['width'] = self.cfg.eval_width
        return batch_dict
    

class MultipromptRandomCameraDataset4FixPrompt(IterableDataset):
    def __init__(self, cfg: Any, split: str,) -> None:
        # just follow original initialization
        self.dataset = RandomCameraDataset(cfg = cfg, split = split)
        self.cfg =  self.dataset.cfg
        self.n_views = self.dataset.n_views
        # then add gaussian noise
        self.noise = torch.zeros(self.cfg.dim_gaussian)
        self.eval_prompt = self.cfg.eval_prompt
        self.target_prompt = self.cfg.target_prompt
        self.ratios = torch.linspace(0, 1, self.n_views)
        self.fix_camera = self.cfg.eval_fix_camera

    def __iter__(self):
        for idx in range(self.n_views):
            yield self.__getitem__(idx)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # just follow original getitem
        if self.fix_camera:
            # use the specified view
            batch_dict = self.dataset.__getitem__(self.fix_camera)
        else:
            batch_dict = self.dataset.__getitem__(idx)
            
        # then add gaussian noise
        batch_dict["noise"] = self.noise
        # then that's it
        batch_dict["prompt"] = self.eval_prompt
        # update the index
        batch_dict["index"] = idx

        # if target_prompt is not None, then add target_prompt
        if self.target_prompt is not None:
            batch_dict["prompt_target"] = self.target_prompt
            batch_dict["ratio"] = self.ratios[idx]

        batch_dict["name"] = '_to_'.join([self.eval_prompt, self.target_prompt]) if self.target_prompt is not None else self.eval_prompt

        return batch_dict

@register("multi-prompt-random-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: MultipromptRandomCameraDataModuleConfig
    
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultipromptRandomCameraDataModuleConfig, cfg)
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
            self.train_dataset = MultipromptRandomCameraIterableDataset(self.cfg, prompt_library=self.prompt_library)
        if stage in (None, "fit", "validate"):
            self.val_dataset = MultipromptRandomCameraDataset4Test(self.cfg, "val", prompt_library=self.prompt_library)
        if stage in (None, "test", "predict"):
            if self.cfg.eval_prompt is not None:
                # fix the prompt during evaluation
                self.use_fix_prompt = True
                self.test_dataset = MultipromptRandomCameraDataset4FixPrompt(self.cfg, "test", )
            else:
                self.use_fix_prompt = False
                self.test_dataset = MultipromptRandomCameraDataset4Test(self.cfg, "test", prompt_library=self.prompt_library)
                # todo, is it ok to use test_dataset for prediction?

    def prepare_data(self) -> None:
        pass

    def general_loader(self, dataset, batch_size, shuffle = None, sampler = None, collate_fn: Optional[Callable] = None) -> DataLoader:
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
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate, #shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=None, collate_fn=self.val_dataset.collate, #sampler=SequentialSampler(self.val_dataset)
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        if self.use_fix_prompt:
            return self.general_loader(
                self.test_dataset, batch_size=1,
            )
        else:
            return self.general_loader(
                self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate, #sampler=SequentialSampler(self.test_dataset)
            )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate, #sampler=SequentialSampler(self.test_dataset)
        )
    
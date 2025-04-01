import bisect
import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pytorch_lightning as pl
import os
import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
import json
from threestudio.utils.config import parse_structured

@dataclass
class MultiviewMultipromptDualRendererDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    batch_size: Any = 1
    eval_batch_size: int = 1
    n_val_views: int = 1 
    n_test_views: int = 32 
    n_view:int=4
    # relative_radius: bool = True
    height: int =256
    width: int =256
    ray_height: int = 64
    ray_width: int = 64
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    # camera ranges are randomized, not specified
    elevation_range: Tuple[float, float] = (-10, 90)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (40, 70,)
    azimuth_range: Tuple[float, float] = (-180, 180)
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    relative_radius: bool = True
    # relightable settings
    light_sample_strategy: str = "dreamfusion"
    # eval camera settings
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    # new config for generative model optimization
    dim_gaussian: List[int] = field(default_factory=lambda: [])
    pure_zeros: bool = False # return pure zeros for the latent code, instead of random noise
    prompt_library: str = "magic3d_prompt_library"
    prompt_library_dir: str = "load"
    prompt_library_format: str = "json"
    eval_prompt: Optional[str] = None
    target_prompt: Optional[str] = None
    eval_fix_camera: Optional[int] = None # can be int, then fix the camera to the specified view
    # prompt processor configs


class MultiviewMultipromptDualRendererDataset(Dataset, Updateable):
    def __init__(self, cfg: Any, split: str, prompt_library: List, prompt_processor = None) -> None:
        super().__init__()
        self.cfg: MultiviewMultipromptDualRendererDataModuleConfig = cfg
        ##############################################################################################################
        # the following config may be updated along with the training process
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        self.ray_heights: List[int] = (
            [self.cfg.ray_height] if isinstance(self.cfg.ray_height, int) else self.cfg.ray_height
        )
        self.ray_widths: List[int] = (
            [self.cfg.ray_width] if isinstance(self.cfg.ray_width, int) else self.cfg.ray_width
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes) == len(self.ray_heights) == len(self.ray_widths)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
            and len(self.ray_heights) == 1
            and len(self.ray_widths) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones
        # since the unit is for volume rendering, its width and height are determined by the ray_height and ray_width
        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.ray_heights, self.ray_widths)
        ]
        self._directions_unit_focals_rasterize = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        ##############################################################################################################
        # the following config is intialized for the 1st iteration
        self.batch_size: int = self.batch_sizes[0]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.ray_height: int = self.ray_heights[0]
        self.ray_width: int = self.ray_widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self._directions_unit_focal_rasterize = self._directions_unit_focals_rasterize[0]
        # the following config is fixed for the whole training process
        self.elevation_range: Tuple[float, float] = self.cfg.elevation_range
        self.azimuth_range: Tuple[float, float] = self.cfg.azimuth_range
        self.camera_distance_range: Tuple[float, float] = self.cfg.camera_distance_range
        self.fovy_range: Tuple[float, float] = self.cfg.fovy_range
        
        ##############################################################################################################
        # the following config is related to the prompt library
        self.prompt_library: Dict = prompt_library
        if prompt_processor is None:
            self.prompt_processor = lambda x: x
        else:
            self.prompt_processor = prompt_processor

        ##############################################################################################################
        # the following config is related to training/testing split
        assert split in ["train", "val", "test"]
        self.split = split


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_index = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.batch_size = self.batch_sizes[size_index]
        self.height = self.heights[size_index]
        self.width = self.widths[size_index]
        self.ray_height = self.ray_heights[size_index]
        self.ray_width = self.ray_widths[size_index]
        self.directions_unit_focal = self.directions_unit_focals[size_index]
        self._directions_unit_focal_rasterize = self._directions_unit_focals_rasterize[size_index]
        threestudio.debug(
            f"Updated batch_size={self.batch_size}, height={self.height}, width={self.width}, ray_height={self.ray_height}, ray_width={self.ray_width}"
        )

    def __len__(self) -> int:
        return len(self.prompt_library)
    
    def __getitem__(self, index: int) -> Dict:
        # load the prompt
        return {
            "prompt": self.prompt_library[index]
        }
    
    def _light_position(self, camera_positions: Float[Tensor, "B 3"]):
        real_batch_size = self.batch_size // self.cfg.n_view
        
        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"

        if self.cfg.light_sample_strategy == "dreamfusion":
            # determine light direction by normalizing camera positions
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )
        return light_positions

    def collate(self, batch) -> Dict[str, Any]:
        if self.split == "train":
            return self._train_collate(batch)
        else:
            return self._test_collate(batch)

    def _test_collate(self, batch) -> Dict[str, Any]:
        # collate the prompts
        prompts = [b["prompt"] for b in batch]
        # set the number of views
        n_views = self.cfg.n_val_views if self.split == "val" else self.cfg.n_test_views
        ##############################################################################################################
        # arrange the azimuth angles
        azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360.0, n_views)
        azimuth = azimuth_deg * math.pi / 180
        ##############################################################################################################
        # set the elevation angles
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        elevation = elevation_deg * math.pi / 180
        ##############################################################################################################
        # set the fovs
        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        ##############################################################################################################
        # set the camera distances
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        ##############################################################################################################
        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1) 
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.ray_height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        directions_rasterize: Float[Tensor, "B H W 3"] = self._directions_unit_focal_rasterize[
            None, :, :, :
        ].repeat(n_views, 1, 1, 1)
        directions_rasterize[:, :, :, :2] = (
            directions_rasterize[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        _, rays_d_rasterize = get_rays(directions_rasterize, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "index": torch.arange(n_views),
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
            "prompt": prompts,
            "noise": torch.randn(1, *self.cfg.dim_gaussian) if not self.cfg.pure_zeros else torch.zeros(1, *self.cfg.dim_gaussian),
            "rays_d_rasterize":  rays_d_rasterize
        }
    
    def _train_collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # collate the prompts
        prompts = [b["prompt"] for b in batch]
        ##############################################################################################################
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"] = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        elevation: Float[Tensor, "B"] = elevation_deg * math.pi / 180

        ##############################################################################################################
        # sample azimuth angles, ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth: Float[Tensor, "B"] = azimuth_deg * math.pi / 180

        ##############################################################################################################
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        ##############################################################################################################
        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        ##############################################################################################################
        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # camera pertubation is not implemented

        # light position is only used in relightable mode, so put it in the function
        light_positions = self._light_position(camera_positions)

        # camera to world matrix
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.ray_height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        directions_rasterize: Float[Tensor, "B H W 3"] = self._directions_unit_focal_rasterize[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions_rasterize[:, :, :, :2] = (
            directions_rasterize[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        _, rays_d_rasterize = get_rays(directions_rasterize, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
            "prompt": prompts,
            "noise": torch.randn(real_batch_size, *self.cfg.dim_gaussian) if not self.cfg.pure_zeros else torch.zeros(real_batch_size, *self.cfg.dim_gaussian),
            "rays_d_rasterize":  rays_d_rasterize
        }


@register("multiprompt-multiview-dualrenderer-camera-datamodule")
class MultiviewRandomCameraDataModule(pl.LightningDataModule):
    cfg: MultiviewMultipromptDualRendererDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewMultipromptDualRendererDataModuleConfig, cfg)
        path = os.path.join(
            self.cfg.prompt_library_dir, 
            self.cfg.prompt_library) \
                + "." + self.cfg.prompt_library_format
        
        with open(path, "r") as f:
            self.prompt_library = json.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset =MultiviewMultipromptDualRendererDataset(
                self.cfg, 
                "train", 
                prompt_library=self.prompt_library["train"]
            )
        if stage in (None, "fit", "validate"):
            self.val_dataset = MultiviewMultipromptDualRendererDataset(
                self.cfg, 
                "val", 
                prompt_library=self.prompt_library["val"]
            )
        if stage in (None, "test", "predict"):
            if self.cfg.eval_prompt is not None:
                # fix the prompt during evaluation
                raise NotImplementedError
            else:
                self.test_dataset = MultiviewMultipromptDualRendererDataset(
                    self.cfg, 
                    "test", 
                    prompt_library=self.prompt_library["test"]
                )
                # todo, is it ok to use test_dataset for prediction?

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=1, # not used
            collate_fn=self.train_dataset.collate,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1, # not used
            collate_fn=self.val_dataset.collate,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1, # not used
            collate_fn=self.test_dataset.collate,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1, # not used
            collate_fn=self.test_dataset.collate,
        )
    

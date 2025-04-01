import bisect
import math
import random
from dataclasses import dataclass, field
from collections import OrderedDict

import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

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


import json
from threestudio.utils.config import parse_structured

from PIL import Image
from functools import partial

@dataclass
class MultiviewMultipromptDualRendererMultiStepDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    batch_size: Any = 4
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
    # relightable settings
    light_sample_strategy: str = "dreamfusion"
    # the following settings are for unsupervised training ############################
    # the prompt corpus
    prompt_library: str = "magic3d_prompt_library"
    prompt_library_dir: str = "load"
    prompt_library_format: str = "json"
    # camera ranges are randomized, not specified
    unsup_elevation_range: Tuple[float, float] = (-10, 90)
    unsup_camera_distance_range: Tuple[float, float] = (1, 1.5)
    unsup_fovy_range: Tuple[float, float] = (40, 70,)
    unsup_azimuth_range: Tuple[float, float] = (-180, 180)
    unsup_light_distance_range: Tuple[float, float] = (0.8, 1.5)
    relative_radius: bool = True
    # eval camera settings if not specified
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    # the following settings are for supervised training ############################
    obj_library: str = "objaverse_debug"
    obj_library_dir: str = "datasets"
    meta_json: str = "filtered_3DTopia-objaverse-caption-361k.json"
    rgb_data_dir: str = "exported_rgb"
    rgb_bg: Tuple[float, float, float] = (0.5, 0.5, 0.5) # gray
    normal_data_dir: str = "exported_normal"
    normal_bg: Tuple[float, float, float] = (0, 0, 0) # black
    depth_data_dir: str = "exported_depth"
    depth_bg: Tuple[float, float, float] = (0, 0, 0) # black
    camera_data_dir: str = "exported_json"
    frontal_idx: int = 24 # start from 0

    # the following settings are for the preprocessing of the text prompt ############################
    # applied to both supervised and unsupervised data
    guidance_processor_type: str = ""
    guidance_processor: dict = field(default_factory=dict)

    condition_processor_type: str = ""
    condition_processor: dict = field(default_factory=dict)

    # the sup / unsup ratio
    sup_unsup_mode: str = "50/50" # "vanilla"

    # new config for generative model optimization
    dim_gaussian: List[int] = field(default_factory=lambda: [])
    pure_zeros: bool = False # return pure zeros for the latent code, instead of random noise
    # eval settings
    eval_prompt: Optional[str] = None
    target_prompt: Optional[str] = None
    eval_fix_camera: Optional[int] = None # can be int, then fix the camera to the specified view

    # number of steps for the training
    n_steps: int = 4

class BaseDataset(Dataset, Updateable):
    def __init__(
            self, 
            cfg: Any, 
            unsup_prompt_library: List, 
            sup_obj_library: List,
            guidance_processor = None,
            condition_processor = None
        ) -> None:
        super().__init__()
        self.cfg: MultiviewMultipromptDualRendererMultiStepDataModuleConfig = cfg        
        ##############################################################################################################
        self.batch_size: int = self.cfg.batch_size
        # the following config may be updated along with the training process
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )

        self.ray_heights: List[int] = (
            [self.cfg.ray_height] if isinstance(self.cfg.ray_height, int) else self.cfg.ray_height
        )
        self.ray_widths: List[int] = (
            [self.cfg.ray_width] if isinstance(self.cfg.ray_width, int) else self.cfg.ray_width
        )
        assert len(self.heights) == len(self.widths) == len(self.ray_heights) == len(self.ray_widths)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
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
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.ray_height: int = self.ray_heights[0]
        self.ray_width: int = self.ray_widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self._directions_unit_focal_rasterize = self._directions_unit_focals_rasterize[0]
        # the following config is fixed for the whole training process
        self.elevation_range: Tuple[float, float] = self.cfg.unsup_elevation_range
        self.azimuth_range: Tuple[float, float] = self.cfg.unsup_azimuth_range
        self.camera_distance_range: Tuple[float, float] = self.cfg.unsup_camera_distance_range
        self.fovy_range: Tuple[float, float] = self.cfg.unsup_fovy_range
        
        ##############################################################################################################
        # the following config is related to the prompt library without ground truth
        self.unsup_prompt_library: List = unsup_prompt_library
        self.unsup_length = len(self.unsup_prompt_library)

        ##############################################################################################################
        # the following config is related to the prompt  library with ground truth
        self.sup_obj_library: OrderedDict = OrderedDict(sup_obj_library)
        self.sup_length = len(self.sup_obj_library.keys())

        ##############################################################################################################
        # the following config is related to the prompt processor
        self.guidance_processor = guidance_processor
        self.condition_processor = condition_processor


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        ##############################################################################################################
        # the following is the conventional way to update along with the training process
        size_index = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_index]
        self.width = self.widths[size_index]
        self.ray_height = self.ray_heights[size_index]
        self.ray_width = self.ray_widths[size_index]
        self.directions_unit_focal = self.directions_unit_focals[size_index]
        self._directions_unit_focal_rasterize = self._directions_unit_focals_rasterize[size_index]
        threestudio.debug(
            f"Updated height={self.height}, width={self.width}, ray_height={self.ray_height}, ray_width={self.ray_width}"
        )


    def __len__(self) -> int:
        None

    def __getitem__(self, idx: int) -> Dict:
        None

    def collate(self, batch) -> Dict[str, Any]:

        # the guidance_utils is a list of guidance_utils, each is a object, 
        # need to merge them into a single object
        guidance_util_list = [x.pop("guidance_utils") for x in batch]
        guidance_utils = guidance_util_list[0]
        for other_guidance_utils in guidance_util_list[1:]:

            # merge the attributes in these guidance_utils, sort of hard-coded, apologies
            # these attributes include global_text_embeddings, local_text_embeddings, text_embeddings_vd
            assert hasattr(guidance_utils, "appendable_attributes"), "Cannot merge guidance_utils"
            assert type(guidance_utils.appendable_attributes) == list, "Cannot merge guidance_utils"
            has_appendable_attributes = False
            for attr in guidance_utils.appendable_attributes:
                if hasattr(guidance_utils, attr):
                    has_appendable_attributes = True
                    getattr(guidance_utils, attr).extend(getattr(other_guidance_utils, attr))
            if not has_appendable_attributes:
                raise NotImplementedError("The Merge of guidance_utils is not implemented, please check the guidance_utils")


        condition_util_list = [x.pop("condition_utils") for x in batch]
        condition_utils = condition_util_list[0]
        for other_condition_utils in condition_util_list[1:]:
            # merge the attributes in these condition_utils, sort of hard-coded, apologies
            # these attributes include global_text_embeddings, local_text_embeddings, text_embeddings_vd
            assert hasattr(condition_utils, "appendable_attributes"), "Cannot merge condition_utils"
            assert type(condition_utils.appendable_attributes) == list, "Cannot merge condition_utils"
            has_appendable_attributes = False
            for attr in condition_utils.appendable_attributes:
                if hasattr(condition_utils, attr):
                    has_appendable_attributes = True
                    getattr(condition_utils, attr).extend(getattr(other_condition_utils, attr))
            if not has_appendable_attributes:
                raise NotImplementedError("The Merge of condition_utils is not implemented, please check the condition_utils")
            
        batch = torch.utils.data.default_collate(batch)
        batch.update(
            {
                "guidance_utils": guidance_utils,
                "condition_utils": condition_utils,
            }
        )
        return batch



    def _create_camera_from_angle(
        self,
        elevation_deg: Float[Tensor, "B"],
        azimuth_deg: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        fovy_deg: Float[Tensor, "B"],
        relative_radius: bool = False,
        phase: str = "train", # "train" or "test"
    ) -> Dict[str, Any]:
        # this function is independent of the self.cfg.n_view

        assert elevation_deg.shape == azimuth_deg.shape == camera_distances.shape == fovy_deg.shape
        batch_size = elevation_deg.shape[0]

        fovy = fovy_deg * math.pi / 180
        azimuth: Float[Tensor, "B"] = azimuth_deg * math.pi / 180
        elevation: Float[Tensor, "B"] = elevation_deg * math.pi / 180

        ##############################################################################################################
        # in MV-Dream, the camera distance is relative and related to the focal length
        # the following is the default setting, 
        # however, the relative camera distance is not used in supervised training
        camera_distances_relative = camera_distances
        if relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

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
        ).to(torch.float32)

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(batch_size, 1)

        # camera pertubation is not implemented

        # light position is only used in relightable mode, so put it in the function
        if phase == "train":
            light_positions = self._random_camera_to_light_position(camera_positions)
        else:
            light_positions = camera_positions

        # camera to world matrix
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.linalg.cross(lookat, up), dim=-1)
        up = F.normalize(torch.linalg.cross(right, lookat), dim=-1)
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
        ].repeat(batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        directions_rasterize: Float[Tensor, "B H W 3"] = self._directions_unit_focal_rasterize[
            None, :, :, :
        ].repeat(batch_size, 1, 1, 1)
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
            "camera_distances_relative": camera_distances_relative,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
            "rays_d_rasterize":  rays_d_rasterize
        }


    def _random_camera_to_light_position(self, camera_positions: Float[Tensor, "B 3"]):
        # this function is dependent on the self.cfg.n_view
        batch_size = camera_positions.shape[0]
        real_batch_size = batch_size // self.cfg.n_view
        
        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.unsup_light_distance_range[1] - self.cfg.unsup_light_distance_range[0])
            + self.cfg.unsup_light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

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
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )
        return light_positions

    def _load_im(
            self, 
            path: str,
            color: Optional[Float[Tensor, "3"]] = None,
        ):
        '''
        replace background pixel with specified color in rendering
        '''
        try:
            pil_img = Image.open(path)
        except:
            raise ValueError(f"Failed to load image: {path}")

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha
        if color is not None:
            image = image + (1 - alpha) * color

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def _load_images(
        self,
        load_indices: List[int] = [0],
        rgb_data_dir: Optional[str] = None,
        normal_data_dir: Optional[str] = None,
        depth_data_dir: Optional[str] = None,
    ):
        # load the rgb, normal, depth and mask images
        rgb_images = []
        normal_images = []
        depth_images = []
        mask_images = []
        for idx in load_indices:
            # the data name is of 000.png, 001.png, ...
            # rgb image
            rgb_file = os.path.join(rgb_data_dir, "{:03d}.png".format(idx))
            rgb_image, alpha = self._load_im(rgb_file, color=self.cfg.rgb_bg)
            rgb_images.append(rgb_image)

            # normal image
            normal_file = os.path.join(normal_data_dir, "{:03d}.png".format(idx))
            normal_image, _ = self._load_im(normal_file, color=self.cfg.normal_bg)
            normal_images.append(normal_image)


            # depth image
            depth_file = os.path.join(depth_data_dir, "{:03d}.png".format(idx))
            depth_image, _ = self._load_im(depth_file, color=self.cfg.depth_bg)
            depth_images.append(depth_image)


            # mask image
            mask_images.append(alpha)

        return rgb_images, normal_images, depth_images, mask_images
        
class MultiviewMultipromptDualRendererSemiSupervisedDataModule4Test(BaseDataset):
    def __init__(
            self, 
            cfg: Any, 
            unsup_prompt_library: List, 
            sup_obj_library: List,
            guidance_processor = None,
            condition_processor = None,
            split: str = "val" # "test"
        ) -> None:
        super().__init__(
            cfg, 
            unsup_prompt_library, 
            sup_obj_library,
            guidance_processor,
            condition_processor
        )
        self.split = split

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        ##############################################################################################################
        # the following is the conventional way to update along with the training process
        super().update_step(epoch, global_step, on_load_weights)

    def __len__(self) -> int:
        return self.unsup_length + self.sup_length
    
    def __getitem__(self, idx: int) -> Dict:
        # load the prompt
        #  [0, unsup_length) is unsupervised data, [unsup_length, unsup_length + sup_length) is supervised data]
        if idx < self.unsup_length:

            # set the number of views
            n_views = self.cfg.n_val_views if self.split == "val" else self.cfg.n_test_views

            ##############################################################################################################
            # arrange the azimuth angles
            azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360.0, n_views)
            # azimuth = azimuth_deg * math.pi / 180

            ##############################################################################################################
            # set the elevation angles
            elevation_deg: Float[Tensor, "B"] = torch.full_like(
                azimuth_deg, self.cfg.eval_elevation_deg
            )
            # elevation = elevation_deg * math.pi / 180

            ##############################################################################################################
            # set the fovs
            fovy_deg: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_fovy_deg
            )
            # fovy = fovy_deg * math.pi / 180

            ##############################################################################################################
            # set the camera distances
            camera_distances: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_camera_distance
            )

            prompt = self.unsup_prompt_library[idx]
            return {
                "prompt": prompt,
                "guidance_utils": self.guidance_processor(prompts = prompt),
                "condition_utils": self.condition_processor(prompts = prompt),
                "azimuths_deg": azimuth_deg,
                "elevations_deg": elevation_deg,
                "distances": camera_distances,
                "fovys_deg": fovy_deg
            }
        
        else:
            idx = idx - self.unsup_length
            #  get the idx-th prompt, self.sup_obj_library is a dict
            obj_name, obj_attributes = list(self.sup_obj_library.items())[idx]
            # we mainly need the caption from the obj_attributes
            prompt = obj_attributes["caption"]

            ############################################################################################################
            # select the views to load
            n_view = len(
                os.listdir(
                    os.path.join(
                        self.cfg.obj_library_dir,
                        self.cfg.obj_library,
                        self.cfg.rgb_data_dir,
                        obj_name
                    )
                )
            )

            # we should load all the views
            load_indices = np.arange(
                self.cfg.frontal_idx,
                self.cfg.frontal_idx + n_view
            ) % n_view

            ##############################################################################################################
            # load camera pose
            azimuths_deg = np.arange(0, 360, 360 / n_view, dtype=np.float32)
            
            with open(
                    os.path.join(
                        self.cfg.obj_library_dir,
                        self.cfg.obj_library,
                        self.cfg.camera_data_dir, 
                        obj_name,
                        "extrinsics.json"
                    ), 
                    "r"
                ) as f:
                camera_data = json.load(f)["000.png"]
                # only need to load the elevation, distance, fovy
                # all views share the same these parameters
                elevations_deg = torch.as_tensor([90 - camera_data["elevation"]] * n_view, dtype=torch.float32) # elevation should be in (-90, 90)
                distances = torch.as_tensor([camera_data["distance"]] * n_view, dtype=torch.float32)
                fovys_deg = torch.as_tensor([camera_data["fov"]] * n_view, dtype=torch.float32)


            ##############################################################################################################
            # load images
            rgb_imgs, normal_imgs, depth_imgs, mask_imgs = self._load_images(
                load_indices,
                rgb_data_dir=os.path.join(
                    self.cfg.obj_library_dir,
                    self.cfg.obj_library,
                    self.cfg.rgb_data_dir, 
                    obj_name
                ),
                normal_data_dir=os.path.join(
                    self.cfg.obj_library_dir,
                    self.cfg.obj_library,
                    self.cfg.normal_data_dir, 
                    obj_name
                ),
                depth_data_dir=os.path.join(
                    self.cfg.obj_library_dir,
                    self.cfg.obj_library,
                    self.cfg.depth_data_dir, 
                    obj_name
                )
            )

            return {
                "prompt": prompt,
                "guidance_utils": self.guidance_processor(prompts = prompt),
                "condition_utils": self.condition_processor(prompts = prompt),
                # ground truth images
                "rgb_imgs": torch.stack(rgb_imgs),
                "normal_imgs": torch.stack(normal_imgs),
                "depth_imgs": torch.stack(depth_imgs),
                "mask_imgs": torch.stack(mask_imgs),
                # camera data
                "azimuths_deg": azimuths_deg,
                "elevations_deg": elevations_deg,
                "distances": distances,
                "fovys_deg": fovys_deg
            }
        
    def collate(self, batch) -> Dict[str, Any]:
        n_views = self.cfg.n_val_views if self.split == "val" else self.cfg.n_test_views
        real_batch_size = self.cfg.eval_batch_size

        # the following items are applied to both supervised and unsupervised data
        batch = super().collate(batch)
        batch.update(
            self._create_camera_from_angle(
                elevation_deg=batch.pop("elevations_deg").view(-1), # the following items are popped
                azimuth_deg=batch.pop("azimuths_deg").view(-1),
                camera_distances=batch.pop("distances").view(-1),
                fovy_deg=batch.pop("fovys_deg").view(-1),
                relative_radius=False,
                phase="test"
            )
        )
        
        batch.update(
            {
                "index": torch.arange(n_views),
                "noise": torch.randn(real_batch_size, *self.cfg.dim_gaussian).view(-1, *self.cfg.dim_gaussian[1:]) \
                    if not self.cfg.pure_zeros \
                        else torch.zeros(real_batch_size, *self.cfg.dim_gaussian).view(-1, *self.cfg.dim_gaussian[1:]) 
            }
        )

        return batch


class MultiviewMultipromptDualRendererSemiSupervisedDataModule4Training(BaseDataset):
    def __init__(
            self, 
            cfg: Any, 
            unsup_prompt_library: List, 
            sup_obj_library: List,
            guidance_processor = None,
            condition_processor = None,
        ) -> None:
        super().__init__(
            cfg, 
            unsup_prompt_library, 
            sup_obj_library,
            guidance_processor,
            condition_processor
        )

        ##############################################################################################################
        # decide the schedule of choosing supervised and unsupervised data
        assert self.cfg.sup_unsup_mode in ["50/50", "vanilla"]
        if self.cfg.sup_unsup_mode == "50/50":
            self.data_schedule = ["sup", "unsup"]
        elif self.cfg.sup_unsup_mode == "vanilla":
            data_schedule = ["unsup"] * 100
            # insert the supervised data
            sup_ratio = self.sup_length / (self.sup_length + self.unsup_length)
            # the number of supervised data in each 100 samples
            # choose a deterministic way to insert the supervised data
            sup_interval = int(100 * sup_ratio)
            sup_indices = np.arange(0, 100, sup_interval) if sup_interval > 0 else []
            for i in sup_indices:
                data_schedule[i] = "sup"
            # the data schedule is fixed with [sup, unsup, sup, unsup, ...]
            self.data_schedule = data_schedule

        self.sup_or_unsup = "unsup" # the initial value, will be updated in update_step

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        ##############################################################################################################
        # the following is the conventional way to update along with the training process
        super().update_step(epoch, global_step, on_load_weights)

        ##############################################################################################################
        # the following related to the schedule of choosing supervised and unsupervised data
        self.sup_or_unsup = self.data_schedule[global_step % len(self.data_schedule)]

    def __len__(self) -> int:
        return self.unsup_length + self.sup_length
    
    def __getitem__(self, idx: int) -> Dict:
        # load the prompt
        assert hasattr(self, "sup_or_unsup")
        assert self.sup_or_unsup in ["sup", "unsup"]

        if self.sup_or_unsup == "sup":

            idx = np.random.randint(0, self.sup_length)
            #  get the idx-th prompt, self.sup_obj_library is a dict
            obj_name, obj_attributes = list(self.sup_obj_library.items())[idx]
            # we mainly need the caption from the obj_attributes
            prompt = obj_attributes["caption"]

            #################################################################################################
            #  decide the views to load 
            n_view = len(
                os.listdir(
                    os.path.join(
                        self.cfg.obj_library_dir,
                        self.cfg.obj_library,
                        self.cfg.rgb_data_dir,
                        obj_name
                    )
                )
            )
            azimuth_interval = 360 / n_view # generally 360 / 32 = 11.25
            indice_interval = n_view // self.cfg.n_view # generally 32 / 4 = 8

            all_azimuths = np.arange(0, 360, azimuth_interval, dtype=np.float32)
            all_indices = np.arange(
                self.cfg.frontal_idx, 
                self.cfg.frontal_idx + n_view
            ) % n_view
            first_azimuth = (random.uniform(0, 1) / self.cfg.n_view * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[0]) % 360 # follow MVDream
            first_indices = np.argmin(np.abs(all_azimuths - first_azimuth)) # find the indice with the closest azimuth
            load_indices = []
            # find all the indices to load
            for i in range(self.cfg.n_view):
                load_indices.append(
                    all_indices[
                        (first_indices + i * indice_interval) % n_view
                        ]
                    )

            ##############################################################################################################
            # load camera pose
            azimuths_deg = all_azimuths[load_indices]
            with open(
                    os.path.join(
                        self.cfg.obj_library_dir,
                        self.cfg.obj_library,
                        self.cfg.camera_data_dir, 
                        obj_name,
                        "extrinsics.json"
                    ), 
                    "r"
                ) as f:
                camera_data = json.load(f)["000.png"] # sort of hard-coded
                # only need to load the elevation, distance, fovy
                # all views share the same these parameters
                elevations_deg = torch.as_tensor([90 - camera_data["elevation"]] * self.cfg.n_view, dtype=torch.float32) # elevation should be in (-90, 90)
                distances = torch.as_tensor([camera_data["distance"]] * self.cfg.n_view, dtype=torch.float32)
                fovys_deg = torch.as_tensor([camera_data["fov"]] * self.cfg.n_view, dtype=torch.float32)
                
            ##############################################################################################################
            # load images
            rgb_imgs, normal_imgs, depth_imgs, mask_imgs = self._load_images(
                load_indices,
                rgb_data_dir=os.path.join(
                    self.cfg.obj_library_dir,
                    self.cfg.obj_library,
                    self.cfg.rgb_data_dir, 
                    obj_name
                ),
                normal_data_dir=os.path.join(
                    self.cfg.obj_library_dir,
                    self.cfg.obj_library,
                    self.cfg.normal_data_dir, 
                    obj_name
                ),
                depth_data_dir=os.path.join(
                    self.cfg.obj_library_dir,
                    self.cfg.obj_library,
                    self.cfg.depth_data_dir, 
                    obj_name
                )
            )

            ##############################################################################################################
            return {
                "prompt": prompt,
                "guidance_utils": self.guidance_processor(prompts = prompt),
                "condition_utils": self.condition_processor(prompts = prompt),
                # ground truth images
                "rgb_imgs": torch.stack(rgb_imgs),
                "normal_imgs": torch.stack(normal_imgs),
                "depth_imgs": torch.stack(depth_imgs),
                "mask_imgs": torch.stack(mask_imgs),
                # camera data
                "azimuths_deg": azimuths_deg,
                "elevations_deg": elevations_deg,
                "distances": distances,
                "fovys_deg": fovys_deg
            }


        elif self.sup_or_unsup == "unsup":

            real_batch_size = 1
            return_list = []

            #################################################################################################
            # sample the prompt
            idx = np.random.randint(0, self.unsup_length)
            prompt = self.unsup_prompt_library[idx]
            
            # loop for n_steps
            for i in range(self.cfg.n_steps):
                    
                # generate camera data for n_steps batches
                #################################################################################################
                # sample elevation angles
                elevation_deg: Float[Tensor, "B"] = (
                        torch.rand(real_batch_size)
                        * (self.elevation_range[1] - self.elevation_range[0])
                        + self.elevation_range[0]
                ).repeat_interleave(self.cfg.n_view, dim=0)
                # elevation: Float[Tensor, "B"] = elevation_deg * math.pi / 180

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
                # azimuth: Float[Tensor, "B"] = azimuth_deg * math.pi / 180

                ##############################################################################################################
                # sample fovs from a uniform distribution bounded by fov_range
                fovy_deg: Float[Tensor, "B"] = (
                    torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
                    + self.fovy_range[0]
                ).repeat_interleave(self.cfg.n_view, dim=0)
                # fovy = fovy_deg * math.pi / 180

                ##############################################################################################################
                # sample distances from a uniform distribution bounded by distance_range
                camera_distances: Float[Tensor, "B"] = (
                    torch.rand(real_batch_size)
                    * (self.camera_distance_range[1] - self.camera_distance_range[0])
                    + self.camera_distance_range[0]
                ).repeat_interleave(self.cfg.n_view, dim=0)

                return_list.append(
                    {
                        "prompt": prompt,
                        "guidance_utils": self.guidance_processor(prompts = prompt),
                        "condition_utils": self.condition_processor(prompts = prompt),
                        # camera data
                        "azimuths_deg": azimuth_deg,
                        "elevations_deg": elevation_deg,
                        "distances": camera_distances,
                        "fovys_deg": fovy_deg
                    }
                )
            return return_list
        
    def collate(self, batch_list) -> Dict[str, Any]:

        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view


        if self.sup_or_unsup == "sup":
            raise ValueError("The supervised data should be used in a single step")
            # # ground truth images and camera data should be stacked
            # batch = super().collate(batch)
            # batch.update(
            #     self._create_camera_from_angle(
            #         elevation_deg=batch.pop("elevations_deg").view(-1), # the following items are popped
            #         azimuth_deg=batch.pop("azimuths_deg").view(-1),
            #         camera_distances=batch.pop("distances").view(-1),
            #         fovy_deg=batch.pop("fovys_deg").view(-1),
            #         relative_radius=False,
            #         phase="train"
            #     )
            # )
        else:
            return_list = []
            for i in range(self.cfg.n_steps):
                batch = super().collate([sup_batch[i] for sup_batch in batch_list])
                batch.update(
                    self._create_camera_from_angle(
                        elevation_deg=batch.pop("elevations_deg").view(-1), # the following items are popped
                        azimuth_deg=batch.pop("azimuths_deg").view(-1),
                        camera_distances=batch.pop("distances").view(-1),
                        fovy_deg=batch.pop("fovys_deg").view(-1),
                        relative_radius=self.cfg.relative_radius,
                        phase="train"
                    )
                )

                if i == 0: # only add the following items once
                    # the following items are applied to both supervised and unsupervised data
                    batch.update(
                        {
                            "noise": torch.randn(real_batch_size, *self.cfg.dim_gaussian).view(-1, *self.cfg.dim_gaussian[1:]) \
                                if not self.cfg.pure_zeros \
                                    else torch.zeros(real_batch_size, *self.cfg.dim_gaussian).view(-1, *self.cfg.dim_gaussian[1:]) 
                        }
                    )
                return_list.append(batch)

        return return_list
            
            
@register("multiview-multiprompt-dualrenderer-multistep-datamodule-v2-fix")
class MultiviewMultipromptDualRendererMultiStepDataModuleFix(pl.LightningDataModule):
    cfg: MultiviewMultipromptDualRendererMultiStepDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewMultipromptDualRendererMultiStepDataModuleConfig, cfg)
        ##############################################################################################################
        # load the prompt library
        path = os.path.join(
            self.cfg.prompt_library_dir, 
            self.cfg.prompt_library) \
                + "." + self.cfg.prompt_library_format
        with open(path, "r") as f:
            self.unsup_prompt_library = json.load(f)
        
        ##############################################################################################################
        # load the meta json of the supervised data
        if self.cfg.obj_library.lower() == 'none':
            self.sup_obj_library = {
                "train": {},
                "val": {},
                "test": {}
            }
        else:
            path = os.path.join(
                self.cfg.obj_library_dir, 
                self.cfg.obj_library,
                self.cfg.meta_json
            )
            with open(path, "r") as f:
                self.sup_obj_library = json.load(f)

        ##############################################################################################################
        self.num_workers = 2 # for debugging
        self.pin_memory = False
        self.prefetch_factor = 2 if self.num_workers > 0 else None

        # print the info
        negative_prompt = self.cfg.guidance_processor.negative_prompt
        negative_prompt_2nd = self.cfg.guidance_processor.negative_prompt_2nd if hasattr(self.cfg.guidance_processor, "negative_prompt_2nd") else None
        info = f"Using prompt library located in [{self.cfg.prompt_library}] and obj dataset in [{self.cfg.obj_library}], \n with egative prompt [{negative_prompt}]"
        if negative_prompt_2nd is not None:
            info += f" and 2nd negative prompt [{negative_prompt_2nd}]"
        threestudio.info(info)  

        negative_prompt = self.cfg.condition_processor.negative_prompt
        info = f"Using condition processor with negative prompt [{negative_prompt}]"
        threestudio.info(info)


    def setup(self, stage: Optional[str] = None) -> None:

        # load the prompt processor after the ddp is initialized
        self.guidance_processor = threestudio.find(self.cfg.guidance_processor_type)(
            self.cfg.guidance_processor
        )
        self.condition_processor = threestudio.find(self.cfg.condition_processor_type)(
            self.cfg.condition_processor
        )

        if stage in (None, "fit"):
            # prepare the dataset
            prompt_lists = self.unsup_prompt_library["train"] \
                + self.unsup_prompt_library["val"] \
                + self.unsup_prompt_library["test"] \
                + [obj['caption'] for obj in self.sup_obj_library["train"].values()] \
                + [obj['caption'] for obj in self.sup_obj_library["val"].values()] \
                + [obj['caption'] for obj in self.sup_obj_library["test"].values()]
            self.condition_processor.prepare_text_embeddings(
                prompt_lists
            )
            self.guidance_processor.prepare_text_embeddings(
                prompt_lists
            )
            
            self.train_dataset = MultiviewMultipromptDualRendererSemiSupervisedDataModule4Training(
                self.cfg, 
                unsup_prompt_library= self.unsup_prompt_library["train"],
                sup_obj_library=self.sup_obj_library["train"],
                guidance_processor=self.guidance_processor,
                condition_processor=self.condition_processor
            )

        if stage in (None, "fit", "validate"):
            # prepare the dataset
            prompt_lists = self.unsup_prompt_library["val"] \
                + [obj['caption'] for obj in self.sup_obj_library["val"].values()]
            self.condition_processor.prepare_text_embeddings(
                prompt_lists
            )
            self.guidance_processor.prepare_text_embeddings(
                prompt_lists
            )

            self.val_dataset = MultiviewMultipromptDualRendererSemiSupervisedDataModule4Test(
                self.cfg, 
                unsup_prompt_library= self.unsup_prompt_library["val"],
                sup_obj_library=self.sup_obj_library["val"],
                guidance_processor=self.guidance_processor,
                condition_processor=self.condition_processor,
                split="val"
            )

        if stage in (None, "test", "predict"):
            # prepare the dataset
            if self.cfg.eval_prompt is not None:
                # fix the prompt during evaluation
                raise NotImplementedError
            else:
                prompt_lists = self.unsup_prompt_library["test"] \
                    + [obj['caption'] for obj in self.sup_obj_library["test"].values()]
            self.condition_processor.prepare_text_embeddings(
                prompt_lists
            )
            self.guidance_processor.prepare_text_embeddings(
                prompt_lists
            )
            
            self.test_dataset = MultiviewMultipromptDualRendererSemiSupervisedDataModule4Test(
                self.cfg, 
                unsup_prompt_library= self.unsup_prompt_library["test"],
                sup_obj_library=self.sup_obj_library["test"],
                guidance_processor=self.guidance_processor,
                condition_processor=self.condition_processor,
                split="test"
            )

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size // self.cfg.n_view,
            collate_fn=self.train_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            collate_fn=self.val_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            collate_fn=self.test_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            collate_fn=self.test_dataset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
        )
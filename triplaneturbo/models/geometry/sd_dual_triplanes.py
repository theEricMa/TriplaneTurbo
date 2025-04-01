import os
from dataclasses import dataclass, field
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from jaxtyping import Float
from torch import Tensor

from ...utils.general_utils import (
    config_to_primitive,
    contract_to_unisphere_custom,
    sample_from_planes,
)
from ..networks import get_mlp


@dataclass
class StableDiffusionTriplaneDualAttentionConfig:
    n_feature_dims: int = 3
    space_generator_config: dict = field(
        default_factory=lambda: {
            "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
            "training_type": "self_lora_rank_16-cross_lora_rank_16-locon_rank_16",
            "output_dim": 32,
            "gradient_checkpoint": False,
            "self_lora_type": "hexa_v1",
            "cross_lora_type": "hexa_v1",
            "locon_type": "vanilla_v1",
        }
    )

    mlp_network_config: dict = field(
        default_factory=lambda: {
            "otype": "VanillaMLP",
            "activation": "ReLU",
            "output_activation": "none",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
    )

    backbone: str = "one_step_triplane_dual_stable_diffusion"
    finite_difference_normal_eps: Union[
        float, str
    ] = 0.01  # in [float, "progressive"]    finite_difference_normal_eps: Union[float, str] = 0.01
    sdf_bias: Union[float, str] = 0.0
    sdf_bias_params: Optional[Any] = None

    isosurface_remove_outliers: bool = False
    # rotate planes to fit the conventional direction of image generated by SD
    # in right-handed coordinate system
    # xy plane should looks that a img from top-down / bottom-up view
    # xz plane should looks that a img from right-left / left-right view
    # yz plane should looks that a img from front-back / back-front view
    rotate_planes: Optional[str] = None
    split_channels: Optional[str] = None

    geo_interpolate: str = "v1"
    tex_interpolate: str = "v1"

    isosurface_deformable_grid: bool = True


class StableDiffusionTriplaneDualAttention(nn.Module):
    def __init__(
        self,
        config: StableDiffusionTriplaneDualAttentionConfig,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
    ):
        super().__init__()

        self.cfg = (
            StableDiffusionTriplaneDualAttentionConfig(**config)
            if isinstance(config, dict)
            else config
        )

        # set up the space generator
        from ...extern.sd_dual_triplane_modules import (
            OneStepTriplaneDualStableDiffusion as Generator,
        )

        self.space_generator = Generator(
            self.cfg.space_generator_config,
            vae=vae,
            unet=unet,
        )

        input_dim = self.space_generator.output_dim  # feat_xy + feat_xz + feat_yz
        assert self.cfg.split_channels in [None, "v1"]
        if self.cfg.split_channels in ["v1"]:  # split geometry and texture
            input_dim = input_dim // 2

        assert self.cfg.geo_interpolate in ["v1", "v2"]
        if self.cfg.geo_interpolate in ["v2"]:
            geo_input_dim = input_dim * 3  # concat[feat_xy, feat_xz, feat_yz]
        else:
            geo_input_dim = input_dim  # feat_xy + feat_xz + feat_yz

        assert self.cfg.tex_interpolate in ["v1", "v2"]
        if self.cfg.tex_interpolate in ["v2"]:
            tex_input_dim = input_dim * 3  # concat[feat_xy, feat_xz, feat_yz]
        else:
            tex_input_dim = input_dim  # feat_xy + feat_xz + feat_yz

        self.sdf_network = get_mlp(
            geo_input_dim,
            1,
            self.cfg.mlp_network_config,
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                tex_input_dim,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.isosurface_deformable_grid:
            self.deformation_network = get_mlp(
                geo_input_dim,
                3,
                self.cfg.mlp_network_config,
            )

        # hard-coded for now
        self.unbounded = False
        radius = 1.0

        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-radius, -radius, -radius],
                    [radius, radius, radius],
                ],
                dtype=torch.float32,
            ),
        )

    def initialize_shape(self) -> None:
        # not used
        pass

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def generate_space_cache(
        self,
        styles: Float[Tensor, "B Z"],
        text_embed: Float[Tensor, "B C"],
    ) -> Any:
        output = self.space_generator(
            text_embed=text_embed,
            styles=styles,
        )
        return output

    def denoise(
        self, noisy_input: Any, text_embed: Float[Tensor, "B C"], timestep
    ) -> Any:
        output = self.space_generator.forward_denoise(
            text_embed=text_embed, noisy_input=noisy_input, t=timestep
        )
        return output

    def decode(
        self,
        latents: Any,
    ) -> Any:
        triplane = self.space_generator.forward_decode(latents=latents)
        if self.cfg.split_channels == None:
            return triplane
        elif self.cfg.split_channels == "v1":
            B, _, C, H, W = triplane.shape
            # geometry triplane uses the first n_feature_dims // 2 channels
            # texture triplane uses the last n_feature_dims // 2 channels
            used_indices_geo = torch.tensor(
                [True] * (self.space_generator.output_dim // 2)
                + [False] * (self.space_generator.output_dim // 2)
            )
            used_indices_tex = torch.tensor(
                [False] * (self.space_generator.output_dim // 2)
                + [True] * (self.space_generator.output_dim // 2)
            )
            used_indices = torch.stack(
                [used_indices_geo] * 3 + [used_indices_tex] * 3, dim=0
            ).to(triplane.device)
            return triplane[:, used_indices].view(B, 6, C // 2, H, W)

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
        only_geo: bool = False,
    ):
        batch_size, n_points, n_dims = points.shape
        # the following code is similar to EG3D / OpenLRM

        assert self.cfg.rotate_planes in [None, "v1", "v2"]

        if self.cfg.rotate_planes == None:
            raise NotImplementedError("rotate_planes == None is not implemented yet.")

        space_cache_rotated = torch.zeros_like(space_cache)
        if self.cfg.rotate_planes == "v1":
            # xy plane, diagonal-wise
            space_cache_rotated[:, 0::3] = torch.transpose(space_cache[:, 0::3], 3, 4)
            # xz plane, rotate 180° counterclockwise
            space_cache_rotated[:, 1::3] = torch.rot90(
                space_cache[:, 1::3], k=2, dims=(3, 4)
            )
            # zy plane, rotate 90° clockwise
            space_cache_rotated[:, 2::3] = torch.rot90(
                space_cache[:, 2::3], k=-1, dims=(3, 4)
            )
        elif self.cfg.rotate_planes == "v2":
            # all are the same as v1, except for the xy plane
            # xy plane, row-wise flip
            space_cache_rotated[:, 0::3] = torch.flip(space_cache[:, 0::3], dims=(4,))
            # xz plane, rotate 180° counterclockwise
            space_cache_rotated[:, 1::3] = torch.rot90(
                space_cache[:, 1::3], k=2, dims=(3, 4)
            )
            # zy plane, rotate 90° clockwise
            space_cache_rotated[:, 2::3] = torch.rot90(
                space_cache[:, 2::3], k=-1, dims=(3, 4)
            )

        # the 0, 1, 2 axis of the space_cache_rotated is for geometry
        geo_feat = sample_from_planes(
            plane_features=space_cache_rotated[:, 0:3].contiguous(),
            coordinates=points,
            interpolate_feat=self.cfg.geo_interpolate,
        ).view(*points.shape[:-1], -1)

        if only_geo:
            return geo_feat
        else:
            # the 3, 4, 5 axis of the space_cache is for texture
            tex_feat = sample_from_planes(
                plane_features=space_cache_rotated[:, 3:6].contiguous(),
                coordinates=points,
                interpolate_feat=self.cfg.tex_interpolate,
            ).view(*points.shape[:-1], -1)

            return geo_feat, tex_feat

    def rescale_points(
        self,
        points: Float[Tensor, "*N Di"],
    ):
        # transform points from original space to [-1, 1]^3
        points = contract_to_unisphere_custom(points, self.bbox, self.unbounded)
        return points

    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Any,
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, n_points, n_dims = points.shape

        points_unscaled = points
        points = self.rescale_points(points)

        enc_geo, enc_tex = self.interpolate_encodings(points, space_cache)
        sdf_orig = self.sdf_network(enc_geo).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf_orig)
        output = {
            "sdf": sdf.view(batch_size * n_points, 1),  # reshape to [B*N, 1]
        }
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc_tex).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update(
                {
                    "features": features.view(
                        batch_size * n_points, self.cfg.n_feature_dims
                    )
                }
            )
        return output

    def forward_sdf(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Float[Tensor, "*N 1"]:
        batch_size = points.shape[0]
        assert (
            points.shape[0] == batch_size
        ), "points and space_cache should have the same batch size in forward_sdf"
        points_unscaled = points

        points = self.rescale_points(points)

        # sample from planes
        enc_geo = self.interpolate_encodings(
            points.reshape(batch_size, -1, 3), space_cache, only_geo=True
        ).reshape(*points.shape[:-1], -1)
        sdf = self.sdf_network(enc_geo).reshape(*points.shape[:-1], 1)

        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    def forward_field(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        batch_size = points.shape[0]
        assert (
            points.shape[0] == batch_size
        ), "points and space_cache should have the same batch size in forward_sdf"
        points_unscaled = points

        points = self.rescale_points(points)

        # sample from planes
        enc_geo = self.interpolate_encodings(points, space_cache, only_geo=True)
        sdf = self.sdf_network(enc_geo).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc_geo).reshape(
                *points.shape[:-1], 3
            )
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # TODO: is this function correct?
        return field - threshold

    def export(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
        **kwargs,
    ) -> Dict[str, Any]:
        # TODO: is this function correct?
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out

        orig_shape = points.shape
        points = points.view(1, -1, 3)

        # assume the batch size is 1
        points_unscaled = points
        points = self.rescale_points(points)

        # sample from planes
        _, enc_tex = self.interpolate_encodings(points, space_cache)
        features = self.feature_network(enc_tex).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {"features": features.view(orig_shape[:-1] + (self.cfg.n_feature_dims,))}
        )
        return out

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()

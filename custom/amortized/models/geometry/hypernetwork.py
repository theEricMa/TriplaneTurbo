import os
from dataclasses import dataclass, field
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *
from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp

class LinearHyperNetwork(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        config: dict,
    ):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.config = config

        self.c_dim = self.config["c_dim"]
        self.out_dims = self.config.get("out_dims", 
            {
                "sdf_weights": [64, 1],
                "feature_weights": [64, 3],
            }
        )

        for key, val in self.out_dims.items():
            if isinstance(val, ListConfig):
                val = [v for v in val] # convert to list
            if isinstance(val, list):
                self.out_dims[key] = [n_input_dims] + val
            else:
                self.out_dims[key] = [n_input_dims, val]

        self.spectral_norm = self.config["spectral_norm"]

        # update the n_input_dims in out_dims
        n_output_dims = 0
        for item, channels in self.out_dims.items():
            for in_channels, out_channels in zip(channels[:-1], channels[1:]):
                n_output_dims += in_channels * out_channels
        self.n_output_dims = n_output_dims

        self.n_neurons, self.n_hidden_layers, self.spectual_norm = (
            config["n_neurons"],
            config["n_hidden_layers"],
            config.get("spectral_norm", False),
        )

        layers = [
            self.make_linear(self.c_dim, self.n_neurons, is_first=True, is_last=False, has_bias=False),
            nn.LayerNorm(self.n_neurons),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False, has_bias=True
                ),
                nn.LayerNorm(self.n_neurons),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(
                self.n_neurons, self.n_output_dims, is_first=False, is_last=True, has_bias=True
            ),
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            out = self.layers(x)
            if self.output_activation is not None:
                out = self.output_activation(out)

        # split the output into different parts
        out_dict = {}
        start_idx = 0
        for item, channels in self.out_dims.items():
            params = []
            for in_channels, out_channels in zip(channels[:-1], channels[1:]):
                end_idx = start_idx + in_channels * out_channels
                params.append(out[:, start_idx:end_idx].reshape(*x.shape[:-1], in_channels, out_channels))
                start_idx = end_idx
            out_dict[item] = params

        return out_dict

    def make_linear(self, dim_in, dim_out, is_first, is_last, has_bias):
        layer = nn.Linear(dim_in, dim_out, bias=has_bias) if not self.spectral_norm \
            else nn.utils.spectral_norm(nn.Linear(dim_in, dim_out, bias=has_bias))
        if has_bias:
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(layer.weight, gain = 1.)
        return layer

    def make_activation(self):
        # as recommended in att3d
        return nn.SiLU(inplace=True) 


@threestudio.register("Hypernet-sdf")
class Hypernet_Sdf(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        hypernet_config: dict = field(
            default_factory=lambda: {
                "c_dim": 768,
                "out_dims": {
                    "sdf_weights": [64, 1],
                    "feature_weights": [64, 3],
                },
                "spectral_norm": False,
                "n_neurons": 64,
                "n_hidden_layers": 1,
                "output_activation": None,
            }
        )

        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )

        backbone: str = "linear_hypernetwork"

        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

    def configure(self) -> None:
        super().configure()

        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )

        # set up hypernet
        if self.cfg.backbone == "linear_hypernetwork":
            self.hypernet = LinearHyperNetwork(
                self.encoding.n_output_dims,
                self.cfg.hypernet_config,
            )
        else:
            raise NotImplementedError

        if self.cfg.normal_type == "pred":
            raise NotImplementedError("normal_type == pred is not implemented yet.")
        
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                input_dim,
                3,
                self.cfg.mlp_network_config,
            )

        self.finite_difference_normal_eps: Optional[float] = None


    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        raise NotImplementedError # TODO

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
        text_embed: Optional[Float[Tensor, "B T"]] = None,
    ) -> Any:
        # noises are not used in hypernet
        output_dict = self.hypernet(text_embed)
        return output_dict

    def hypernet_forward(
        self,
        enc: Float[Tensor, "*N C"],
        params: Float[Tensor, "*N C_out"],
        activation: Optional[Callable[[Float[Tensor, "*N C_out"]], Float[Tensor, "*N C_out"]]] = torch.relu,
        output_activation: Optional[Callable[[Float[Tensor, "*N C_out"]], Float[Tensor, "*N C_out"]]] = None,
    ) -> Float[Tensor, "*N 1"]:
        if torch.is_tensor(params):
            params = [params]
        
        for idx, p in enumerate(params):
            assert enc.shape[0] == p.shape[0]
            assert enc.shape[-1] == p.shape[1]
            enc = torch.bmm(enc, p) # no bias is used, as is tested in the original code

            # apply activation
            if activation is not None and idx < len(params) - 1:
                enc = activation(enc)
            elif output_activation is not None and idx == len(params) - 1:
                enc = output_activation(enc)

        return enc

    def forward(
        self, 
        points: Float[Tensor, "*N Di"], 
        space_cache: Dict,
        output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        batch_size, n_points, n_dims = points.shape
        points_unscaled = points
        points = contract_to_unisphere(
            points, 
            self.bbox, 
            self.unbounded
        )  # points normalized to (1, 1)

        if output_normal and self.cfg.normal_type == "analytic":
            raise NotImplementedError("analytic normal is not implemented yet.")

        enc = self.encoding(
                points.view(-1, self.cfg.n_input_dims)
            ).view(*points.shape[:-1], -1)
        sdf = self.hypernet_forward(enc, space_cache["sdf_weights"]) # (B, N, 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {
                "sdf": sdf.view(batch_size * n_points, 1) # reshape to [B*N, 1]
            }

        if self.cfg.n_feature_dims > 0:
            features = self.hypernet_forward(enc, space_cache["feature_weights"])
            output.update(
                    {
                        "features": features.view(batch_size * n_points, self.cfg.n_feature_dims) # reshape to [B*N, n_feature_dims]
                    }
                )
        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                    [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                ).to(points_unscaled)
                points_offset: Float[Tensor, "... 3 3"] = (
                    points_unscaled[..., None, :] + offsets
                ).clamp(-self.cfg.radius, self.cfg.radius)
                sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                    points_offset, space_cache
                )
                sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            else:
                raise NotImplementedError(
                    f"normal_type == {self.cfg.normal_type} is not implemented yet."
                )
            output.update(
                {
                    "normal": normal.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                    "shading_normal": normal.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                    "sdf_grad": sdf_grad.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                }
            )
        return output

    def forward_sdf(
        self, 
        points: Float[Tensor, "*N Di"],
        space_cache: Dict
    ) -> Float[Tensor, "*N 1"]:
        batch_size = points.shape[0]
        points_unscaled = points

        points = contract_to_unisphere(
            points_unscaled,
            self.bbox,
            self.unbounded
        ) # points normalized to (1, 1)

        enc = self.encoding(
            points.view(-1, self.cfg.n_input_dims)
        ).view(*points.shape[:-1], -1)

        sdf = self.hypernet_forward(
            enc.view(batch_size, -1, self.encoding.n_output_dims),
            space_cache["sdf_weights"]
        ).view(*points.shape[:-1], -1)

        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf
        
    def forward_field(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Dict,
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        batch_size = points.shape[0]
        points_unscaled = points

        points = contract_to_unisphere(
            points_unscaled,
            self.bbox,
            self.unbounded
        )

        enc = self.encoding(
            points.view(-1, self.cfg.n_input_dims)
        ).view(*points.shape[:-1], -1)
        sdf = self.hypernet_forward(enc, space_cache["sdf_weights"])
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
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
    **kwargs) -> Dict[str, Any]:
        # TODO: is this function correct?
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(
            points_unscaled, 
            self.bbox, 
            self.unbounded
        )

        enc = self.encoding(
            points.view(-1, self.cfg.n_input_dims)
        ).view(*points.shape[:-1], -1)
        features = self.hypernet_forward(
                enc, 
                space_cache["feature_weights"]
            ).view(*points.shape[:-1], self.cfg.n_feature_dims)

        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
        else:
            raise NotImplementedError(
                f"normal_type == {self.cfg.normal_type} is not implemented yet."
            )
        
            
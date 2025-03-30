
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

from ..geometry.hypernetwork import LinearHyperNetwork


@threestudio.register("multi-prompt-neural-hashgrid-environment-map-background")
class MultipromptNeuralHashgridEnvironmentMapBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 4,
                "per_level_scale": 1.8114473285278132, # desired resolution = 256
            }
        )
        hypernet_config: dict = field(
            default_factory=lambda: {
                "c_dim": 1024,
                "out_dims": {
                    "bg_weights": [64, 3],
                },
                "spectral_norm": False,
                "n_neurons": 64,
                "n_hidden_layers": 1,
                "output_activation": None,
            }
        )
        random_aug: bool = False
        random_aug_prob: float = 0.5
        random_aug_color: Optional[Tuple[float, float, float]] = None
        eval_color: Optional[Tuple[float, float, float]] = None

    cfg: Config

    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.pos_encoding_config)
        self.hypernet = LinearHyperNetwork(
            self.encoding.n_output_dims,
            self.cfg.hypernet_config,
        )
        # a flag to enable hypernetwork training
        self.enabling_hypernet = True

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
            # the number of views might be different from the number of parameters
            num_views = enc.shape[0] // p.shape[0]
            p = p.repeat_interleave(num_views, dim=0)
            
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
            dirs: Float[Tensor, "B H W 3"],
            text_embed: Optional[Float[Tensor, "B T"]] = None,
        ) -> Float[Tensor, "B H W Nc"]:
        batch_size, height, width, _ = dirs.shape

        if not self.training and self.cfg.eval_color is not None:
            return torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(
                dirs
            ) * torch.as_tensor(self.cfg.eval_color).to(dirs)
        
        bg_cache = self.hypernet(text_embed)

        # viewdirs must be normalized before passing to this function
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, 3))
        color = self.hypernet_forward(
                dirs_embd.view(batch_size, height*width, -1),
                bg_cache["bg_weights"],
            ).view(*dirs.shape[:-1], self.cfg.n_output_dims)
        
        color = get_activation(self.cfg.color_activation)(color)
        if (
            self.training
            and self.cfg.random_aug
            and random.random() < self.cfg.random_aug_prob
        ):
            random_color = torch.rand(dirs.shape[0], 1, 1, self.cfg.n_output_dims) \
                if self.cfg.random_aug_color is None \
                else torch.ones(dirs.shape[0], 1, 1, self.cfg.n_output_dims) * torch.tensor(self.cfg.random_aug_color)
            color = color * 0 + (  # prevent checking for unused parameters in DDP
                random_color
                .to(dirs)
                .expand(*dirs.shape[:-1], -1)
            )
        return color

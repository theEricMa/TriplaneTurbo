import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.general_utils import config_to_primitive
from dataclasses import dataclass
from typing import Optional, Literal

def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "sigmoid-mipnerf":
        return lambda x: torch.sigmoid(x) * (1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        # Convert dict to MLPConfig if needed
        if isinstance(config, dict):
            config = MLPConfig(**config)
            
        self.n_neurons = config.n_neurons
        self.n_hidden_layers = config.n_hidden_layers
        
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.output_activation)

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)

@dataclass
class MLPConfig:
    otype: str = "VanillaMLP"
    activation: str = "ReLU"
    output_activation: str = "none"
    n_neurons: int = 64
    n_hidden_layers: int = 2

def get_mlp(input_dim: int, output_dim: int, config: dict) -> nn.Module:
    """Create MLP network based on config"""
    # Convert dict to MLPConfig
    if isinstance(config, dict):
        config = MLPConfig(**config)

    if config.otype == "VanillaMLP":
        network = VanillaMLP(input_dim, output_dim, config)
    else:
        raise ValueError(f"Unknown MLP type: {config.otype}")
    return network
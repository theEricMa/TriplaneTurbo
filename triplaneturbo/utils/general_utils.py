from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import OmegaConf
from torch import Tensor


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def scale_tensor(
    dat: Float[Tensor, "... D"],
    inp_scale: Union[Tuple[float, float], Float[Tensor, "2 D"]],
    tgt_scale: Union[Tuple[float, float], Float[Tensor, "2 D"]],
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def contract_to_unisphere_custom(
    x: Float[Tensor, "... 3"], bbox: Float[Tensor, "2 3"], unbounded: bool = False
) -> Float[Tensor, "... 3"]:
    if unbounded:
        x = scale_tensor(x, bbox, (-1, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (-1, 1))
    return x


# bug fix in https://github.com/NVlabs/eg3d/issues/67
planes = torch.tensor(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    ],
    dtype=torch.float32,
)


def grid_sample(input, grid):
    # if grid.requires_grad and _should_use_custom_op():
    #     return grid_sample_2d(input, grid, padding_mode='zeros', align_corners=False)
    return torch.nn.functional.grid_sample(
        input=input,
        grid=grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = (
        coordinates.unsqueeze(1)
        .expand(-1, n_planes, -1, -1)
        .reshape(N * n_planes, M, 3)
    )
    inv_planes = (
        torch.linalg.inv(planes)
        .unsqueeze(0)
        .expand(N, -1, -1, -1)
        .reshape(N * n_planes, 3, 3)
    )
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(
    plane_features,
    coordinates,
    mode="bilinear",
    padding_mode="zeros",
    box_warp=2,
    interpolate_feat: Optional[str] = "None",
):
    assert padding_mode == "zeros"
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)

    coordinates = (2 / box_warp) * coordinates  # add specific box bounds

    if interpolate_feat in [None, "v1"]:
        projected_coordinates = project_onto_planes(
            planes.to(coordinates), coordinates
        ).unsqueeze(1)
        output_features = grid_sample(plane_features, projected_coordinates.float())
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features = output_features.sum(dim=1, keepdim=True).reshape(N, M, C)

    elif interpolate_feat in ["v2"]:
        projected_coordinates = project_onto_planes(
            planes.to(coordinates), coordinates
        ).unsqueeze(1)
        output_features = grid_sample(plane_features, projected_coordinates.float())
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features = output_features.permute(0, 2, 1, 3).reshape(
            N, M, n_planes * C
        )

    return output_features.contiguous()

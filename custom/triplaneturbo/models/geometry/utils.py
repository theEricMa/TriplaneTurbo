
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *

from ...extern.grid_sample_gradfix.cuda_gridsample import grid_sample_2d
#----------------------------------------------------------------------------

enabled = True  # Enable the custom op by setting this to true.

#----------------------------------------------------------------------------

def grid_sample(input, grid):
    if grid.requires_grad and _should_use_custom_op():
        return grid_sample_2d(input, grid, padding_mode='zeros', align_corners=False)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    return enabled

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
planes =  torch.tensor(
            [
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ],
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ]
            ], dtype=torch.float32)


quaplanes =  torch.tensor(
            [
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ],
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ],
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ] # the 4th plane is the same as the 3rd plane
            ], dtype=torch.float32)


Hplanes = torch.tensor(
            [
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ],
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ],
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ] # the 4th plane is the same as the 3rd plane
            ], dtype=torch.float32)


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
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=2, interpolate_feat: Optional[str] = 'None'):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # add specific box bounds

    if interpolate_feat in [None, "v1"]:
        projected_coordinates = project_onto_planes(planes.to(coordinates), coordinates).unsqueeze(1)
        output_features = grid_sample(plane_features, projected_coordinates.float())
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features = output_features.sum(dim=1, keepdim=True).reshape(N, M, C)

    elif interpolate_feat in ["v2"]:
        projected_coordinates = project_onto_planes(planes.to(coordinates), coordinates).unsqueeze(1)
        output_features = grid_sample(plane_features, projected_coordinates.float())
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features = output_features.permute(0, 2, 1, 3).reshape(N, M, n_planes*C)        

    elif interpolate_feat in ["v3"]:
        projected_coordinates = project_onto_planes(planes.to(coordinates), coordinates).unsqueeze(1)
        plane_features = torch.sigmoid(plane_features[:, -1:, ...]) * plane_features[:, :-1, ...]
        output_features = grid_sample(plane_features, projected_coordinates.float())
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C - 1)
        output_features = output_features.sum(dim=1, keepdim=True).reshape(N, M, C - 1)

    elif interpolate_feat in ["v4"]:
        projected_coordinates = project_onto_planes(planes.to(coordinates), coordinates).unsqueeze(1)
        plane_features =  torch.tanh(plane_features)
        output_features = grid_sample(plane_features, projected_coordinates.float())
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features = output_features.sum(dim=1, keepdim=True).reshape(N, M, C)

    return output_features.contiguous()

    

def sample_from_quaplanes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=2, interpolate_feat: Optional[str] = 'None'):
    assert padding_mode == 'zeros'
    assert interpolate_feat in [None, "v1"]

    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # add specific box bounds

    projected_coordinates = project_onto_planes(quaplanes.to(coordinates), coordinates).unsqueeze(1)
    # output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)
    output_features = grid_sample(plane_features, projected_coordinates.float())
    ooutput_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    
    # the following is from https://github.com/3DTopia/OpenLRM/blob/d4caebbea3f446904d9faafaf734e797fcc4ec89/lrm/models/rendering/synthesizer.py#L42
    output_features = output_features.permute(0, 2, 1, 3).reshape(N, M, n_planes*C)
    
    # decide whether to use front or back feature
    # front
    front_feat_idx = torch.cat(
        [
            torch.arange(0 * C, 1 * C), 
            torch.arange(1 * C, 2 * C), 
            torch.arange(2 * C, 3 * C)
        ]
    ).to(coordinates.device)
    # back
    back_feat_idx = torch.cat(
        [
            torch.arange(0 * C, 1 * C),
            torch.arange(1 * C, 2 * C),
            torch.arange(3 * C, 4 * C)
        ]
    ).to(coordinates.device)

    if not interpolate_feat:
        output_features_new = torch.zeros(N, M, (n_planes - 1) * C, device=coordinates.device)
        mask = coordinates[..., 0] > 0 # front has x > 0, back has x < 0
        output_features_new[mask]  = 1 * output_features[mask][..., front_feat_idx]# + 0 * output_features[mask][..., back_feat_idx]
        output_features_new[~mask] = 1 * output_features[~mask][..., back_feat_idx]# + 0 * output_features[~mask][..., front_feat_idx]
    else:
        alpha_front = 0.5 + 0.5 * coordinates[..., :1] # front has alpha_front > 0.5, alpha_back < 0.5
        alpha_back  = 0.5 - 0.5 * coordinates[..., :1] # back has alpha_back > 0.5, alpha_front < 0.5
        output_features_new = alpha_front * output_features[..., front_feat_idx] + alpha_back * output_features[..., back_feat_idx]

    return output_features_new.contiguous()

def sample_from_Hplanes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=2, interpolate_feat: Optional[str] = 'None'):
    assert padding_mode == 'zeros'
    assert interpolate_feat in [None, "v1", "v2", "v3", "v4", "v5", "v6", "v7"]

    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # add specific box bounds

    projected_coordinates = project_onto_planes(Hplanes.to(coordinates), coordinates).unsqueeze(1)
    # output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)
    output_features = grid_sample(plane_features, projected_coordinates.float())
    output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    
    if interpolate_feat in [None, "v1"]:
        # the following is from https://github.com/3DTopia/OpenLRM/blob/d4caebbea3f446904d9faafaf734e797fcc4ec89/lrm/models/rendering/synthesizer.py#L42
        output_features = output_features.permute(0, 2, 1, 3).reshape(N, M, n_planes*C)

        # decide whether to use front or back feature
        # front
        front_feat_idx = torch.cat(
            [
                torch.arange(0 * C, 1 * C), 
                torch.arange(1 * C, 2 * C)
            ]
        ).to(coordinates.device)
        # back
        back_feat_idx = torch.cat(
            [
                torch.arange(0 * C, 1 * C),
                torch.arange(2 * C, 3 * C)
            ]
        ).to(coordinates.device)


        if interpolate_feat in [None]:

            output_features_new = torch.zeros(N, M, (n_planes - 1) * C, device=coordinates.device)
            mask = coordinates[..., 0] > 0
            output_features_new[mask]  = 1 * output_features[mask][..., front_feat_idx]
            output_features_new[~mask] = 1 * output_features[~mask][..., back_feat_idx]

            return output_features_new.contiguous()

        elif interpolate_feat in ["v1"]:
            output_features = output_features.permute(0, 2, 1, 3).reshape(N, M, n_planes*C)

            alpha_front = 0.5 + 0.5 * coordinates[..., :1]
            alpha_back  = 0.5 - 0.5 * coordinates[..., :1]
            output_features_new = alpha_front * output_features[..., front_feat_idx] + alpha_back * output_features[..., back_feat_idx]

            return output_features_new.contiguous()

    elif interpolate_feat in ["v2"]:
        # the following is from v1
        alpha_front = 0.5 + 0.5 * coordinates[..., 0:1]
        alpha_back  = 0.5 - 0.5 * coordinates[..., 0:1]
        # the following is for v2
        alpha_side  = 1.0 - 1.0 * coordinates[..., 1:2].abs() # alpha = 1 - |y| for -1 <= y <= 1
        alpha = torch.cat([alpha_side, alpha_front, alpha_back], dim=-1).permute(0, 2, 1).unsqueeze(-1)
        output_features = (output_features * alpha).permute(0, 2, 1, 3).reshape(N, M, n_planes*C)

        return output_features.contiguous()

    elif interpolate_feat in ["v3"]:
        # front mask ########################################################
        front_thres = 0.5
        # part A: from front_thres to 1, decrease alpha from 1 to 0
        alpha_partA = 1 - (coordinates[..., 0:1] - front_thres) / (1 - front_thres)
        # part B: from -1 to front_thres, increase alpha from 0 to 1
        alpha_partB = (coordinates[..., 0:1] + 1) / (front_thres + 1)
        # determine the mask
        partA_mask = (coordinates[..., 0:1] > front_thres).float()
        # combine part A and part B
        alpha_front = alpha_partA * partA_mask + alpha_partB * (1 - partA_mask)
        # back mask ########################################################
        back_thres = -0.5
        # part A: from back_thres to 1, decrease alpha from 1 to 0
        alpha_partA = 1 - (coordinates[..., 0:1] - back_thres) / (1 - back_thres)
        # part B: from -1 to back_thres, increase alpha from 0 to 1
        alpha_partB = (coordinates[..., 0:1] + 1) / (back_thres + 1)
        # determine the mask
        partA_mask = (coordinates[..., 0:1] > back_thres).float()
        # combine part A and part B
        alpha_back = alpha_partA * partA_mask + alpha_partB * (1 - partA_mask)
        # side mask ########################################################
        alpha_side = torch.ones_like(coordinates[..., 1:2])
        # combine all masks
        alpha = torch.cat([alpha_side, alpha_front, alpha_back,], dim=-1).permute(0, 2, 1).unsqueeze(-1)
        output_features = (output_features * alpha).permute(0, 2, 1, 3).reshape(N, M, n_planes*C)

        return output_features.contiguous()

    elif interpolate_feat in ["v4"]:
        # the following is from v3
        # front mask ########################################################
        front_thres = 0.5
        # part A: from front_thres to 1, decrease alpha from 1 to 0
        alpha_partA = 1 - (coordinates[..., 0:1] - front_thres) / (1 - front_thres)
        # part B: from -1 to front_thres, increase alpha from 0 to 1
        alpha_partB = (coordinates[..., 0:1] + 1) / (front_thres + 1)
        # determine the mask
        partA_mask = (coordinates[..., 0:1] > front_thres).float()
        # combine part A and part B
        alpha_front = alpha_partA * partA_mask + alpha_partB * (1 - partA_mask)
        # back mask ########################################################
        back_thres = -0.5
        # part A: from back_thres to 1, decrease alpha from 1 to 0
        alpha_partA = 1 - (coordinates[..., 0:1] - back_thres) / (1 - back_thres)
        # part B: from -1 to back_thres, increase alpha from 0 to 1
        alpha_partB = (coordinates[..., 0:1] + 1) / (back_thres + 1)
        # determine the mask
        partA_mask = (coordinates[..., 0:1] > back_thres).float()
        # combine part A and part B
        alpha_back = alpha_partA * partA_mask + alpha_partB * (1 - partA_mask)
        # side mask ########################################################
        alpha_side = torch.ones_like(coordinates[..., 1:2])
        # we sum up the feat instead of concatenating
        alpha = torch.cat([alpha_side, alpha_front, alpha_back,], dim=-1).permute(0, 2, 1).unsqueeze(-1)
        output_features = (output_features * alpha).sum(dim=1, keepdim=True).reshape(N, M, C)

        return output_features.contiguous()
    
    elif interpolate_feat in ["v5"]:

        alpha_front = 0.5 + 0.5 * coordinates[..., 0:1]
        alpha_back  = 0.5 - 0.5 * coordinates[..., 0:1]
        alpha_side = torch.ones_like(coordinates[..., 1:2])
        alpha = torch.cat([alpha_side, alpha_front, alpha_back], dim=-1).permute(0, 2, 1).unsqueeze(-1)
        output_features = (output_features * alpha).sum(dim=1, keepdim=True).reshape(N, M, C)

        return output_features.contiguous()
    
    elif interpolate_feat in ["v6"]:

        alpha = torch.sigmoid(output_features[..., -1:])
        output_features = (output_features[..., :-1] * alpha).sum(dim=1, keepdim=True).reshape(N, M, C - 1)

        return output_features.contiguous()
    
    elif interpolate_feat in ["v7"]:
        bias_front = 0.5 + 0.5 * coordinates[..., 0:1]
        bias_back  = 0.5 - 0.5 * coordinates[..., 0:1]
        bias_side  = 0.0 + 0.0 * coordinates[..., 1:2]
        alpha_bias = torch.cat([bias_side, bias_front, bias_back], dim=-1).permute(0, 2, 1).unsqueeze(-1)
        alpha = torch.sigmoid(output_features[..., -1:]) + alpha_bias
        output_features = (output_features[..., :-1] * alpha).sum(dim=1, keepdim=True).reshape(N, M, C - 1)

        return output_features.contiguous()

def get_trilinear_feature(
        points: Float[Tensor, "*N Di"],
        voxel: Float[Tensor, "B Df G1 G2 G3"],
    ) -> Float[Tensor, "*N Df"]:
        b = voxel.shape[0]
        points_shape = points.shape[:-1]
        df = voxel.shape[1]
        di = points.shape[-1]
        out = F.grid_sample(
            voxel, points.view(b, 1, 1, -1, di), align_corners=False, mode="bilinear"
        )
        out = out.reshape(df, -1).T.reshape(*points_shape, df)
        return out

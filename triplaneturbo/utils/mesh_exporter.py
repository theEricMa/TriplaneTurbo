from typing import Callable, Dict, List, Optional, Tuple, Any
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass

import torch
import torch.nn as nn
import os
import numpy as np
from .saving import SaverMixin

from ..utils.mesh import Mesh
from ..utils.general_utils import scale_tensor

@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> Float[Tensor, "N 3"]:
        raise NotImplementedError
    
class DiffMarchingCubeHelper(IsosurfaceHelper):
    def __init__(
            self, 
            resolution: int, 
            point_range: Tuple[float, float] = (0, 1)
        ) -> None:
        super().__init__()
        self.resolution = resolution
        self.points_range = point_range

        from diso import DiffMC
        self.mc_func: Callable = DiffMC(dtype=torch.float32)
        self._grid_vertices: Optional[Float[Tensor, "N3 3"]] = None
        self._dummy: Float[Tensor, "..."]
        self.register_buffer(
            "_dummy", torch.zeros(0, dtype=torch.float32), persistent=False
        )

    @property
    def grid_vertices(self) -> Float[Tensor, "N3 3"]:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            verts = verts * (self.points_range[1] - self.points_range[0]) + self.points_range[0]

            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
        isovalue=0.0,
    ) -> Mesh:
        level = level.view(self.resolution, self.resolution, self.resolution)
        if deformation is not None:
            deformation = deformation.view(self.resolution, self.resolution, self.resolution, 3)
        v_pos, t_pos_idx = self.mc_func(level, deformation, isovalue=isovalue)
        v_pos = v_pos * (self.points_range[1] - self.points_range[0]) + self.points_range[0]
        # TODO: if the mesh is good
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)


def isosurface(
        space_cache: Float[Tensor, "B ..."], 
        forward_field: Callable,
        isosurface_helper: Callable,
    ) -> List[Mesh]:

    # the isosurface is dependent on the space cache
    # randomly detach isosurface method if it is differentiable
    # get the batchsize
    if torch.is_tensor(space_cache): #space cache
        batch_size = space_cache.shape[0]
    elif isinstance(space_cache, Dict): #hyper net
        # Dict[str, List[Float[Tensor, "B ..."]]]
        for key in space_cache.keys():
            batch_size = space_cache[key][0].shape[0]
            break

    # scale the points to [-1, 1]
    points = scale_tensor(
        isosurface_helper.grid_vertices.to(space_cache.device),
        isosurface_helper.points_range,
        [-1, 1], # hard coded isosurface_bbox
    )
    # get the sdf values    
    sdf_batch, deformation_batch = forward_field(
        points[None, ...].expand(batch_size, -1, -1),
        space_cache
    )
    
    # get the isosurface
    mesh_list = []

    # check if the sdf is empty
    # for sdf, deformation in zip(sdf_batch, deformation_batch):
    for index in range(sdf_batch.shape[0]):
        sdf = sdf_batch[index]

        # the deformation may be None
        if deformation_batch is None:
            deformation = None
        else:
            deformation = deformation_batch[index]

        # special case when all sdf values are positive or negative, thus no isosurface
        if torch.all(sdf > 0) or torch.all(sdf < 0):
            
            print(f"All sdf values are positive or negative, no isosurface")
            sdf = torch.norm(points, dim=-1) - 1

        mesh = isosurface_helper(sdf, deformation)
        
        mesh.v_pos = scale_tensor(
            mesh.v_pos,
            isosurface_helper.points_range,
            [-1, 1], # hard coded isosurface_bbox
        )

        # TODO: implement outlier removal        
        # if cfg.isosurface_remove_outliers:
        #     mesh = mesh.remove_outlier(cfg.isosurface_outlier_n_faces_threshold)

        mesh_list.append(mesh)
        
    return mesh_list

def colorize_mesh(
    space_cache: Any,
    export_fn: Callable,
    mesh_list: List[Mesh],
    activation: Callable,
) -> List[Mesh]:
    """Colorize the mesh using the geometry's export function and space cache.
    
    Args:
        space_cache: The space cache containing feature information
        export_fn: The export function from geometry that generates features
        mesh_list: List of meshes to colorize
        
    Returns:
        List[Mesh]: List of colorized meshes
    """
    # Process each mesh in the batch
    for i, mesh in enumerate(mesh_list):
        # Get vertex positions
        points = mesh.v_pos[None, ...]  # Add batch dimension [1, N, 3]
        
        # Get the corresponding space cache slice for this mesh
        if torch.is_tensor(space_cache):
            space_cache_slice = space_cache[i:i+1]
        elif isinstance(space_cache, dict):
            space_cache_slice = {}
            for key in space_cache.keys():
                space_cache_slice[key] = [
                    weight[i:i+1] for weight in space_cache[key]
                ]
        
        # Export features for the vertices
        out = export_fn(points, space_cache_slice)
        
        # Update vertex colors if features exist
        if "features" in out:
            features = out["features"].squeeze(0)  # Remove batch dim [N, C]
            # Convert features to RGB colors
            mesh._v_rgb = activation(features)  # Access private attribute directly
            
    return mesh_list

class MeshExporter(SaverMixin):
    def __init__(self, save_dir="outputs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def get_save_dir(self):
        return self.save_dir

    def get_save_path(self, filename):
        return os.path.join(self.save_dir, filename)

    def convert_data(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

def export_obj(
        mesh: Mesh, 
        save_path: str
    ) -> List[str]:
    """
    Export mesh data to OBJ file format.
    
    Args:
        mesh_data: Dictionary containing mesh data (vertices, faces, etc.)
        save_path: Path to save the OBJ file
        
    Returns:
        List of saved file paths
    """

    # Create exporter
    exporter = MeshExporter(os.path.dirname(save_path))
    
    # Export mesh
    save_paths = exporter.save_obj(
        os.path.basename(save_path),
        mesh,
        save_mat=None,
        save_normal=mesh.v_nrm is not None,
        save_uv=False,
        save_vertex_color=mesh.v_rgb is not None,
    )
    
    return save_paths


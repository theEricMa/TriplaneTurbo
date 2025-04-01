from dataclasses import dataclass
from functools import partial
from tqdm import tqdm

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial

from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer

from threestudio.utils.misc import get_device, C
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.models.mesh import Mesh

from threestudio.utils.ops import scale_tensor as scale_tensor

def c2wtow2c(c2w):
    """transfer camera2world to world2camera matrix"""

    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0

    return w2c

@threestudio.register("generative-space-dmtet-rasterize-renderer")
class GenerativeSpaceDmtetRasterizeRenderer(NVDiffRasterizer):
    @dataclass
    class Config(NVDiffRasterizer.Config):
        # the following are from NeuS #########
        isosurface_resolution: int = 128

        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

        context_type: str = "cuda"
        isosurface_method: str = "mt" # "mt" or "mc-cpu"

        enable_bg_rays: bool = False
        normal_direction: str = "camera" # "camera" or "world" or "front"

        # sdf forcing strategy for generative space
        sdf_grad_shrink: float = 1.
        
        def_grad_shrink: float = 1.

        allow_empty_flag: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        # overwrite the geometry
        self.geometry.isosurface = self.isosurface

        assert self.cfg.isosurface_method in ["mt", "mc-cpu", "diffmc"], "Invalid isosurface method"
        if self.cfg.isosurface_method == "mt":
            from threestudio.models.isosurface import MarchingTetrahedraHelper
            self.isosurface_helper = MarchingTetrahedraHelper(
                self.cfg.isosurface_resolution,
                f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
            )
        elif self.cfg.isosurface_method == "mc-cpu":
            from threestudio.models.isosurface import  MarchingCubeCPUHelper
            self.isosurface_helper = MarchingCubeCPUHelper(
                self.cfg.isosurface_resolution,
            )
        elif self.cfg.isosurface_method == "diffmc":
            from threestudio.models.isosurface import  DiffMarchingCubeHelper
            self.isosurface_helper = DiffMarchingCubeHelper(
                self.cfg.isosurface_resolution,
            )

        # detect if the sdf is empty
        self.empty_flag = False

        # follow InstantMesh 
        grid_res = self.cfg.isosurface_resolution

        v = torch.zeros([grid_res] * 3, dtype=torch.bool,)
        v[grid_res // 2:grid_res // 2 + 1, grid_res // 2:grid_res // 2 + 1, grid_res // 2:grid_res // 2 + 1] = True
        self.center_indices = torch.nonzero(v.reshape(-1)).to(self.device)

        v = torch.zeros([grid_res] * 3, dtype=torch.bool,)
        v[:2, :, :] = True; v[-2:, :, :] = True
        v[:, :2, :] = True; v[:, -2:, :] = True
        v[:, :, :2] = True; v[:, :, -2:] = True
        self.border_indices = torch.nonzero(v.reshape(-1)).to(self.device)


    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Float[Tensor, "B ..."]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        render_rgb: bool = True,
        rays_d_rasterize: Optional[Float[Tensor, "B H W 3"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        c2w: Optional[Float[Tensor, "B 4 4"]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        batch_size = mvp_mtx.shape[0]
        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size
        num_views_per_batch = batch_size // batch_size_space_cache

        if space_cache is None:
            space_cache = self.geometry.generate_space_cache(
                styles = noise,
                text_embed = text_embed,
            )

        mesh_list = self.isosurface(space_cache)

        # detect if the sdf is empty
        if self.empty_flag:
            is_emtpy = True
            self.empty_flag = False
        else:
            is_emtpy = False

        out_list = []
        # if render a space cache in multiple views,
        for batch_idx, mesh in enumerate(mesh_list):
            _mvp_mtx: Float[Tensor, "B 4 4"]  = mvp_mtx[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch]
            v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                mesh.v_pos, _mvp_mtx
            )

            # do rasterization
            if self.training: # requires only 4 views, memory efficient:
                rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
                gb_feat, _ = self.ctx.interpolate(v_pos_clip, rast, mesh.t_pos_idx)
                depth = gb_feat[..., -2:-1]
            else: # requires about 40 views, GPU OOM, need a for-loop to rasterize
                rast_list = []
                depth_list = []
                n_views_per_rasterize = 4
                for i in range(0, v_pos_clip.shape[0], n_views_per_rasterize):
                    rast, _ = self.ctx.rasterize(v_pos_clip[i:i+n_views_per_rasterize], mesh.t_pos_idx, (height, width))
                    rast_list.append(rast)
                    gb_feat, _ = self.ctx.interpolate(v_pos_clip[i:i+n_views_per_rasterize], rast, mesh.t_pos_idx)
                    depth_list.append(gb_feat[..., -2:-1])
                rast = torch.cat(rast_list, dim=0)
                depth = torch.cat(depth_list, dim=0)

            mask = rast[..., 3:] > 0

            # special case when no points are visible
            if mask.sum() == 0: # no visible points
                # set the mask to be the first point
                mask[:1] = True

            mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

            # disparity, as required by RichDreamer
            far= (
                camera_distances + 
                torch.sqrt(3 * torch.ones(1, device=camera_distances.device))
            )[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch, None, None, None]
            near = (
                camera_distances - 
                torch.sqrt(3 * torch.ones(1, device=camera_distances.device))
            )[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch, None, None, None]
            disparity_tmp = depth.clamp_max(far)
            disparity_norm = (far - disparity_tmp) / (far - near)
            disparity_norm = disparity_norm.clamp(0, 1)
            disparity_norm = torch.lerp(torch.zeros_like(depth), disparity_norm, mask.float())
            disparity_norm = self.ctx.antialias(disparity_norm, rast, v_pos_clip, mesh.t_pos_idx)

            out = {
                "opacity": mask_aa if not is_emtpy else mask.detach(),
                "mesh": mesh,
                "depth": depth if not is_emtpy else depth.detach(), 
                "disparity": disparity_norm if not is_emtpy else disparity_norm.detach(),
            }

            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal_aa = self.ctx.antialias(
                gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            )
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

            if self.cfg.normal_direction == "camera":
                # for compatibility with RichDreamer #############
                # camera_batch_v_nrm = torch.repeat_interleave(mesh.v_nrm[None, ...], num_views_per_batch, dim=0)
                bg_normal = 0.5 * torch.ones_like(gb_normal)
                bg_normal[..., 2] = 1.0
                bg_normal_white = torch.ones_like(gb_normal)

                # convert_normal_to_cam_space
                w2c: Float[Tensor, "B 4 4"] = torch.inverse(
                    c2w[
                        (batch_idx) * num_views_per_batch : 
                        (batch_idx + 1) * num_views_per_batch
                    ]
                )
                rotate: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]
                # camera_batch_v_nrm = camera_batch_v_nrm @ rotate.permute(0, 2, 1)
                # gb_normal: B H W 3 -> B H W 1 3; rotate: B 3 3 -> B 1 1 3 3
                gb_normal_cam = gb_normal[..., None, :] @ rotate.permute(0, 2, 1)[..., None, None, :, :]
                flip_x = torch.eye(3).to(w2c) # pixel space flip axis so we need built negative y-axis normal
                flip_x[0, 0] = -1
                # camera_batch_v_nrm = camera_batch_v_nrm @ flip_x[None, ...]
                # flip_x: B 3 3 -> B 1 1 3 3
                gb_normal_cam = gb_normal_cam @ flip_x[None, None, None, ...]
                gb_normal_cam = gb_normal_cam.squeeze(-2)
                gb_normal_cam = F.normalize(gb_normal_cam, dim=-1)
                gb_normal_cam = (gb_normal_cam + 1.0) / 2.0

                # camera_gb_normal, _ = self.ctx.interpolate(camera_batch_v_nrm, rast, mesh.t_pos_idx)
                # camera_gb_normal = F.normalize(camera_gb_normal, dim=-1)
                # camera_gb_normal = (camera_gb_normal + 1.0) / 2.0

                # render with bg_normal
                camera_gb_normal_bg = torch.lerp(
                    bg_normal, gb_normal_cam, mask.float()
                )
                camera_gb_normal_bg = self.ctx.antialias(
                    camera_gb_normal_bg, rast, v_pos_clip, mesh.t_pos_idx
                )

                # render with bg_normal_white
                camera_gb_normal_bg_white = torch.lerp(
                    bg_normal_white, gb_normal_cam, mask.float()
                )
                camera_gb_normal_bg_white = self.ctx.antialias(
                    camera_gb_normal_bg_white, rast, v_pos_clip, mesh.t_pos_idx
                )

                out.update({
                    "comp_normal_cam_vis": camera_gb_normal_bg if not is_emtpy else camera_gb_normal_bg.detach(),
                    "comp_normal_cam_vis_white": camera_gb_normal_bg_white if not is_emtpy else camera_gb_normal_bg_white.detach(),
                })

            elif self.cfg.normal_direction == "front":
                # for compatibility with Wonder3D and Era3D #############
                bg_normal_white = torch.ones_like(gb_normal)

                # convert_normal_to_cam_space of the front view
                c2w_front: Float[Tensor, "B 4 4"] = c2w[batch_idx * num_views_per_batch][None, ...].repeat(num_views_per_batch, 1, 1)
                w2c_front: Float[Tensor, "B 4 4"] = torch.inverse(c2w_front)
                rotate_front: Float[Tensor, "B 3 3"] = w2c_front[:, :3, :3]
                gb_normal_cam = gb_normal[..., None, :] @ rotate_front.permute(0, 2, 1)[..., None, None, :, :]

                # flip_x = torch.eye(3).to(w2c) # pixel space flip axis so we need built negative y-axis normal
                # flip_x[0, 0] = -1
                # # camera_batch_v_nrm = camera_batch_v_nrm @ flip_x[None, ...]
                # # flip_x: B 3 3 -> B 1 1 3 3
                # gb_normal_cam = gb_normal_cam @ flip_x[None, None, None, ...]
                
                gb_normal_cam = gb_normal_cam.squeeze(-2)
                gb_normal_cam = F.normalize(gb_normal_cam, dim=-1)
                gb_normal_cam = (gb_normal_cam + 1.0) / 2.0

                # render with bg_normal_white
                camera_gb_normal_bg_white = torch.lerp(
                    bg_normal_white, gb_normal_cam, mask.float()
                )
                camera_gb_normal_bg_white = self.ctx.antialias(
                    camera_gb_normal_bg_white, rast, v_pos_clip, mesh.t_pos_idx
                )

                out.update({
                    "comp_normal_cam_vis_white": camera_gb_normal_bg_white if not is_emtpy else camera_gb_normal_bg_white.detach(),
                })

            if render_rgb:

                # slice the space cache
                if torch.is_tensor(space_cache): #space cache
                    space_cache_slice = space_cache[batch_idx: batch_idx+1]
                elif isinstance(space_cache, Dict): #hyper net
                    # Dict[str, List[Float[Tensor, "B ..."]]]
                    space_cache_slice = {}
                    for key in space_cache.keys():
                        space_cache_slice[key] = [
                            weight[batch_idx: batch_idx+1] for weight in space_cache[key]
                        ]

                selector = mask[..., 0]

                gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
                gb_viewdirs = F.normalize(
                    gb_pos - camera_positions[
                        batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch,
                        None, None, :
                    ], dim=-1
                )
                gb_light_positions = light_positions[
                    batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch,
                    None, None, :
                ].expand(
                    -1, height, width, -1
                )

                positions = gb_pos[selector]

                # # special case when no points are selected
                # if positions.shape[0] == 0:                
                #     out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})
                #     continue

                geo_out = self.geometry(
                    positions[None, ...],
                    space_cache_slice,
                    output_normal= self.training, # only output normal and related info during training
                )

                extra_geo_info = {}
                if self.material.requires_normal:
                    extra_geo_info["shading_normal"] = gb_normal[selector] if not is_emtpy else gb_normal[selector].detach()
                
                if self.material.requires_tangent:
                    gb_tangent, _ = self.ctx.interpolate_one(
                        mesh.v_tng, rast, mesh.t_pos_idx
                    )
                    gb_tangent = F.normalize(gb_tangent, dim=-1)
                    extra_geo_info["tangent"] = gb_tangent[selector] if not is_emtpy else gb_tangent[selector].detach()

                # remove the following keys from geo_out
                geo_out.pop("shading_normal", None)

                # add sdf values for computing loss
                if "sdf_grad" in geo_out:
                    if "sdf_grad" in out:
                        out["sdf_grad"].extend(
                            geo_out["sdf_grad"]
                        )
                    else:
                        out["sdf_grad"] = geo_out[
                            "sdf_grad"
                        ]
                if "sdf" in geo_out:
                    if "sdf" in out:
                        out["sdf"].extend(
                            geo_out["sdf"]
                        )
                    else:
                        out["sdf"] = geo_out[
                            "sdf"
                        ]

                rgb_fg = self.material(
                    viewdirs=gb_viewdirs[selector],
                    positions=positions,
                    light_positions=gb_light_positions[selector],
                    **extra_geo_info,
                    **geo_out
                )
                gb_rgb_fg = torch.zeros(num_views_per_batch, height, width, 3).to(rgb_fg)
                gb_rgb_fg[selector] = rgb_fg


                # background
                if self.cfg.enable_bg_rays:
                    assert rays_d_rasterize is not None
                    view_dirs = rays_d_rasterize[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch]
                else:
                    view_dirs = gb_viewdirs

                if hasattr(self.background, "enabling_hypernet") and self.background.enabling_hypernet:
                    gb_rgb_bg = self.background(
                        dirs=view_dirs,
                        text_embed=text_embed if "text_embed_bg" not in kwargs else kwargs["text_embed_bg"]
                    )
                else:
                    gb_rgb_bg = self.background(
                        dirs=view_dirs,
                    )

                gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
                gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

                out.update(
                    {
                        "comp_rgb": gb_rgb_aa if not is_emtpy else gb_rgb_aa.detach(),
                        "comp_rgb_bg": gb_rgb_bg if not is_emtpy else gb_rgb_bg.detach(),
                    }
                )

            out_list.append(out)

        # stack the outputs
        out = {}
        for key in out_list[0].keys():
            if key not in ["mesh", "sdf_grad", "sdf"]: # hard coded for special case
                out[key] = torch.concat([o[key] for o in out_list], dim=0)
            else:
                out[key] = [o[key] for o in out_list]

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.sdf_grad_shrink = C(
            self.cfg.sdf_grad_shrink, epoch, global_step
        )
        self.def_grad_shrink = C(   
            self.cfg.def_grad_shrink, epoch, global_step
        )

    def isosurface(self, space_cache: Float[Tensor, "B ..."]) -> List[Mesh]:

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
            self.isosurface_helper.grid_vertices.to(self.device),
            self.isosurface_helper.points_range,
            [-1, 1], # hard coded isosurface_bbox
        )
        # get the sdf values    
        sdf_batch, deformation_batch = self.geometry.forward_field(
            points[None, ...].expand(batch_size, -1, -1),
            space_cache
        )

        # change the gradient of the sdf
        if self.sdf_grad_shrink != 0:
            sdf_batch = self.sdf_grad_shrink * sdf_batch + (1 - self.sdf_grad_shrink) * sdf_batch.detach()
        else: # save memory
            sdf_batch = sdf_batch.detach()
            
        if self.def_grad_shrink != 0:
            deformation_batch = self.sdf_grad_shrink * deformation_batch + (1 - self.sdf_grad_shrink) * deformation_batch.detach() \
                if deformation_batch is not None else None
        else: # save memory
            deformation_batch = deformation_batch.detach() \
                if deformation_batch is not None else None
        
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
                threestudio.info("All sdf values are positive or negative, no isosurface")
                self.empty_flag = self.cfg.allow_empty_flag # special operation, set to detach the gradient from wrong isosurface

                # attempt 1
                # # if no sdf with 0, manually add 5% to be 0
                # sdf_copy = sdf.clone()
                # with torch.no_grad():
                #     # select the 1% of the points that are closest to 0
                #     sdf_abs = torch.abs(sdf_copy)
                #     # get the threshold
                #     threshold = torch.topk(sdf_abs.flatten(), int(0.02 * sdf_abs.numel()), largest=False).values[-1]
                #     # find the points that are closest to 0
                #     idx = torch.where(sdf_abs < thres hold)
                # sdf[idx] = 0.0 * sdf[idx]

                # # attempt 2
                # # subtract the mean
                # # sdf_mean = torch.mean(sdf)
                # sdf_mean = torch.mean(sdf).detach()
                # sdf = sdf - sdf_mean

                # follow InstantMesh https://github.com/TencentARC/InstantMesh/blob/main/src/models/lrm_mesh.py
                update_sdf = torch.zeros_like(sdf)
                max_sdf = sdf.max()
                min_sdf = sdf.min()
                update_sdf[self.center_indices] += (-1 - max_sdf) # smaller than zero
                update_sdf[self.border_indices] += (1 - min_sdf) # larger than zero
                new_sdf = sdf + update_sdf
                update_mask = (new_sdf == 0).float()
                sdf = new_sdf * (1 - update_mask) + sdf * update_mask

            if index > 0 and self.cfg.isosurface_method == "diffmc":
                # according to https://github.com/SarahWeiii/diso/issues/2
                # if the batch size is larger than 1, then should use unique isosurface for each data
                if not hasattr(self, f"isosurface_helper_{index}"):
                    from threestudio.models.isosurface import  DiffMarchingCubeHelper
                    setattr(self, f"isosurface_helper_{index}", DiffMarchingCubeHelper(
                        self.cfg.isosurface_resolution,
                    ))
                mesh = getattr(self, f"isosurface_helper_{index}")(sdf, deformation)
            else:
                mesh = self.isosurface_helper(sdf, deformation)
            
            mesh.v_pos = scale_tensor(
                mesh.v_pos,
                self.isosurface_helper.points_range,
                [-1, 1], # hard coded isosurface_bbox
            )
            
            if self.cfg.isosurface_remove_outliers:
                mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
            mesh_list.append(mesh)
            
        return mesh_list


    def train(self, mode=True):
        if hasattr(self.geometry, "train"):
            self.geometry.train(mode)
        return super().train(mode=mode)

    def eval(self):
        if hasattr(self.geometry, "eval"):
            self.geometry.eval()
        return super().eval()
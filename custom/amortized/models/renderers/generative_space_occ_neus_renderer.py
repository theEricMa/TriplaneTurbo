from dataclasses import dataclass
from functools import partial
from tqdm import tqdm

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.neus_volume_renderer import NeuSVolumeRenderer
from threestudio.utils.ops import validate_empty_rays, chunk_batch
from .utils import chunk_batch as chunk_batch_custom # a different chunk_batch from threestudio.utils.ops
from threestudio.utils.typing import *
from threestudio.utils.ops import chunk_batch as chunk_batch_original

from threestudio.models.renderers.neus_volume_renderer import volsdf_density
from threestudio.utils.ops import scale_tensor as scale_tensor
from copy import deepcopy

from nerfacc import OccGridEstimator

class LearnedVariance(nn.Module):
    def __init__(self, init_val, requires_grad=True):
        super(LearnedVariance, self).__init__()
        self.register_parameter("_inv_std", nn.Parameter(torch.tensor(init_val), requires_grad=requires_grad))

    @property
    def inv_std(self):
        val = torch.exp(self._inv_std * 10.0)
        return val

    def forward(self, x):
        return torch.ones_like(x) * self.inv_std.clamp(1.0e-6, 1.0e6)


@threestudio.register("generative-space-occ-neus-renderer")
class GenerativeSpaceOccNeusRenderer(NeuSVolumeRenderer):
    @dataclass
    class Config(NeuSVolumeRenderer.Config):
        # the following are from NeuS #########
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 320000
        learned_variance_init: float = 0.3
        cos_anneal_end_steps: int = 0
        use_volsdf: bool = False

        near_plane: float = 0.0
        far_plane: float = 1e10

        trainable_variance: bool = True

        # for occgrid
        # grid_prune: bool = True
        # prune_alpha_threshold: bool = True
        grid_resolution: int = 64
        occ_thres: float = 1e-2

        # for chunking in training
        train_chunk_size: int = 0


    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.variance = LearnedVariance(self.cfg.learned_variance_init, requires_grad=self.cfg.trainable_variance)
        self.estimator = nerfacc.OccGridEstimator(
                            roi_aabb=self.geometry.bbox.view(-1), 
                            resolution=self.cfg.grid_resolution,
                            levels=1, # for now, we only use one level
                        )
        # if the resolution is too high, we need to chunk the training
        self.chunk_training = self.cfg.train_chunk_size > 0

        grid_res = self.cfg.grid_resolution
        # grid = self.estimator.grid_coords.reshape(grid_res, grid_res, grid_res, 3)
        
        v = torch.zeros([grid_res] * 3, dtype=torch.bool,)
        v[grid_res // 2:grid_res // 2 + 1, grid_res // 2:grid_res // 2 + 1, grid_res // 2:grid_res // 2 + 1] = True
        self.center_indices = torch.nonzero(v.reshape(-1)).to(self.device)

        v = torch.zeros([grid_res] * 3, dtype=torch.bool,)
        v[:2, :, :] = True; v[-2:, :, :] = True
        v[:, :2, :] = True; v[:, -2:, :] = True
        v[:, :, :2] = True; v[:, :, -2:] = True
        self.border_indices = torch.nonzero(v.reshape(-1)).to(self.device)

    def occ_eval_fn(
            self,
            sdf, # sdf is a tensor of any shape
        ):
        inv_std = self.variance(sdf)
        if self.cfg.use_volsdf:
            alpha = self.render_step_size * volsdf_density(sdf, inv_std)
        else:
            estimated_next_sdf = sdf - self.render_step_size * 0.5
            estimated_prev_sdf = sdf + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha
        
    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Float[Tensor, "B ..."]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        c2w: Optional[Float[Tensor, "B 4 4"]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size
        num_views_per_batch = batch_size // batch_size_space_cache

        if space_cache is None:
            space_cache = self.geometry.generate_space_cache(
                styles = noise,
                text_embed = text_embed,
            )
        
        points = scale_tensor(
                self.estimator.grid_coords / (self.estimator.resolution -1),
                [0, 1],
                [-1,1]
            )
        
        with torch.no_grad():
            sdf_batch, _ = self.geometry.forward_field(
                points[None, ...].expand(batch_size_space_cache, -1, -1),
                space_cache
            )

        out_list = []
        for idx in range(batch_size_space_cache):
            # prune the grid 
            estimator = deepcopy(self.estimator)
            sdf = sdf_batch[idx]

            # special case when all sdf values are positive or negative, thus no isosurface
            if torch.all(sdf > 0) or torch.all(sdf < 0):
                # threestudio.info("In volume renderer, all sdf values are positive or negative, no isosurface")

                # follow InstantMesh https://github.com/TencentARC/InstantMesh/blob/main/src/models/lrm_mesh.py
                update_sdf = torch.zeros_like(sdf)
                max_sdf = sdf.max()
                min_sdf = sdf.min()
                update_sdf[self.center_indices] += (-1 - max_sdf) # smaller than zero
                update_sdf[self.border_indices] += (1 - min_sdf) # larger than zero
                new_sdf = sdf + update_sdf
                update_mask = (new_sdf == 0).float()
                sdf = new_sdf * (1 - update_mask) + sdf * update_mask

            estimator.occs = self.occ_eval_fn(sdf).view(-1)
            estimator.binaries = (
                    estimator.occs > torch.clamp(
                        estimator.occs.mean(), 
                        max = self.cfg.occ_thres
                    )
                ).view(estimator.binaries.shape)
            # self.estimator.occs.fill_(True)
            # self.estimator.binaries.fill_(True)

            out = self._forward(
                rays_o=rays_o[
                        idx*num_views_per_batch:(idx+1)*num_views_per_batch
                    ],
                rays_d=rays_d[
                        idx*num_views_per_batch:(idx+1)*num_views_per_batch
                    ],
                light_positions=light_positions[
                        idx*num_views_per_batch:(idx+1)*num_views_per_batch
                    ],
                bg_color=bg_color[
                        idx*num_views_per_batch:(idx+1)*num_views_per_batch
                    ] if bg_color is not None else None,
                space_cache=space_cache[
                        idx:idx+1
                    ], # as the space_cache is tensor,
                text_embed=text_embed[
                        idx:idx+1
                    ] if text_embed is not None else None,
                estimator=estimator,
                camera_distances=camera_distances[
                        idx*num_views_per_batch:(idx+1)*num_views_per_batch
                    ] if camera_distances is not None else None,
                c2w=c2w[
                        idx*num_views_per_batch:(idx+1)*num_views_per_batch
                    ] if c2w is not None else None,
                **kwargs
            )
            out_list.append(out)

        # stack the outputs
        out_dict = {}
        for key in out_list[0].keys():
            if key not in ["mesh", "sdf_grad", "sdf"]: # hard coded for special case
                out_dict[key] = torch.concat([o[key] for o in out_list], dim=0)
            else:
                out_dict[key] = [o[key] for o in out_list]
        out_dict.update({"inv_std": self.variance.inv_std})
        return out_dict

            

    def _forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        estimator: OccGridEstimator,
        bg_color: Optional[Tensor] = None,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Float[Tensor, "B ..."]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        c2w: Optional[Float[Tensor, "B 4 4"]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        # reshape position and direction
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]
        
        # def alpha_fn(
        #     t_starts: Float[Tensor, "Nr Ns"],
        #     t_ends: Float[Tensor, "Nr Ns"],
        #     ray_indices,
        # ):
        #     if ray_indices.sum() == 0:
        #         # no valid rays, just return an empty tensor as t_starts
        #         return t_starts

        #     t_starts, t_ends = t_starts[..., None], t_ends[..., None]
        #     t_origins = rays_o_flatten[ray_indices]
        #     t_positions = (t_starts + t_ends) / 2.0
        #     t_dirs = rays_d_flatten[ray_indices]
        #     positions = t_origins + t_dirs * t_positions
        #     if self.training:
        #         sdf, _ = self.geometry.forward_field(
        #             positions[None, ...],
        #             space_cache,
        #         )
        #         sdf: Float[Tensor, "N"] = sdf[0, ..., 0] # don't need the first index and the last index
        #     else:
        #         sdf, _ = chunk_batch_custom(
        #             partial(
        #                 self.geometry.forward_field,
        #                 space_cache=space_cache,
        #             ),
        #             self.cfg.eval_chunk_size,
        #             positions[None, ...],
        #         )#[..., 0] # TODO: check if we need to add the last index

        #     alpha = self.occ_eval_fn(sdf)
        #     return alpha
             
        ray_indices, t_starts_, t_ends_ = estimator.sampling(
            rays_o_flatten,
            rays_d_flatten,
            alpha_fn=None, #alpha_fn if self.cfg.prune_alpha_threshold else None,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_step_size=self.render_step_size,
            alpha_thre=0, #self.cfg.occ_thres if self.cfg.prune_alpha_threshold else 0.0,
            stratified=self.randomized,
            cone_angle=0.0,
        )
        
        # the following are from NeuS #########
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training and not self.chunk_training:
            geo_out = self.geometry(
                positions[None, ...],
                space_cache=space_cache,
                output_normal=True,
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )

            # background
            if hasattr(self.background, "enabling_hypernet") and self.background.enabling_hypernet:
                comp_rgb_bg = self.background(
                    dirs=rays_d, 
                    text_embed=text_embed if "text_embed_bg" not in kwargs else kwargs["text_embed_bg"]
                )
            else:
                comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch_custom(
                partial(
                    self.geometry,
                    space_cache=space_cache,
                ),
                self.cfg.train_chunk_size if self.training else self.cfg.eval_chunk_size,
                positions[None, ...],
                output_normal=True,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size, # since we donnot change the module here, we can use eval_chunk_size
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )

            # background
            if hasattr(self.background, "enabling_hypernet") and self.background.enabling_hypernet:
                comp_rgb_bg = chunk_batch(
                    self.background, 
                    self.cfg.eval_chunk_size, # since we donnot change the module here, we can use eval_chunk_size
                    dirs=rays_d,
                    text_embed=text_embed if "text_embed_bg" not in kwargs else kwargs["text_embed_bg"]
                )
            else:
                comp_rgb_bg = chunk_batch(
                    self.background, 
                    self.cfg.eval_chunk_size, # since we donnot change the module here, we can use eval_chunk_size
                    dirs=rays_d
                )

        # grad or normal?
        alpha: Float[Tensor, "Nr 1"] = self.get_alpha(
            geo_out["sdf"], geo_out["normal"], t_dirs, t_intervals
        )

        weights: Float[Tensor, "Nr 1"]
        weights_, _ = nerfacc.render_weight_from_alpha(
            alpha[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if bg_color is None:
            bg_color = comp_rgb_bg

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)


        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
        }

        # the following are from richdreamer #########

        far= camera_distances.reshape(-1, 1, 1, 1) + torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
        near = camera_distances.reshape(-1, 1, 1, 1) - torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
        disparity_tmp = out["depth"] * out["opacity"] + (1.0 - out["opacity"]) * far
        disparity_norm = (far - disparity_tmp) / (far - near)
        disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
        out.update(
            {
                "disparity": disparity_norm.view(batch_size, height, width, 1),
            }
        )

        #############################################

        # compute normal is also used in training
        if "normal" in geo_out:
            comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                weights[..., 0],
                values=geo_out["normal"],
                ray_indices=ray_indices,
                n_rays=n_rays,
            )

            comp_normal = F.normalize(comp_normal, dim=-1)
            comp_normal_mask = torch.lerp(
                torch.zeros_like(comp_normal), (comp_normal + 1.0) / 2.0, opacity
            )

            # for compatibility with RichDreamer #############
            bg_normal = 0.5 * torch.ones_like(comp_normal)
            bg_normal[:, 2] = 1.0 # for a blue background
            bg_normal_white = torch.ones_like(comp_normal)

            # comp_normal_vis = (comp_normal + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
            # convert_normal_to_cam_space
            w2c: Float[Tensor, "B 4 4"] = torch.inverse(c2w)
            rot: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]
            comp_normal_cam = comp_normal.view(batch_size, -1, 3) @ rot.permute(0, 2, 1)
            flip_x = torch.eye(3, device=comp_normal_cam.device) #  pixel space flip axis so we need built negative y-axis normal
            flip_x[0, 0] = -1
            comp_normal_cam = comp_normal_cam @ flip_x[None, :, :]
            comp_normal_cam = comp_normal_cam.view(-1, 3) # reshape back to (Nr, 3)
            comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
            comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

            out.update(
                {
                    "comp_normal": comp_normal_mask.view(batch_size, height, width, 3),
                    # "comp_normal_vis": comp_normal_vis.view(batch_size, height, width, 3),
                    "comp_normal_cam_vis": comp_normal_cam_vis.view(batch_size, height, width, 3),
                    "comp_normal_cam_vis_white": comp_normal_cam_vis_white.view(batch_size, height, width, 3),
                }
            )

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )

        return out
    
    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        pass

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if hasattr(self.geometry, "train"):
            self.geometry.train(mode)
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if hasattr(self.geometry, "eval"):
            self.geometry.eval()
        return super().eval()
    
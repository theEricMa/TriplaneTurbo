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
from threestudio.models.renderers.neus_volume_renderer import NeuSVolumeRenderer
from threestudio.utils.ops import validate_empty_rays, chunk_batch
from .utils import chunk_batch as chunk_batch_custom # a different chunk_batch from threestudio.utils.ops
from threestudio.utils.typing import *
from threestudio.utils.ops import chunk_batch as chunk_batch_original

from threestudio.models.renderers.neus_volume_renderer import volsdf_density
from threestudio.utils.misc import C

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


@threestudio.register("generative-space-volsdf-volume-renderer")
class GenerativeSpaceVolSDFVolumeRenderer(NeuSVolumeRenderer):
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
        # in ['occgrid', 'importance']
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for importance
        num_samples_per_ray_importance: int = 64

        # for chunking in training
        train_chunk_size: int = 0

        # for balancing the low-res and high-res gradients
        rgb_grad_shrink: float = 1.0

        # for rendering the normal
        normal_direction: str = "camera"  # "front" or "camera" or "world"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.variance = LearnedVariance(self.cfg.learned_variance_init, requires_grad=self.cfg.trainable_variance)
        if self.cfg.estimator == "occgrid":
            threestudio.error("Occgrid estimator not supported for generative-space-volsdf-volume-renderer")
            raise NotImplementedError
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        else:
            raise NotImplementedError(
                f"Estimator {self.cfg.estimator} not implemented"
            )
        # if the resolution is too high, we need to chunk the training
        self.chunk_training = self.cfg.train_chunk_size > 0

        assert self.cfg.normal_direction in ["front", "camera", "world"], "normal_direction must be in ['front', 'camera', 'world']"


    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Float[Tensor, "B ..."]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size

        if space_cache is None:
            space_cache = self.geometry.generate_space_cache(
                styles = noise,
                text_embed = text_embed,
            )

        # self.training = True #  for debugging
        if self.training:
            if batch_size_space_cache != batch_size:
                # copy the space cache in training
                assert batch_size > batch_size_space_cache
                if torch.is_tensor(space_cache): # for triplane-transformer and 3DConv-net
                    space_cache = space_cache.repeat_interleave(
                        batch_size // batch_size_space_cache,
                        dim = 0
                    )
                elif isinstance(space_cache, Dict): # for hyper-net
                    new_space_cache = {}
                    for key, value in space_cache.items():
                        if torch.is_tensor(value):
                            new_space_cache[key] = value.repeat_interleave(
                                batch_size // batch_size_space_cache,
                                dim = 0
                            )
                        elif isinstance(value, list):
                            new_space_cache[key] = [
                                v.repeat_interleave(
                                    batch_size // batch_size_space_cache,
                                    dim = 0
                                ) for v in value
                            ]
                    space_cache = new_space_cache
                else:
                    raise NotImplementedError

            return self._forward(
                rays_o=rays_o,
                rays_d=rays_d,
                light_positions=light_positions,
                bg_color=bg_color,
                noise=noise,
                space_cache=space_cache,
                text_embed=text_embed,
                **kwargs
            )

        else: # inference
            if batch_size_space_cache != batch_size:
                assert batch_size_space_cache == 1, "batch_size of space_cache must be 1 or equal to batch_size of rays_o"
                
                chunk_size = 1 # hard coded, for fully utilizing the GPU memory for inference

                # now loop over batch_size
                func = partial(
                    self._forward,
                    space_cache=space_cache,   
                    noise=noise,
                    text_embed=text_embed,
                    text_embed_bg = kwargs.pop("text_embed_bg", None)
                )
                out = chunk_batch_original(
                    func,
                    chunk_size=chunk_size,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    light_positions=light_positions,
                    bg_color=bg_color,
                    **kwargs
                )
                
                if 'inv_std' in out: # means during training
                    out['inv_std'] = out['inv_std'][0]

                return out  

            else:
                return self._forward(
                    rays_o=rays_o,
                    rays_d=rays_d,
                    light_positions=light_positions,
                    bg_color=bg_color,
                    noise=noise,
                    space_cache=space_cache,
                    text_embed=text_embed,
                    **kwargs
                )
            
            

    def _forward(
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

        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size
        num_views_per_batch = batch_size // batch_size_space_cache

        # important for generative space
        if space_cache is None:
            assert noise is not None, "Either space_cache or noise must be provided"
            space_cache = self.geometry.generate_space_cache(
                styles = noise,
                text_embed = text_embed,
            )
        
        # check the shape of space_cache
        if torch.is_tensor(space_cache):
            assert space_cache.shape[0] == batch_size, "space_cache must have the same batch size as rays_o"

        if self.cfg.estimator == "occgrid":
            raise NotImplementedError
        elif self.cfg.estimator == "importance":
            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
                space_cache: Float[Tensor, "B ..."],
            ):
                if torch.is_tensor(space_cache):
                    B = space_cache.shape[0] # batch size, used for chunk_batch
                elif isinstance(space_cache, Dict):
                    for key, value in space_cache.items():
                        if torch.is_tensor(value):
                            B = value.shape[0]
                        if isinstance(value, list):
                            B = value[0].shape[0]
                        break
                else:
                    raise ValueError("space_cache must be a tensor or a dict")
                
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                with torch.no_grad():
                    if self.training and not self.chunk_training:
                        geo_out = self.geometry(
                            positions.reshape(B, -1, 3),
                            space_cache=space_cache,
                            output_normal=False,
                        )

                    else:
                        geo_out = chunk_batch_custom(
                            partial(
                                proposal_network,
                                space_cache=space_cache,
                            ),
                            self.cfg.train_chunk_size if self.training else self.cfg.eval_chunk_size,
                            positions.reshape(B, -1, 3),
                            output_normal=False,
                        )
                    inv_std = self.variance(geo_out["sdf"])

                    if self.cfg.use_volsdf:
                        density:  Float[Tensor, "B Ns"] = volsdf_density(geo_out["sdf"], inv_std).reshape(positions.shape[:2])
                    else:
                        sdf = geo_out["sdf"]
                        estimated_next_sdf = sdf - self.render_step_size * 0.5
                        estimated_prev_sdf = sdf + self.render_step_size * 0.5
                        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                        p = prev_cdf - next_cdf
                        c = prev_cdf
                        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                        density = (alpha / self.render_step_size).reshape(positions.shape[:2])
                        
                return density

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[
                        partial(
                            prop_sigma_fn, 
                            proposal_network=self.geometry, 
                            space_cache=space_cache,
                        )
                    ],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError
        
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
                positions.reshape(batch_size, -1, 3),
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
                positions.reshape(batch_size, -1, 3),
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

        if self.rgb_grad_shrink != 1.0:
            # shrink the gradient of rgb_fg_all
            # this is to balance the low-res and high-res gradients
            rgb_fg_all = self.rgb_grad_shrink * rgb_fg_all + (1.0 - self.rgb_grad_shrink) * rgb_fg_all.detach()

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
            out.update(
                {
                    "comp_normal": comp_normal.view(batch_size, height, width, 3),
                }
            )

            if self.cfg.normal_direction == "camera":
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
                        "comp_normal_cam_vis": comp_normal_cam_vis.view(batch_size, height, width, 3),
                        "comp_normal_cam_vis_white": comp_normal_cam_vis_white.view(batch_size, height, width, 3),
                    }
                )
            elif self.cfg.normal_direction == "front":

                # for compatibility with Wonder3D and Era3D #############
                bg_normal_white = torch.ones_like(comp_normal)

                # convert_normal_to_cam_space of the front view
                c2w_front = c2w[0::num_views_per_batch].repeat_interleave(num_views_per_batch, dim=0)
                w2c_front: Float[Tensor, "B 4 4"] = torch.inverse(c2w_front)                
                rot: Float[Tensor, "B 3 3"] = w2c_front[:, :3, :3]
                comp_normal_front = comp_normal.view(batch_size, -1, 3) @ rot.permute(0, 2, 1)

                # the following is not necessary for Wonder3D and Era3D
                # flip_x = torch.eye(3, device=comp_normal_front.device) #  pixel space flip axis so we need built negative y-axis normal
                # flip_x[0, 0] = -1
                # comp_normal_front = comp_normal_front @ flip_x[None, :, :]
                
                comp_normal_front = comp_normal_front.view(-1, 3) # reshape back to (Nr, 3)
                comp_normal_front_vis_white = (comp_normal_front + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                out.update(
                    {
                        "comp_normal_cam_vis_white": comp_normal_front_vis_white.view(batch_size, height, width, 3),
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

            out.update({"inv_std": self.variance.inv_std})
        return out
    
    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.rgb_grad_shrink = C(
            self.cfg.rgb_grad_shrink, epoch, global_step
        )

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
    
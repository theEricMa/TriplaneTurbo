import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.utils.ops import perpendicular_component
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from extern.mvdream.model_zoo import build_model
from extern.mvdream.camera_utils import normalize_camera

@threestudio.register("mvdream-asynchronous-score-distillation-guidance")
class MVDreamTimestepShiftedScoreDistillationGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        guidance_scale: float = 7.5

        n_view: int = 4
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = "sds"

        plus_ratio: float = 0.1
        plus_random: bool = False

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False
    
    cfg: Config

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)


    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        if hasattr(self.model, "cond_stage_model"):
            # delete unused models
            del self.model.cond_stage_model # text encoder
            cleanup()

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Multiview Diffusion!")

        # DDPM
        self.alphas = self.model.alphas_cumprod


    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B
    
    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 32 32"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, size=(32, 32), mode="bilinear", align_corners=False
            ) # TODO: check if this is correct, or * 2.0 - 1.0
        else:
            # rgb_BCHW_256 = F.interpolate(
            #     rgb_BCHW, size=(512, 512), mode="bilinear", align_corners=False
            # )
            # # encode image into latents
            # latents = self.encode_images(rgb_BCHW_256)
            # # encode again
            # latents = F.interpolate(
            #     latents, size=(32, 32), mode="bilinear", align_corners=False
            # )

            # memory efficient one
            rgb_BCHW_256 = F.interpolate(
                rgb_BCHW, size=(256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents
            latents = self.encode_images(rgb_BCHW_256)

        return latents

    def get_t_plus(
        self, 
        t: Float[Tensor, "B"]
    ):

        t_plus = self.cfg.plus_ratio * (t - self.min_step)
        if self.cfg.plus_random:
            t_plus = (t_plus * torch.rand(*t.shape,device = self.device)).to(torch.long)
        else:
            t_plus = t_plus.to(torch.long)
        t_plus = t + t_plus
        t_plus = torch.clamp(
            t_plus,
            1, # T_min
            self.num_train_timesteps - 1, # T_max
        )
        return t_plus


    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        input_is_latent=False,
        **kwargs,
    ):
        camera = c2w
        batch_size = rgb.shape[0]


        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        # noise is shared
        # noise = torch.randn_like(latents)[:batch_size // 4] # trick: this helps for using triplane-transformer with mv-dream
        # noise = noise.repeat_interleave(4, dim=0)
        noise = torch.randn_like(latents)
        


        # prepare text embeddings
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        text_batch_size = text_embeddings.shape[0] // 2
        # double the prompt
        text_embeddings_vd     = text_embeddings[0 * text_batch_size: 1 * text_batch_size].repeat(batch_size // text_batch_size, 1, 1)
        text_embeddings_uncond = text_embeddings[1 * text_batch_size: 2 * text_batch_size].repeat(batch_size // text_batch_size, 1, 1)
        text_embeddings = torch.cat(
            [
                text_embeddings_vd, 
                text_embeddings_uncond, 
                text_embeddings_vd
            ], 
            dim=0
        )

        assert self.min_step is not None and self.max_step is not None
        with torch.no_grad():
            # random timestamp for the first diffusion model
            _t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=self.device,
            )
            t = _t.repeat(batch_size)
            latents_noisy = self.model.q_sample(latents, t, noise=noise)

            # bigger timestamp 
            _t_plus = self.get_t_plus(_t)
            t_plus = _t_plus.repeat(batch_size)

            latents_noisy_second = self.model.q_sample(latents, t_plus, noise=noise)

            # prepare input for UNet
            latents_model_input = torch.cat(
                [
                    latents_noisy,
                    latents_noisy,
                    latents_noisy_second,
                ],
                dim=0,
            )
            t_expand = torch.cat(
                [
                    t,
                    t,
                    t_plus,
                ],
                dim=0,
            )
                    
            # the following is different from stable-diffusion ###
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(3, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}

            noise_pred = self.model.apply_model(
                latents_model_input, 
                t_expand,
                context,
            )

        # perform guidance
        noise_pred_text, noise_pred_uncond, noise_pred_text_second = noise_pred.chunk(
            3
        )
        noise_pred_first = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        noise_pred_second = noise_pred_text_second
        
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
            
        grad = (noise_pred_first - noise_pred_second) * w
        grad = torch.nan_to_num(grad)
        # clip grad for stability?
        if self.grad_clip_val is not None:
            grad = torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_tssd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return {
            "loss_asd": loss_tssd,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        } 

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        


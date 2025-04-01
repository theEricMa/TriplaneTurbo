import random
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

@threestudio.register("stable-diffusion-asynchronous-score-distillation-guidance")
class SDTimestepShiftedScoreDistillationGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        # FIXME: xformers error
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = True
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = "sds"

        plus_ratio: float = 0.1
        plus_random: bool = False

        view_dependent_prompting: bool = True

        guidance_perp_neg: float = 0.
        
    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            pipe.unet.to(memory_format=torch.channels_last)

        del pipe.text_encoder
        cleanup()

        # Create model
        self.vae = pipe.vae.eval().to(self.device)
        self.unet = pipe.unet.eval().to(self.device)

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        # perp-neg
        self.use_perp_neg = self.cfg.guidance_perp_neg != 0

        threestudio.info(f"Loaded Stable Diffusion!")
        del pipe

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

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


    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents
    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        # noise is shared
        noise = torch.randn_like(latents)

        assert self.min_step is not None and self.max_step is not None
        with torch.no_grad():

            # random timestamp for the first diffusion model
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # bigger timestamp 
            t_plus = self.get_t_plus(t)
            latents_noisy_second = self.scheduler.add_noise(latents, noise, t_plus)

            # pred noise
            noise_pred, noise_pred_second = self.get_eps(
                latents_noisy, 
                latents_noisy_second,
                t,
                t_plus,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                self.unet,
            )

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
            
        grad = w * (noise_pred - noise_pred_second)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_asd": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return guidance_out
    
    def get_t_plus(
        self, 
        t: Float[Tensor, "B"]
    ):
        if self.cfg.plus_random:
            if self.cfg.plus_ratio >= 0.:
                # random timestamp for qt
                t_plus = self.cfg.plus_ratio * torch.rand(*t.shape,device = self.device) * (self.max_step - t)
                t_plus = t + t_plus.to(torch.long)
            else:
                # sample according to lower bound
                t_plus = self.cfg.plus_ratio * torch.rand(*t.shape,device = self.device) * (t - self.min_step)
                t_plus = t - t_plus.to(torch.long)
        else:
            if self.cfg.plus_ratio >= 0.:
                # bigger timestamp for qt
                t_plus = t + (self.cfg.plus_ratio * (self.max_step - t)).to(torch.long)
            else:
                # smaller timestamp for qt
                t_plus = t - (self.cfg.plus_ratio * (t - self.min_step)).to(torch.long)
        t_plus = torch.clamp(
            t_plus, 
            self.min_step, 
            self.max_step
        ) # make t_plus in range [min_step, max_step]
        return t_plus
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: nn.Module,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def get_eps(
        self,
        latents_noisy: Float[Tensor, "B 4 64 64"],
        latents_noisy_second: Float[Tensor, "B 4 64 64"],
        t: Float[Tensor, "B"],
        t_plus: Float[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        unet: nn.Module,
        use_perp_neg: bool = False,
    ) -> Float[Tensor, "B 4 64 64"]:

        batch_size = latents_noisy.shape[0]

        # assign values to optional arguments
        guidenace_scale=self.cfg.guidance_scale
        guidenace_scale_perp_neg=self.cfg.guidance_perp_neg
        use_perp_neg=self.use_perp_neg
        
        # view dependent prompting
        if use_perp_neg:
            assert prompt_utils.use_perp_neg
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting,
            )
            neg_guidance_weights = neg_guidance_weights * -1 * guidenace_scale_perp_neg # multiply by a negative value to control its magnitude
            text_embeddings_vd     = text_embeddings[0 * batch_size: 1 * batch_size]
            text_embeddings_uncond = text_embeddings[1 * batch_size: 2 * batch_size]
            text_embeddings_vd_neg = text_embeddings[2 * batch_size: 4 * batch_size]
        else:
            text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            neg_guidance_weights = None
            text_embeddings_vd     = text_embeddings[0 * batch_size: 1 * batch_size]
            text_embeddings_uncond = text_embeddings[1 * batch_size: 2 * batch_size]
            text_embeddings_vd_neg = None

        # collect text embeddings
        text_embeddings = [text_embeddings_vd] # for the first diffusion model
        text_embeddings.append(text_embeddings_uncond) 
        if use_perp_neg:
            text_embeddings.append(text_embeddings_vd_neg)
        text_embeddings.append(text_embeddings_vd) # for the second diffusion model
        text_embeddings = torch.cat(text_embeddings, dim=0).to(self.device)

        # collect other inputs
        batch_size_t = text_embeddings.shape[0]
        num_repeats = batch_size_t // batch_size - 1 # -1 for the second diffusion model
        input_t = torch.cat(
            [t] * num_repeats + [t_plus],
            dim=0
        ).to(self.device)
        input_latents_noisy = torch.cat(
            [latents_noisy] * num_repeats + [latents_noisy_second],
            dim=0
        ).to(self.device)

        # compute eps
        with torch.no_grad():
            noise_pred = self.forward_unet(
                unet,
                input_latents_noisy,
                input_t,
                encoder_hidden_states=text_embeddings,
            )

        # split noise_pred
        noise_pred_text   = noise_pred[0 * batch_size: 1 * batch_size]
        noise_pred_uncond = noise_pred[1 * batch_size: 2 * batch_size]
        noise_pred_vd_neg = noise_pred[2 * batch_size: 4 * batch_size]if use_perp_neg else None
        noise_pred_second = noise_pred[4 * batch_size: 5 * batch_size] if use_perp_neg else noise_pred[2 * batch_size: 3 * batch_size]

        # aggregate noise_pred
        eps_pos = noise_pred_text - noise_pred_uncond
        if neg_guidance_weights is not None: # same as use_perp_neg
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                eps_vd_neg = noise_pred_vd_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, *[1] * (eps_vd_neg.ndim - 1)
                ) * perpendicular_component(eps_vd_neg, eps_pos) # eps_vd_neg # v2

            # noise_pred_p = (eps_pos) * guidenace_scale + noise_pred_uncond + accum_grad
            noise_pred_p = (eps_pos + accum_grad) * guidenace_scale + noise_pred_uncond 

        else: # if not use_perp_neg
            noise_pred_p = eps_pos  * guidenace_scale + noise_pred_uncond

        return noise_pred_p.to(self.device), noise_pred_second.to(self.device)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

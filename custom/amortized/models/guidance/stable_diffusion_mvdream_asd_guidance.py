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

from extern.mvdream.model_zoo import build_model
from extern.mvdream.camera_utils import normalize_camera

from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from torch.nn.modules.container import ModuleList
from extern.mvdream.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from torch.autograd import Variable, grad as torch_grad
from threestudio.utils.ops import SpecifyGradient

@threestudio.register("stable-diffusion-mvdream-asynchronous-score-distillation-guidance")
class SDMVAsynchronousScoreDistillationGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # specify the model name for the mvdream model and the stable diffusion model
        mv_model_name_or_path: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        mv_ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        sd_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"

        # the following is specific to mvdream
        mv_n_view: int = 4
        mv_camera_condition_type: str = "rotation"
        mv_view_dependent_prompting: bool = False
        mv_image_size: int = 256
        mv_guidance_scale: float = 7.5
        mv_weight: float = 0.25 # 1 / 4
        mv_2nd_render_only: bool = False

        # the following is specific to stable diffusion
        sd_view_dependent_prompting: bool = True
        sd_all_views: bool = False
        sd_image_size: int = 512
        sd_guidance_perp_neg: float = 0.0
        sd_guidance_scale: float = 7.5
        sd_weight: float = 1.
        sd_1st_render_only: bool = False
        sd_weighting_strategy: str = "uniform" # ASD is suitable for uniform weighting, but can be extended to other strategies

        # the following is shared between mvdream and stable diffusion
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        # the following is specific to ASD
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        plus_schedule: str = "linear"  # linear or sqrt_<bias>

        # the following is specific to the combination of MVDream and Stable Diffusion with ASD
        mv_plus_random: bool = True
        mv_plus_ratio: float = 0.1
        sd_plus_random: bool = True
        sd_plus_ratio: float = 0.1
        mv_weighting_strategy: str = "uniform" # ASD is suitable for uniform weighting, but can be extended to other strategies

        # the following is specific to the compatibility with dual rendering
        dual_render_sync_t: bool = False
        dual_render_sync_view_sd: bool = True

        # strategy to save memory
        gradient_checkpoint: bool = False
        auto_grad: bool = False

    cfg: Config

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def adapt_timestep_range(self, timestep_range: Optional[Tuple[float, float]]):
        # determine the timestamp
        if timestep_range is None:
            min_t = self.min_step
            max_t = self.max_step
        else:
            assert len(timestep_range) == 2
            min_t_ratio, max_t_ratio = timestep_range
            max_t = int(max_t_ratio * (self.max_step - self.min_step) + self.min_step)
            max_t = max(min(max_t, self.max_step), self.min_step) # clip the value
            min_t = int(min_t_ratio * (self.max_step - self.min_step) + self.min_step)
            min_t = max(min(min_t, self.max_step), self.min_step) # clip the value
        return min_t, max_t


    def configure(self) -> None:

        ################################################################################################
        if self.cfg.sd_weight > 0:
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
                self.cfg.sd_model_name_or_path,
                **pipe_kwargs,
            ).to(self.device)
            del pipe.text_encoder
            cleanup()

            # Create model
            self.sd_vae = pipe.vae.eval().to(self.device)
            self.sd_unet = pipe.unet.eval().to(self.device)

            for p in self.sd_vae.parameters():
                p.requires_grad_(False)
            for p in self.sd_unet.parameters():
                p.requires_grad_(False)

            self.sd_scheduler = DDPMScheduler.from_pretrained(
                self.cfg.sd_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

            self.sd_use_perp_neg = self.cfg.sd_guidance_perp_neg != 0

            if self.cfg.gradient_checkpoint:
                self.sd_vae.enable_gradient_checkpointing()
        else:
            threestudio.info("Stable Diffusion is disabled.")

        ################################################################################################
        if self.cfg.mv_weight > 0:
            threestudio.info(f"Loading Multiview Diffusion ...")

            self.mv_model = build_model(
                self.cfg.mv_model_name_or_path,
                ckpt_path=self.cfg.mv_ckpt_path
            ).to(self.device)
            for p in self.mv_model.parameters():
                p.requires_grad_(False)

            if hasattr(self.mv_model, "cond_stage_model"):
                # delete unused models
                del self.mv_model.cond_stage_model # text encoder
                cleanup()


        else:
            threestudio.info("Multiview Diffusion is disabled.")

        ################################################################################################
        # the folowing is shared between mvdream and stable diffusion
        self.alphas = self.mv_model.alphas_cumprod if self.cfg.mv_weight > 0 else self.sd_scheduler.alphas_cumprod # should be the same as self.scheduler.alphas_cumprod
        self.grad_clip_val: Optional[float] = None
        self.num_train_timesteps = 1000
        self.set_min_max_steps()  # set to default value


    def get_t_plus(
        self, 
        t: Float[Tensor, "B"],
        module: str # "sd" or "mv"
    ):
        
        # determine the attributes that differ between SD and MV
        if module == "sd":
            plus_random = self.cfg.sd_plus_random
            plus_ratio = self.cfg.sd_plus_ratio
        elif module == "mv":
            plus_random = self.cfg.mv_plus_random
            plus_ratio = self.cfg.mv_plus_ratio
        else:
            raise ValueError(f"Invalid module: {module}")

        # determine the timestamp for the second diffusion model
        if self.cfg.plus_schedule == "linear":
            t_plus = plus_ratio * (t - self.min_step)
        
        elif self.cfg.plus_schedule.startswith("sqrt"):
            bias = 0
            if self.cfg.plus_schedule.startswith("sqrt_"): # if bias is specified
                try:
                    bias = float(self.cfg.plus_schedule.split("_")[1])
                except:
                    raise ValueError(f"Invalid sqrt bias: {self.cfg.plus_schedule}")
                
            # t_plus = plus_ratio * torch.sqrt(t - self.min_step + bias)
            t_plus = plus_ratio * torch.sqrt(t + bias)
        else:
            raise ValueError(f"Invalid plus_schedule: {self.cfg.plus_schedule}")

        # clamp t_plus to the range [0, T_max - t], added in the revision
        t_plus = torch.clamp(
            t_plus,
            torch.zeros_like(t), 
            self.num_train_timesteps - t - 1,
        )

        # add the offset
        if plus_random:
            t_plus = (t_plus * torch.rand(*t.shape,device = self.device)).to(torch.long)
        else:
            t_plus = t_plus.to(torch.long)
        t_plus = t + t_plus

        # double check the range in [1, 999]
        t_plus = torch.clamp(
            t_plus,
            1, # T_min = 1
            max = self.num_train_timesteps - 1, # T_max = 999
        )
        return t_plus

################################################################################################
# the following is specific to MVDream
    def _mv_get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.mv_camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.mv_camera_condition_type}"
            )
        return camera

    def _mv_encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0

        if self.cfg.gradient_checkpoint:
            h = self.mv_model.first_stage_model.encoder(imgs, gradient_checkpoint = True)
            moments = self.mv_model.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
        
            latents = self.mv_model.get_first_stage_encoding(
                posterior
            )
        else:
            latents = self.mv_model.get_first_stage_encoding(
                self.mv_model.encode_first_stage(imgs)
            )
        return latents  # [B

    def mv_get_latents(
        self, 
        rgb_BCHW: Float[Tensor, "B C H W"], 
        rgb_BCHW_2nd: Optional[Float[Tensor, "B C H W"]] = None,
        rgb_as_latents=False
    ) -> Float[Tensor, "B 4 32 32"]:
        if rgb_as_latents:
            size = self.cfg.mv_image_size // 8
            latents = F.interpolate(
                rgb_BCHW, size=(size, size), mode="bilinear", align_corners=False
            )
            # resize the second latent if it exists
            if rgb_BCHW_2nd is not None:
                latents_2nd = F.interpolate(
                    rgb_BCHW_2nd, size=(size, size), mode="bilinear", align_corners=False
                )
                # concatenate the two latents
                latents = torch.cat([latents, latents_2nd], dim=0)
        else:
            size = self.cfg.mv_image_size
            rgb_BCHW_resize = F.interpolate(
                rgb_BCHW, size=(size, size), mode="bilinear", align_corners=False
            )
            # resize the second image if it exists
            if rgb_BCHW_2nd is not None:
                rgb_BCHW_2nd_resize = F.interpolate(
                    rgb_BCHW_2nd, size=(size, size), mode="bilinear", align_corners=False
                )
                # concatenate the two images
                rgb_BCHW_resize = torch.cat([rgb_BCHW_resize, rgb_BCHW_2nd_resize], dim=0)
            # encode image into latents
            latents = self._mv_encode_images(rgb_BCHW_resize)
        return latents


    def _mv_noise_pred(
        self,
        mv_latents: Float[Tensor, "B 4 32 32"],
        mv_noise: Float[Tensor, "B 4 32 32"],
        t: Float[Tensor, "B"],
        t_plus: Float[Tensor, "B"],
        text_embeddings_vd: Float[Tensor, "B 77 1024"],
        text_embeddings_uncond: Float[Tensor, "B 77 1024"],
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
        is_dual: bool = False,
    ):
        
        # determine attributes ################################################################################################
        use_t_plus = self.cfg.mv_plus_ratio > 0

        # prepare text embeddings ################################################################################################
        text_embeddings = [
            text_embeddings_vd if not is_dual else text_embeddings_vd.repeat(2, 1, 1), # for the 1st diffusion model's conditional guidance
            text_embeddings_uncond if not is_dual else text_embeddings_uncond.repeat(2, 1, 1), # for the 1st diffusion model's unconditional guidance
        ]
        if use_t_plus:
            text_embeddings += [
                text_embeddings_vd if not is_dual else text_embeddings_vd.repeat(2, 1, 1), # for the 2nd diffusion model's conditional guidance
            ]
        text_embeddings = torch.cat(
            text_embeddings,
            dim=0
        )

        # prepare noisy input ################################################################################################
        # random timestamp for the first diffusion model
        latents_noisy = self.mv_model.q_sample(mv_latents, t, noise=mv_noise)

        latents_model_input = [
            latents_noisy,
        ] * 2 # 2 for the conditional and unconditional guidance for the 1st diffusion model

        if use_t_plus:
            # random timestamp for the second diffusion model
            latents_noisy_second = self.mv_model.q_sample(mv_latents, t_plus, noise=mv_noise)

            latents_model_input += [
                latents_noisy_second,
            ]

        latents_model_input = torch.cat(
            latents_model_input,
            dim=0,
        )

        # prepare timestep input ################################################################################################
        t_expand = [
            t,
        ] * 2 # 2 for the conditional and unconditional guidance for the 1st diffusion model

        if use_t_plus:
            t_expand += [
                t_plus,
            ]

        t_expand = torch.cat(
            t_expand,
            dim=0,
        )

        # prepare context input ################################################################################################
        assert camera is not None
        camera = self._mv_get_camera_cond(camera, fovy=fovy)
        num_repeat = 3 if use_t_plus else 2
        if is_dual:
            num_repeat *= 2
        camera = camera.repeat(
            num_repeat, 
            1
        ).to(text_embeddings)
        context = {
            "context": text_embeddings,
            "camera": camera,
            "num_frames": self.cfg.mv_n_view,
        }

        # u-net forward pass ################################################################################################
        noise_pred = self.mv_model.apply_model(
            latents_model_input, 
            t_expand,
            context,
        )

        # split the noise_pred ################################################################################################
        if use_t_plus:
            noise_pred_text, noise_pred_uncond, noise_pred_text_second = noise_pred.chunk(
                3
            )
        else:
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(
                2
            )
            noise_pred_text_second = noise_pred_text

        return noise_pred_text, noise_pred_uncond, noise_pred_text_second

    def _mv_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        timestep_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):

        camera = c2w

        # determine if dual rendering is enabled
        is_dual = True if rgb_2nd is not None else False

        view_batch_size = rgb.shape[0]
        img_batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        # special case for dual rendering
        if is_dual:
            img_batch_size *= 2
            rgb_2nd_BCHW = rgb_2nd.permute(0, 3, 1, 2)

        ################################################################################################
        # the following is specific to MVDream
        mv_latents = self.mv_get_latents(
            rgb_BCHW,
            rgb_BCHW_2nd=rgb_2nd_BCHW if is_dual else None,
            rgb_as_latents=rgb_as_latents,
        )

        # prepare noisy input
        mv_noise = torch.randn_like(mv_latents)

        # prepare text embeddings
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation=elevation, 
            azimuth=azimuth, 
            camera_distances=camera_distances, 
            view_dependent_prompting=self.cfg.mv_view_dependent_prompting,
            use_2nd_uncond = True
        )
        text_batch_size = text_embeddings.shape[0] // 2
        
        # repeat the text embeddings w.r.t. the number of views
        """
            assume n_view = 4
            prompts: [
                promt_1,
                promt_2,
            ]
            ->
            [
                promt_1, view_1,
                ...
                promt_1, view_4,
                promt_2, view_1,
                ...
                promt_2, view_4,
            ]
            do so for text_embeddings_vd and text_embeddings_uncond
        """
        text_embeddings_vd     = text_embeddings[0 * text_batch_size: 1 * text_batch_size].repeat_interleave(
            view_batch_size // text_batch_size, dim = 0
        )
        text_embeddings_uncond = text_embeddings[1 * text_batch_size: 2 * text_batch_size].repeat_interleave(
            view_batch_size // text_batch_size, dim = 0
        )

        """
            assume n_view = 4
            prompts: [
                promt_1, view_1,
                ...
                promt_1, view_4,
                promt_2, view_1,
                ...
                promt_2, view_4,
            ] 
            ->
            [
                render_1st, promt_1, view_1,
                ...
                render_1st, promt_1, view_4,
                render_1st, promt_2, view_1,
                ...
                render_1st, promt_2, view_4,
                render_2nd, promt_1, view_1,
                ...
                render_2nd, promt_1, view_4,
                render_2nd, promt_2, view_1,
                ...
                render_2nd, promt_2, view_4,
            ]
            do so for text_embeddings_vd and text_embeddings_uncond
            then concatenate them
        """


        assert self.min_step is not None and self.max_step is not None
        with torch.no_grad():

            min_t, max_t = self.adapt_timestep_range(timestep_range)
            if is_dual and self.cfg.dual_render_sync_t:
                _t = torch.randint(
                    min_t,
                    max_t,
                    [text_batch_size],
                    dtype=torch.long,
                    device=self.device,
                )

                # repeat for the 1st and 2nd renderings
                _t = _t.repeat(2) 

                # bigger timestamp, the following is specific to ASD
                # as t_plus is randomly sampled in ASD, 
                # sample different t_plus for the 1st and 2nd renderings covers larger range
                _t_plus = self.get_t_plus(_t, module="mv")

            else:
                _t = torch.randint(
                    min_t,
                    max_t,
                    [text_batch_size if not is_dual else 2 * text_batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                # bigger timestamp, the following is specific to ASD
                _t_plus = self.get_t_plus(_t, module="mv")

            # keep consistent with the number of views for each prompt
            t = _t.repeat_interleave(self.cfg.mv_n_view)
            t_plus = _t_plus.repeat_interleave(self.cfg.mv_n_view)

            # perform noise prediction
            noise_pred_text, noise_pred_uncond, noise_pred_text_second = self._mv_noise_pred(
                mv_latents, 
                mv_noise,
                t,
                t_plus,
                text_embeddings_vd,
                text_embeddings_uncond,
                camera,
                fovy=fovy,
                is_dual=is_dual,
            )

        # determine the weight
        if self.cfg.mv_weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.mv_weighting_strategy == "uniform":
            w = 1
        elif self.cfg.mv_weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        elif self.cfg.mv_weighting_strategy == "sds_sqrt":
            w = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.mv_weighting_strategy}"
            )

        noise_pred_first = noise_pred_uncond + self.cfg.mv_guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        noise_pred_second = noise_pred_text_second

        grad = (noise_pred_first - noise_pred_second) * w
        grad = torch.nan_to_num(grad)
        # clip grad for stability?
        if self.grad_clip_val is not None:
            grad = torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (mv_latents - grad).detach()

        if not is_dual:
            loss_asd = 0.5 * F.mse_loss(mv_latents, target, reduction="sum") / view_batch_size 
            return loss_asd, grad.norm()
        else:
            # split the loss and grad_norm for the 1st and 2nd renderings
            loss_asd = torch.stack(
                [
                    0.5 * F.mse_loss(mv_latents[:view_batch_size], target[:view_batch_size], reduction="sum") / view_batch_size,
                    0.5 * F.mse_loss(mv_latents[view_batch_size:], target[view_batch_size:], reduction="sum") / view_batch_size,
                ]
            )
            grad_norm = torch.stack(
                [
                    grad[:view_batch_size].norm(),
                    grad[view_batch_size:].norm(),
                ]
            )
            return loss_asd, grad_norm

    def mv_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        **kwargs,
    ):
        if self.cfg.auto_grad:
            rgb_var = Variable(rgb, requires_grad=True)
            if rgb_2nd is not None:
                rgb_2nd_var = Variable(rgb_2nd, requires_grad=True)
                loss_asd, norm_asd = self._mv_grad_asd(
                    rgb_var, # change to rgb_var
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd_var, # change to rgb_2nd_var
                    **kwargs,
                )
                grad_rgb, grad_rgb_2nd = torch_grad(loss_asd.sum(), ([rgb_var, rgb_2nd_var]))
                loss_asd = torch.cat(
                    [
                        SpecifyGradient.apply(rgb, grad_rgb), 
                        SpecifyGradient.apply(rgb_2nd, grad_rgb_2nd)
                    ],
                    dim=0
                )
                return loss_asd, norm_asd
            else:
                loss_asd, norm_asd = self._mv_grad_asd(
                    rgb_var, # change to rgb_var
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd,
                    **kwargs,
                )
                grad_rgb = torch_grad(loss_asd, rgb_var)[0]
                loss_asd = SpecifyGradient.apply(rgb, grad_rgb)
                return loss_asd, norm_asd
        else:
            return self._mv_grad_asd(
                rgb,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                c2w,
                rgb_as_latents=rgb_as_latents,
                fovy=fovy,
                rgb_2nd=rgb_2nd,
                **kwargs,
            )


################################################################################################
# the following is specific to Stable Diffusion
    def _sd_encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.sd_vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.sd_vae.config.scaling_factor
        return latents.to(input_dtype)

    def sd_get_latents(
        self, 
        rgb_BCHW: Float[Tensor, "B C H W"], 
        rgb_BCHW_2nd: Optional[Float[Tensor, "B C H W"]] = None,
        rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            size = self.cfg.sd_image_size // 8
            latents = F.interpolate(
                rgb_BCHW, size=(size, size), mode="bilinear", align_corners=False
            )
            # resize the second latent if it exists
            if rgb_BCHW_2nd is not None:
                latents_2nd = F.interpolate(
                    rgb_BCHW_2nd, size=(size, size), mode="bilinear", align_corners=False
                )
                # concatenate the two latents
                latents = torch.cat([latents, latents_2nd], dim=0)
        else:
            size = self.cfg.sd_image_size
            rgb_BCHW_resize = F.interpolate(
                rgb_BCHW, size=(size, size), mode="bilinear", align_corners=False
            )
            # resize the second image if it exists
            if rgb_BCHW_2nd is not None:
                rgb_BCHW_2nd_resize = F.interpolate(
                    rgb_BCHW_2nd, size=(size, size), mode="bilinear", align_corners=False
                )
                # concatenate the two images
                rgb_BCHW_resize = torch.cat([rgb_BCHW_resize, rgb_BCHW_2nd_resize], dim=0)
            # encode image into latents
            latents = self._sd_encode_images(rgb_BCHW_resize)
        return latents


    def _sd_noise_pred(
        self,
        sd_latents: Float[Tensor, "B 4 64 64"],
        sd_noise: Float[Tensor, "B 4 64 64"],
        t: Float[Tensor, "B"],
        t_plus: Float[Tensor, "B"],
        text_embeddings_vd: Float[Tensor, "B 77 1024"],
        text_embeddings_uncond: Float[Tensor, "B 77 1024"],
        text_embeddings_vd_2nd: Optional[Float[Tensor, "B 77 1024"]] = None,
        text_embeddings_vd_neg: Optional[Float[Tensor, "B 77 1024"]] = None,
        is_dual: bool = False,
    ):
        """
            Wrapper for the noise prediction of the stable diffusion model
        """
        img_batch_size = sd_latents.shape[0]
        use_t_plus = self.cfg.sd_plus_ratio > 0 # important for the 2nd diffusion model

        # prepare text embeddings ################################################################################################

        if self.sd_use_perp_neg:
            assert text_embeddings_vd_neg is not None
            raise NotImplementedError("The perp_neg is not implemented yet")
            # text_embeddings = torch.cat(
            #     [
            #         text_embeddings_vd if not is_dual else text_embeddings_vd.repeat(2, 1, 1), # for the 1st diffusion model
            #         text_embeddings_uncond if not is_dual else text_embeddings_uncond.repeat(2, 1, 1),
            #         text_embeddings_vd_neg if not is_dual else text_embeddings_vd_neg.repeat(2, 1, 1),
            #         text_embeddings_vd if not is_dual else text_embeddings_vd.repeat(2, 1, 1), # for the 2nd diffusion model
            #     ],
            #     dim=0
            # )
        else:
            text_embeddings = [
                    text_embeddings_vd if not is_dual else 
                        torch.cat([text_embeddings_vd, text_embeddings_vd_2nd], dim=0),
                    text_embeddings_uncond if not is_dual else 
                        text_embeddings_uncond.repeat(2, 1, 1), # same for the 1st and 2nd renderings
            ]
            if use_t_plus: # require more text embeddings for the 2nd diffusion model
                text_embeddings += [
                    text_embeddings_vd if not is_dual else 
                        torch.cat([text_embeddings_vd, text_embeddings_vd_2nd], dim=0),
                ]
                # otherwise we save 1/3 batch size
            text_embeddings = torch.cat(
                text_embeddings,
                dim=0
            )

        # repeat the latents for the first diffusion model, if use_perp_neg is enabled, then
        # require 4 repeats, otherwise 2
        num_repeats = 2 if not self.sd_use_perp_neg else 4
 
        # prepare latent input for UNet ################################################################################################
        latents_noisy = self.sd_scheduler.add_noise(sd_latents, sd_noise, t)

        latents_model_input = [
            latents_noisy,
        ] * num_repeats 

        if use_t_plus:

            # random timestamp for the second diffusion model
            latents_noisy_second = self.sd_scheduler.add_noise(sd_latents, sd_noise, t_plus)

            latents_model_input += [
                latents_noisy_second,
            ] 

        latents_model_input = torch.cat(
            latents_model_input,
            dim=0,
        )

        # prepare timestep input for UNet ################################################################################################
        t_expand = [
            t,
        ] * num_repeats

        if use_t_plus:
            t_expand += [
                t_plus,
            ]

        t_expand = torch.cat(
            t_expand,
            dim=0,
        )

        # forward pass ################################################################################################
        noise_pred = self.sd_unet(
            latents_model_input.to(self.weights_dtype),
            t_expand.to(self.weights_dtype),
            encoder_hidden_states=text_embeddings.to(self.weights_dtype),
        ).sample.to(sd_latents.dtype)


        # split the noise_pred
        if self.sd_use_perp_neg:
            noise_pred_text    = noise_pred[0 * img_batch_size: 1 * img_batch_size]
            noise_pred_uncond  = noise_pred[1 * img_batch_size: 2 * img_batch_size]
            noise_pred_vd_neg  = noise_pred[2 * img_batch_size: 4 * img_batch_size]
            noise_pred_second  = noise_pred[4 * img_batch_size: 5 * img_batch_size] if use_t_plus else noise_pred_text
        else:
            noise_pred_text    = noise_pred[0 * img_batch_size: 1 * img_batch_size]
            noise_pred_uncond  = noise_pred[1 * img_batch_size: 2 * img_batch_size]
            noise_pred_vd_neg  = None
            noise_pred_second  = noise_pred[2 * img_batch_size: 3 * img_batch_size] if use_t_plus else noise_pred_text

        return noise_pred_text, noise_pred_uncond, noise_pred_vd_neg, noise_pred_second


    def _sd_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        azimuth_2nd: Optional[Float[Tensor, "B"]] = None,
        timestep_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        # determine if dual rendering is enabled
        is_dual = True if rgb_2nd is not None else False

        view_batch_size = rgb.shape[0] # the number of views 
        img_batch_size = rgb.shape[0] 
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        # special case for dual rendering
        if is_dual:
            img_batch_size *= 2
            rgb_2nd_BCHW = rgb_2nd.permute(0, 3, 1, 2)
            assert azimuth_2nd is not None, "azimuth_2nd is required for dual rendering"

        ################################################################################################
        # the following is specific to MVDream
        sd_latents = self.sd_get_latents(
            rgb_BCHW,
            rgb_BCHW_2nd=rgb_2nd_BCHW if is_dual else None,
            rgb_as_latents=rgb_as_latents,
        )

        # prepare noisy input
        sd_noise = torch.randn_like(sd_latents)

        # prepare text embeddings
        if not self.sd_use_perp_neg:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation=elevation, 
                azimuth=azimuth, 
                camera_distances=camera_distances, 
                view_dependent_prompting=self.cfg.sd_view_dependent_prompting,
                use_2nd_uncond = False
            )
            text_batch_size = text_embeddings.shape[0] // 2
            
            # repeat the text embeddings w.r.t. the number of views
            text_embeddings_vd     = text_embeddings[0 * text_batch_size: 1 * text_batch_size].repeat_interleave(
                view_batch_size // text_batch_size, dim = 0
            )
            text_embeddings_uncond = text_embeddings[1 * text_batch_size: 2 * text_batch_size].repeat_interleave(
                view_batch_size // text_batch_size, dim = 0
            )
            text_embeddings_vd_neg = None

            # if dual rendering is enabled, fetch the text embeddings for the 2nd rendering
            if is_dual:
                text_embeddings_vd_2nd = prompt_utils.get_text_embeddings(
                    elevation=elevation, 
                    azimuth=azimuth_2nd, 
                    camera_distances=camera_distances,
                    view_dependent_prompting=self.cfg.sd_view_dependent_prompting,
                    use_2nd_uncond = False
                )
                # we only need the view-dependent text embeddings
                text_embeddings_vd_2nd = text_embeddings_vd_2nd[0 * text_batch_size: 1 * text_batch_size].repeat_interleave(
                    view_batch_size // text_batch_size, dim = 0
                )

            neg_guidance_weights = None
        else:
            assert prompt_utils.use_perp_neg # just to make sure
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, 
                view_dependent_prompting=self.cfg.sd_view_dependent_prompting,
                use_2nd_uncond = False
            )

            raise NotImplementedError("The perp_neg is not implemented yet")
        
            # text_batch_size = text_embeddings.shape[0] // 4
            # neg_guidance_weights = neg_guidance_weights * -1 * self.cfg.sd_guidance_perp_neg # multiply by a negative value to control its magnitude

            # text_embeddings_vd     = text_embeddings[0 * text_batch_size: 1 * text_batch_size].repeat_interleave(
            #     view_batch_size // text_batch_size, dim = 0
            # )
            # text_embeddings_uncond = text_embeddings[1 * text_batch_size: 2 * text_batch_size].repeat_interleave(
            #     view_batch_size // text_batch_size, dim = 0
            # )
            # text_embeddings_vd_neg = text_embeddings[2 * text_batch_size: 4 * text_batch_size].repeat_interleave(
            #     view_batch_size // text_batch_size, dim = 0
            # )

            # if is_dual: # repeat the neg_guidance_weights
            #     neg_guidance_weights = neg_guidance_weights.repeat(2, 1)

        assert self.min_step is not None and self.max_step is not None
        with torch.no_grad():

            min_t, max_t = self.adapt_timestep_range(timestep_range)
            if is_dual and self.cfg.dual_render_sync_t:
                t = torch.randint(
                    min_t,
                    max_t,
                    [view_batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                
                t = t.repeat(2) # repeat for the 1st and 2nd renderings

                # bigger timestamp, the following is specific to ASD
                # as t_plus is randomly sampled in ASD, 
                # sample different t_plus for the 1st and 2nd renderings covers larger range
                t_plus = self.get_t_plus(t, module="sd")
                
            else:
                t = torch.randint(
                    min_t,
                    max_t,
                    [img_batch_size],
                    dtype=torch.long,
                    device=self.device,
                )

                # bigger timestamp, the following is specific to ASD
                t_plus = self.get_t_plus(t, module="sd")

            # perform the noise prediction
            noise_pred_text, noise_pred_uncond, noise_pred_vd_neg, noise_pred_second = self._sd_noise_pred(
                sd_latents,
                sd_noise,
                t,
                t_plus,
                text_embeddings_vd,
                text_embeddings_uncond,
                text_embeddings_vd_2nd = text_embeddings_vd_2nd if is_dual else None,
                text_embeddings_vd_neg = text_embeddings_vd_neg if self.sd_use_perp_neg else None,
                is_dual = is_dual,
            )

                
        # determine the weight
        if self.cfg.sd_weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.sd_weighting_strategy == "uniform":
            w = 1
        elif self.cfg.sd_weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        elif self.cfg.sd_weighting_strategy == "sds_sqrt":
            w = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.sd_weighting_strategy}"
            )

        # aggregate the noise_pred
        eps_pos = noise_pred_text - noise_pred_uncond
        if neg_guidance_weights is not None: # use_perp_neg is enabled
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                eps_vd_neg = noise_pred_vd_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, *[1] * (eps_vd_neg.ndim - 1)
                ) * perpendicular_component(eps_vd_neg, eps_pos) # eps_vd_neg # v2

            # noise_pred_p = (eps_pos) * guidenace_scale + noise_pred_uncond + accum_grad
            noise_pred_first = (eps_pos + accum_grad) * self.cfg.sd_guidance_scale + noise_pred_uncond 

        else: # if not use_perp_neg
            noise_pred_first = eps_pos                * self.cfg.sd_guidance_scale + noise_pred_uncond

        grad = (noise_pred_first - noise_pred_second) * w
        grad = torch.nan_to_num(grad)
        # clip grad for stability?
        if self.grad_clip_val is not None:
            grad = torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (sd_latents - grad).detach()
        if not is_dual:
            loss_asd = 0.5 * F.mse_loss(sd_latents, target, reduction="sum") / view_batch_size
            return loss_asd, grad.norm()
        else:
            # split the grad into two parts
            loss_asd = torch.stack(
                [
                    0.5 * F.mse_loss(sd_latents[:view_batch_size], target[:view_batch_size], reduction="sum") / view_batch_size,
                    0.5 * F.mse_loss(sd_latents[view_batch_size:], target[view_batch_size:], reduction="sum") / view_batch_size,
                ]
            )
            grad_norm = torch.stack(
                [
                    grad[:view_batch_size].norm(),
                    grad[view_batch_size:].norm(),
                ]
            )
            return loss_asd, grad_norm

    def sd_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        azimuth_2nd: Optional[Float[Tensor, "B"]] = None,
        **kwargs,
    ):
        if self.cfg.auto_grad:
            rgb_var = Variable(rgb, requires_grad=True)
            if rgb_2nd is not None:
                rgb_2nd_var = Variable(rgb_2nd, requires_grad=True)
                loss_asd, norm_asd = self._sd_grad_asd(
                    rgb_var, # change to rgb_var
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd_var, # change to rgb_2nd_var
                    azimuth_2nd=azimuth_2nd,
                    **kwargs,
                )
                grad_rgb, grad_rgb_2nd = torch_grad(loss_asd.sum(), ([rgb_var, rgb_2nd_var]))
                loss_asd = torch.cat(
                    [
                        SpecifyGradient.apply(rgb, grad_rgb), 
                        SpecifyGradient.apply(rgb_2nd, grad_rgb_2nd)
                    ],
                    dim=0
                )
                return loss_asd, norm_asd
            else:
                loss_asd, norm_asd = self._sd_grad_asd(
                    rgb_var, # change to rgb_var
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd,
                    azimuth_2nd=azimuth_2nd,
                    **kwargs,
                )
                grad_rgb = torch_grad(loss_asd, rgb_var)[0]
                loss_asd = SpecifyGradient.apply(rgb, grad_rgb)
                return loss_asd, norm_asd
        else:
            return self._sd_grad_asd(
                rgb,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                c2w,
                rgb_as_latents=rgb_as_latents,
                fovy=fovy,
                rgb_2nd=rgb_2nd,
                azimuth_2nd=azimuth_2nd,
                **kwargs,
            )
        
################################################################################################


    def _one_of_n_view(
        self,
        view_batch_size: int,
        differ_from_idx: Optional[Float[Tensor, "B"]] = None
    ):

        if self.cfg.sd_view_dependent_prompting: 
            # if view_dependent_prompting is enabled, we can select any view (azimuth: -180, 180) for the guidance
            idx = torch.randint(
                0, self.cfg.mv_n_view, 
                (view_batch_size // self.cfg.mv_n_view, ), device=self.device, dtype=torch.long
            )
        else:
            # otherwise, select the frontal view (azimuth: -90, 90) for the guidance
            # because sd is more capable of handling the frontal view
            assert self.cfg.mv_n_view % 4 == 0 # only support 4, 8, 12, 16, ...
            left = self.cfg.mv_n_view // 4 * 1
            right = self.cfg.mv_n_view // 4 * 2 + 1
            idx = torch.randint(
                left, right, 
                (view_batch_size // self.cfg.mv_n_view, ), device=self.device, dtype=torch.long
            )

        # if differ_from_idx is not None, select the different view idx for the guidance
        if differ_from_idx is not None:
            orig_idx = (
                differ_from_idx - # remove the added idx offset
                torch.arange(0, view_batch_size, self.cfg.mv_n_view, device=self.device, dtype=torch.long)
            )
            offset = idx.clamp(min=1, max=self.cfg.mv_n_view - 1) # the offset should be in [1, n_view - 1]
            idx = (orig_idx + offset) % self.cfg.mv_n_view

        idx += torch.arange(0, view_batch_size, self.cfg.mv_n_view, device=self.device, dtype=torch.long)
        return idx

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
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        **kwargs,
    ):

        """
            # illustration of the concatenated rgb and rgb_2nd, assume n_view = 4
            # rgb: Float[Tensor, "B H W C"]
            [
                render_1st, prompt_1, view_1,
                ...
                render_1st, prompt_1, view_4,
                render_1st, prompt_2, view_1,
                ...
                render_1st, prompt_2, view_4,
                render_2nd, prompt_1, view_1,
                ...
                render_2nd, prompt_1, view_4,
                render_2nd, prompt_2, view_1,
                ...
                render_2nd, prompt_2, view_4,
            ]
        """
        # determine if dual rendering is enabled
        is_dual = True if rgb_2nd is not None else False

        ################################################################################################
        # the following is specific to MVDream
        if self.cfg.mv_weight > 0:
            if is_dual and self.cfg.mv_2nd_render_only:
                # only use the 2nd rendering for the guidance
                loss_mv, grad_mv = self.mv_grad_asd(
                    rgb_2nd,
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    **kwargs,
                )
            else:
                loss_mv, grad_mv = self.mv_grad_asd(
                    rgb,
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd,
                    **kwargs,
                )
        else:
            loss_mv = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0],
                device=self.device
            )
            grad_mv = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0],
                device=self.device
            )
    
       ################################################################################################

        # due to the computation cost
        # the following is specific to Stable Diffusion
        # for any n_view, select only one view for the guidance
        if self.cfg.sd_all_views:
            idx = torch.arange(0, rgb.shape[0], device=self.device, dtype=torch.long)
            if is_dual:
                idx_2nd = torch.arange(0, rgb_2nd.shape[0], device=self.device, dtype=torch.long)
        else: # only select one view for the guidance
            idx = self._one_of_n_view(
                view_batch_size=rgb.shape[0]
            )

            # special case for dual rendering
            if is_dual:
                if self.cfg.dual_render_sync_view_sd: # select the same view for the 1st and 2nd renderings
                    idx_2nd = idx
                else:                                 # select different views for the 1st and 2nd renderings
                    idx_2nd = self._one_of_n_view(
                        view_batch_size=rgb_2nd.shape[0],
                        differ_from_idx=idx
                    )

        # select only one view for the guidance
        if self.cfg.sd_weight > 0:
            if is_dual and self.cfg.sd_1st_render_only:
                # only use the 1st rendering for the guidance
                loss_sd, grad_sd = self.sd_grad_asd(
                    rgb[idx],
                    prompt_utils,
                    elevation[idx],
                    azimuth[idx],
                    camera_distances[idx],
                    c2w[idx],
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    **kwargs,
                )
            else:
                loss_sd, grad_sd = self.sd_grad_asd(
                    rgb[idx],
                    prompt_utils,
                    elevation[idx],
                    azimuth[idx],
                    camera_distances[idx],
                    c2w[idx],
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd[idx_2nd] if is_dual else None,
                    azimuth_2nd=azimuth[idx_2nd] if is_dual else None, # mvdream's n_view only varies azimuth
                    **kwargs,
                )
        else:
            loss_sd = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0],
                device=self.device
            )
            grad_sd = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0],
                device=self.device
            )

        # return the loss and grad_norm
        if not is_dual:
            return {
                "loss_asd": self.cfg.sd_weight * loss_sd + self.cfg.mv_weight * loss_mv,
                "grad_norm_asd": self.cfg.sd_weight * grad_sd + self.cfg.mv_weight * grad_mv,
                # "min_step": self.min_step,
                # "max_step": self.max_step,
            }
        else:
            # return the loss and grad_norm for the 1st renderings
            loss = 0
            grad_norm = 0
            # special case for sd_1st_render_only
            loss += self.cfg.sd_weight * (loss_sd if self.cfg.sd_1st_render_only else loss_sd[0])
            grad_norm += self.cfg.sd_weight * (grad_sd if self.cfg.sd_1st_render_only else grad_sd[0])
            # special case for mv_2nd_render_only
            loss += self.cfg.mv_weight * (0 if self.cfg.mv_2nd_render_only else loss_mv[0])
            grad_norm += self.cfg.mv_weight * (0 if self.cfg.mv_2nd_render_only else grad_mv[0])

            guidance_1st =  {
                "loss_asd": loss,
                "grad_norm_asd": grad_norm,
                # "min_step": self.min_step,
                # "max_step": self.max_step,
            }

            # return the loss and grad_norm for the 2nd renderings
            loss = 0
            grad_norm = 0
            # special case for sd_1st_render_only
            loss += self.cfg.sd_weight * (0 if self.cfg.sd_1st_render_only else loss_sd[1])
            grad_norm += self.cfg.sd_weight * (0 if self.cfg.sd_1st_render_only else grad_sd[1])
            # special case for mv_2nd_render_only
            loss += self.cfg.mv_weight * (loss_mv if self.cfg.mv_2nd_render_only else loss_mv[1])
            grad_norm += self.cfg.mv_weight * (grad_mv if self.cfg.mv_2nd_render_only else grad_mv[1])
                                               
            guidance_2nd =  {
                "loss_asd": loss,
                "grad_norm_asd": grad_norm,
                # "min_step": self.min_step,
                # "max_step": self.max_step,
            }
            return guidance_1st, guidance_2nd


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

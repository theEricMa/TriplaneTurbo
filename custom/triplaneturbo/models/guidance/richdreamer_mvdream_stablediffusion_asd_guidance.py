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

from extern.mvdream.model_zoo import build_model as build_model_mv
from extern.mvdream.camera_utils import normalize_camera

from extern.nd_sd.model_zoo import build_model as build_model_rd

from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from torch.nn.modules.container import ModuleList
from extern.mvdream.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from scipy.spatial.transform import Rotation as R

from torch.autograd import Variable, grad as torch_grad
from threestudio.utils.ops import SpecifyGradient

@threestudio.register("richdreamer-mvdream-stablediffusion-asynchronous-score-distillation-guidance")
class RDMVASDsynchronousScoreDistillationGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # specify the model name for the mvdream model and the stable diffusion model
        mv_model_name_or_path: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        mv_ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        
        rd_model_name_or_path: str = "nd-4view"
        rd_ckpt_path: Optional[
            str
        ] = None

        sd_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"

        n_view: int = 4

        # the following is specific to richdreamer
        rd_min_step_percent: Optional[float] = None
        rd_max_step_percent: Optional[float] = None
        rd_image_size: int = 32
        rd_guidance_scale: float = 7.5
        rd_weight: float = 1.
        rd_weighting_strategy: str = "uniform" # asd is suitable for uniform weighting, but can be extended to other strategies
        rd_use_sds: bool = False
        rd_recon_std_rescale: float = 0.
        cam_method: str = "rel_x2"  # rel_x2 or abs or rel

        # the following is specific to mvdream
        mv_min_step_percent: Optional[float] = None
        mv_max_step_percent: Optional[float] = None
        mv_camera_condition_type: str = "rotation"
        mv_image_size: int = 256
        mv_guidance_scale: float = 7.5
        mv_weight: float = 0.25 # 1 / 4
        mv_weighting_strategy: str = "uniform" # asd is suitable for uniform weighting, but can be extended to other strategies

        # the following is specific to stable diffusion
        sd_min_step_percent: Optional[float] = None
        sd_max_step_percent: Optional[float] = None
        sd_image_size: int = 512
        sd_guidance_scale: float = 7.5
        sd_weight: float = 1.
        sd_weighting_strategy: str = "uniform" # ASD is suitable for uniform weighting, but can be extended to other strategies
        sd_hq_only: bool = False
        sd_oppo_side: bool = False

        # the following is shared between mvdream and stable diffusion
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        max_step_percent: float = 0.98
        min_step_percent: float = 0.02

        


        plus_schedule: str = "linear"  # linear or sqrt_<bias>

        # the following is specific to the combination of MVDream and Stable Diffusion with asd
        mv_plus_random: bool = True
        mv_plus_ratio: float = 0.1
        rd_plus_random: bool = True
        rd_plus_ratio: float = 0.1
        sd_plus_random: bool = True
        sd_plus_ratio: float = 0.1
        t_plus_type: str = "v1" # v1, v2, v3, v4

        # strategy to save memory
        gradient_checkpoint: bool = False
        auto_grad: bool = False

        eps: float = 0.01

    cfg: Config



    def configure(self) -> None:

        ################################################################################################
        if type(self.cfg.sd_weight) in [float, int] and self.cfg.sd_weight == 0:
            threestudio.info("Stable Diffusion is disabled.")
        else:
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

        ################################################################################################
        if type(self.cfg.mv_weight) in [float, int] and self.cfg.mv_weight == 0:
            threestudio.info("MVDream is disabled.")
        else:
            threestudio.info(f"Loading MVDream ...")

            self.mv_model = build_model_mv(
                self.cfg.mv_model_name_or_path,
                ckpt_path=self.cfg.mv_ckpt_path
            ).to(self.device)
            for p in self.mv_model.parameters():
                p.requires_grad_(False)

            if hasattr(self.mv_model, "cond_stage_model"):
                # delete unused models
                del self.mv_model.cond_stage_model # text encoder
                cleanup()

        ################################################################################################
        if type(self.cfg.rd_weight) in [float, int] and self.cfg.rd_weight == 0:
            threestudio.info(f"RichDreamer is disabled")
        else:
            threestudio.info(f"Loading RichDreamer ...")
            rd_model, rd_cfg = build_model_rd(
                self.cfg.rd_model_name_or_path,
                ckpt_path=self.cfg.rd_ckpt_path,
                return_cfg=True,
                strict=False,
            )
            self.rd_model = rd_model.to(self.device)
            for p in self.rd_model.parameters():
                p.requires_grad_(False)

            self.rd_cond_method = (
                rd_cfg.model.params.cond_method
                if hasattr(rd_cfg.model.params, "cond_method")
                else "ori"
            )
            if hasattr(self.rd_model, "cond_stage_model"):
                # delete unused models
                del self.rd_model.cond_stage_model # text encoder
                cleanup()

        ################################################################################################
        # the folowing is shared between mvdream and stable diffusion
        self.alphas = self.mv_model.alphas_cumprod if hasattr(self, "mv_model") else self.rd_model.alphas_cumprod
        self.grad_clip_val: Optional[float] = None
        self.num_train_timesteps = 1000


    def get_t_plus(
        self, 
        t: Float[Tensor, "B"],
        module: str # "rd" or "mv"
    ):

        
        # determine the attributes that differ between rd and MV
        if module == "rd":
            plus_random = self.cfg.rd_plus_random
            plus_ratio = self.cfg.rd_plus_ratio
            min_step = self.rd_min_step
            max_step = self.rd_max_step
        elif module == "mv":
            plus_random = self.cfg.mv_plus_random
            plus_ratio = self.cfg.mv_plus_ratio
            min_step = self.mv_min_step
            max_step = self.mv_max_step
        elif module == "sd":
            plus_random = self.cfg.sd_plus_random
            plus_ratio = self.cfg.sd_plus_ratio
            min_step = self.sd_min_step
            max_step = self.sd_max_step
        else:
            raise ValueError(f"Invalid module: {module}")

        # determine the timestamp for the second diffusion model
        if self.cfg.plus_schedule == "linear":

            if self.cfg.t_plus_type in ["v1", "v2"]:
                t_plus = plus_ratio * (t - min_step)
            elif self.cfg.t_plus_type in ["v3", "v4"]:
                t_plus = plus_ratio * t
            else:
                raise ValueError(f"Invalid t_plus_type: {self.cfg.t_plus_type}")
                    
        elif self.cfg.plus_schedule.startswith("sqrt"):
            bias = 0
            if self.cfg.plus_schedule.startswith("sqrt_"): # if bias is specified
                try:
                    bias = float(self.cfg.plus_schedule.split("_")[1])
                except:
                    raise ValueError(f"Invalid sqrt bias: {self.cfg.plus_schedule}")
                
            t_plus = plus_ratio * torch.sqrt(t + bias)
        else:
            raise ValueError(f"Invalid plus_schedule: {self.cfg.plus_schedule}")

        # clamp t_plus to the range [0, T_max - t], added in the revision
        if self.cfg.t_plus_type in ["v1", "v3"]:
            t_plus = torch.clamp(
                t_plus,
                torch.zeros_like(t), 
                self.num_train_timesteps - t - 1,
            )
        elif self.cfg.t_plus_type in ["v2", "v4"]:
            t_plus = torch.clamp(
                t_plus,
                0,
                self.num_train_timesteps - 1,
            )
        else:
            raise ValueError(f"Invalid t_plus_type: {self.cfg.t_plus_type}")

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
            "num_frames": self.cfg.n_view,
        }

        # u-net forwasd pass ################################################################################################
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
        _, text_embeddings = prompt_utils.get_text_embeddings() # mvdream uses the second text embeddings
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
        text_embeddings = torch.cat(
            [
                text_embeddings_vd if not is_dual else text_embeddings_vd.repeat(2, 1, 1),
                text_embeddings_uncond if not is_dual else text_embeddings_uncond.repeat(2, 1, 1),
                text_embeddings_vd if not is_dual else text_embeddings_vd.repeat(2, 1, 1),
            ], 
            dim=0
        )

        with torch.no_grad():

            min_t, max_t = self.mv_min_step, self.mv_max_step
            _t = torch.randint(
                min_t,
                max_t,
                [text_batch_size if not is_dual else 2 * text_batch_size],
                dtype=torch.long,
                device=self.device,
            )
            # bigger timestamp, the following is specific to asd
            _t_plus = self.get_t_plus(_t, module="mv")

            # keep consistent with the number of views for each prompt
            t = _t.repeat_interleave(self.cfg.n_view)
            t_plus = _t_plus.repeat_interleave(self.cfg.n_view)

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

        noise_pred_first = noise_pred_uncond + self.cfg.mv_guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        noise_pred_second = noise_pred_text_second

        # determine the weight
        if self.cfg.mv_weighting_strategy in ["sds", "uniform", "fantasia3d", "sds_sqrt"]:
            if self.cfg.mv_weighting_strategy == "sds":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.mv_weighting_strategy == "uniform":
                w = 1
            elif self.cfg.mv_weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            elif self.cfg.mv_weighting_strategy == "sds_sqrt":
                w = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)

            grad = (noise_pred_first - noise_pred_second) * w

        elif self.cfg.mv_weighting_strategy in ["dmd", "dmd+1", "dmd_mv", "dmd_mv+1"]:
            # eps to mu
            with torch.no_grad():
                alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
                sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
                latent_first = (mv_latents - sigma * noise_pred_first) / alpha
                latent_second = (mv_latents - sigma * noise_pred_second) / alpha

                if self.cfg.mv_weighting_strategy == "dmd":
                    w = torch.abs(mv_latents - latent_first).mean(dim=(1, 2, 3), keepdim=True)
                elif self.cfg.mv_weighting_strategy == "dmd_mv":
                    w = torch.abs(mv_latents - latent_first).mean(dim=(1, 2, 3))
                    w = w.view(img_batch_size // self.cfg.n_view, self.cfg.n_view).mean(dim=-1, keepdim=True).repeat_interleave(self.cfg.n_view, dim=0)
                    w = w.view(-1, 1, 1, 1)
                elif self.cfg.mv_weighting_strategy == "dmd+1":
                    _noise_pred_second =  noise_pred_uncond + self.cfg.mv_guidance_scale * (
                        noise_pred_text_second - noise_pred_uncond
                    )
                    _alpha = (self.alphas[t_plus] ** 0.5).view(-1, 1, 1, 1)
                    _sigma = ((1 - self.alphas[t_plus]) ** 0.5).view(-1, 1, 1, 1)
                    _latent_second = (mv_latents - _sigma * _noise_pred_second) / _alpha
                    w = torch.abs(mv_latents - _latent_second).mean(dim=(1, 2, 3), keepdim=True)
                elif self.cfg.mv_weighting_strategy == "dmd_mv+1":
                    _noise_pred_second =  noise_pred_uncond + self.cfg.mv_guidance_scale * (
                        noise_pred_text_second - noise_pred_uncond
                    )
                    _alpha = (self.alphas[t_plus] ** 0.5).view(-1, 1, 1, 1)
                    _sigma = ((1 - self.alphas[t_plus]) ** 0.5).view(-1, 1, 1, 1)
                    _latent_second = (mv_latents - _sigma * _noise_pred_second) / _alpha
                    w = torch.abs(mv_latents - _latent_second).mean(dim=(1, 2, 3), keepdim=True)
                    w = w.view(img_batch_size // self.cfg.n_view, self.cfg.n_view).mean(dim=-1, keepdim=True).repeat_interleave(self.cfg.n_view, dim=0)
                    w = w.view(-1, 1, 1, 1)

            grad = (latent_second - latent_first) / (w + self.cfg.eps) # avoid zero division

        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.mv_weighting_strategy}"
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stability?
        if self.grad_clip_val is not None:
            grad = torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (mv_latents - grad).detach()

        if not is_dual:
            loss_asd = 0.5 * F.mse_loss(mv_latents, target, reduction="sum") / self.cfg.n_view
            return loss_asd, grad.norm()
        else:
            # split the loss and grad_norm for the 1st and 2nd renderings
            loss_asd = torch.stack(
                [
                    0.5 * F.mse_loss(mv_latents[:view_batch_size], target[:view_batch_size], reduction="sum") / self.cfg.n_view,
                    0.5 * F.mse_loss(mv_latents[view_batch_size:], target[view_batch_size:], reduction="sum") / self.cfg.n_view,
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
    def _rd_get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        distances: Float[Tensor, "B"],
        fovy=None,
    ):
        camera = normalize_camera(camera)
        camera = camera.view(-1, 4, 4)

        r = R.from_euler("z", -90, degrees=True).as_matrix()
        rotate_mat = torch.eye(4, dtype=camera.dtype, device=camera.device)
        rotate_mat[:3, :3] = torch.from_numpy(r)
        rotate_mat = rotate_mat.unsqueeze(0).repeat(camera.shape[0], 1, 1)
        camera = torch.matmul(rotate_mat, camera)

        distances = distances[:, None]
        camera[..., :3, 3] = camera[..., :3, 3] * distances
        camera = camera.flatten(start_dim=1)

        return camera

    def rd_get_latents(
        self, 
        geo_BCHW: Float[Tensor, "B C H W"], 
        geo_BCHW_2nd: Optional[Float[Tensor, "B C H W"]] = None,
    ) -> Float[Tensor, "B 4 64 64"]:

        size = self.cfg.rd_image_size
        geo_BCHW_resize = F.adaptive_avg_pool2d(
            geo_BCHW, output_size=(size, size)
        )
        # resize the second image if it exists
        if geo_BCHW_2nd is not None:
            geo_BCHW_2nd_resize = F.adaptive_avg_pool2d(
                geo_BCHW_2nd, output_size=(size, size)
            )
            # concatenate the two images
            geo_BCHW_resize = torch.cat([geo_BCHW_resize, geo_BCHW_2nd_resize], dim=0)
        return geo_BCHW_resize


    def _rd_noise_pred(
        self,
        rd_latents: Float[Tensor, "B 4 64 64"],
        rd_noise: Float[Tensor, "B 4 64 64"],
        t: Float[Tensor, "B"],
        t_plus: Float[Tensor, "B"],
        text_embeddings_cond: Float[Tensor, "B 77 1024"],
        text_embeddings_uncond: Float[Tensor, "B 77 1024"],
        camera: Float[Tensor, "B 4 4"],
        camera_distances: Float[Tensor, "B"],
        fovy=None,
        is_dual: bool = False,
    ):
        """
            Wrapper for the noise prediction of the RichDreamer model
        """
        # determine attributes ################################################################################################
        use_t_plus = self.cfg.rd_plus_ratio > 0 and not self.cfg.rd_use_sds

        # prepare text embeddings ################################################################################################
        text_embeddings = [
            text_embeddings_cond if not is_dual else text_embeddings_cond.repeat(2, 1, 1), # for the 1st diffusion model's conditional guidance
            text_embeddings_uncond if not is_dual else text_embeddings_uncond.repeat(2, 1, 1), # for the 1st diffusion model's unconditional guidance
        ]
        if use_t_plus:
            text_embeddings += [
                text_embeddings_cond if not is_dual else text_embeddings_cond.repeat(2, 1, 1), # for the 2nd diffusion model's conditional guidance
            ]
        text_embeddings = torch.cat(
            text_embeddings,
            dim=0
        )

        # prepare noisy input ################################################################################################
        # random timestamp for the first diffusion model
        latents_noisy = self.rd_model.q_sample(rd_latents, t, noise=rd_noise)

        latents_model_input = [
            latents_noisy,
        ] * 2

        if use_t_plus:
            # random timestamp for the second diffusion model
            latents_noisy_second = self.rd_model.q_sample(rd_latents, t_plus, noise=rd_noise)

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
        camera = self._rd_get_camera_cond(camera, fovy=fovy, distances=camera_distances)
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
            "num_frames": self.cfg.n_view,
        }

        # u-net forwasd pass ################################################################################################
        noise_pred = self.rd_model.apply_model(
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
            noise_pred_text_second = noise_pred_text if not self.cfg.rd_use_sds else rd_noise

        return noise_pred_text, noise_pred_uncond, noise_pred_text_second


    def _rd_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        normal_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        depth_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        timestep_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):


        camera = c2w

        # determine if dual rendering is enabled
        is_dual = True if normal_2nd is not None else False

        view_batch_size = normal.shape[0]
        img_batch_size = normal.shape[0]
        # rgb_BCHW = rgb.permute(0, 3, 1, 2)
        geo_BCHW = torch.cat([normal, depth], dim=-1).permute(0, 3, 1, 2)

        # special case for dual rendering
        if is_dual:
            img_batch_size *= 2
            # rgb_2nd_BCHW = rgb_2nd.permute(0, 3, 1, 2)
            geo_2nd_BCHW = torch.cat([normal_2nd, depth_2nd], dim=-1).permute(0, 3, 1, 2)

        rd_latents = self.rd_get_latents(
            geo_BCHW,
            geo_BCHW_2nd=geo_2nd_BCHW if is_dual else None,
        )        

        # prepare noisy input
        rd_noise = torch.randn_like(rd_latents)

        # prepare text embeddings
        text_embeddings, _ = prompt_utils.get_text_embeddings() # rdreamer uses the first text embeddings
        text_batch_size = text_embeddings.shape[0] // 2
        # same as _mv_grad_asd
        text_embeddings_cond     = text_embeddings[0 * text_batch_size: 1 * text_batch_size].repeat_interleave(
            view_batch_size // text_batch_size, dim = 0
        )
        text_embeddings_uncond = text_embeddings[1 * text_batch_size: 2 * text_batch_size].repeat_interleave(
            view_batch_size // text_batch_size, dim = 0
        )

        with torch.no_grad():
            min_t, max_t = self.rd_min_step, self.rd_max_step
            _t = torch.randint(
                min_t,
                max_t,
                [text_batch_size if not is_dual else 2 * text_batch_size],
                dtype=torch.long,
                device=self.device,
            )
            # bigger timestamp, the following is specific to asd
            _t_plus = self.get_t_plus(_t, module="rd")

            # keep consistent with the number of views for each prompt
            t = _t.repeat_interleave(self.cfg.n_view)
            t_plus = _t_plus.repeat_interleave(self.cfg.n_view)

            # perform noise prediction
            noise_pred_text, noise_pred_uncond, noise_pred_text_second = self._rd_noise_pred(
                rd_latents,
                rd_noise,
                t,
                t_plus,
                text_embeddings_cond,
                text_embeddings_uncond,
                camera,
                camera_distances,
                fovy=fovy,
                is_dual=is_dual,
            )

            noise_pred_first = noise_pred_uncond + self.cfg.rd_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            noise_pred_second = noise_pred_text_second

        if self.cfg.rd_weighting_strategy in ["sds", "uniform", "fantasia3d", "sds_sqrt"]:
            # determine the weight
            if self.cfg.rd_weighting_strategy == "sds":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.rd_weighting_strategy == "uniform":
                w = 1
            elif self.cfg.rd_weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            elif self.cfg.rd_weighting_strategy == "sds_sqrt":
                w = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)


            if self.cfg.rd_recon_std_rescale > 0:
                # gradient in the latent space
                latents_noisy = self.rd_model.q_sample(rd_latents, t, noise=rd_noise)
                latents_recon = self.rd_model.predict_start_from_noise(latents_noisy, t, noise_pred_first)
                # rescale the reconstruction
                latents_recon_nocfg = self.rd_model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
                factor = (
                    latents_recon_nocfg.view(-1,self.cfg.n_view, *latents_recon_nocfg.shape[1:]).std(dim=[1,2,3,4], keepdim=True) +
                    1e-8
                ) / (
                    latents_recon.view(-1,self.cfg.n_view, *latents_recon.shape[1:]).std(dim=[1,2,3,4], keepdim=True) +
                    1e-8
                )
                latents_recon = self.cfg.rd_recon_std_rescale * (
                    latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
                ) + (1 - self.cfg.rd_recon_std_rescale) * latents_recon

                # determine if second latent is used
                use_t_plus = self.cfg.rd_plus_ratio > 0 and not self.cfg.rd_use_sds
                if use_t_plus:
                    # overwrite the rd_latents with the second laten
                    latents_noisy_second = self.rd_model.q_sample(rd_latents, t_plus, noise=rd_noise)
                    latents = self.rd_model.predict_start_from_noise(latents_noisy_second, t_plus, noise_pred_second)
                else:
                    latents = rd_latents

                loss = 0.5 * F.mse_loss(latents, latents_recon, reduction="sum") / self.cfg.n_view
                grad = torch_grad(loss, rd_latents)[0]
            else:
                # gradient in the noise space
                grad = (noise_pred_first - noise_pred_second) * w

        elif self.cfg.rd_weighting_strategy in ["dmd", "dmd+1", "dmd_mv", "dmd_mv+1"]:
            with torch.no_grad():
                alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
                sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
                latent_first = (rd_latents - sigma * noise_pred_first) / alpha
                latent_second = (rd_latents - sigma * noise_pred_second) / alpha
                
                if self.cfg.rd_weighting_strategy == "dmd":
                    w = torch.abs(rd_latents - latent_first).mean(dim=(1, 2, 3), keepdim=True)
                elif self.cfg.rd_weighting_strategy == "dmd_mv":
                    w = torch.abs(rd_latents - latent_first).mean(dim=(1, 2, 3))
                    w = w.view(img_batch_size // self.cfg.n_view, self.cfg.n_view).mean(dim=-1, keepdim=True).repeat_interleave(self.cfg.n_view, dim=0)
                    w = w.view(-1, 1, 1, 1)
                elif self.cfg.rd_weighting_strategy == "dmd+1":
                    _noise_pred_second =  noise_pred_uncond + self.cfg.rd_guidance_scale * (
                        noise_pred_text_second - noise_pred_uncond
                    )
                    _alpha = (self.alphas[t_plus] ** 0.5).view(-1, 1, 1, 1)
                    _sigma = ((1 - self.alphas[t_plus]) ** 0.5).view(-1, 1, 1, 1)
                    _latent_second = (rd_latents - _sigma * _noise_pred_second) / _alpha
                    w = torch.abs(rd_latents - _latent_second).mean(dim=(1, 2, 3), keepdim=True)
                elif self.cfg.rd_weighting_strategy == "dmd_mv+1":
                    _noise_pred_second =  noise_pred_uncond + self.cfg.rd_guidance_scale * (
                        noise_pred_text_second - noise_pred_uncond
                    )
                    _alpha = (self.alphas[t_plus] ** 0.5).view(-1, 1, 1, 1)
                    _sigma = ((1 - self.alphas[t_plus]) ** 0.5).view(-1, 1, 1, 1)
                    _latent_second = (rd_latents - _sigma * _noise_pred_second) / _alpha
                    w = torch.abs(rd_latents - _latent_second).mean(dim=(1, 2, 3), keepdim=True)
                    w = w.view(img_batch_size // self.cfg.n_view, self.cfg.n_view).mean(dim=-1, keepdim=True).repeat_interleave(self.cfg.n_view, dim=0)
                    w = w.view(-1, 1, 1, 1)
            
                grad = (latent_second - latent_first) / (w + self.cfg.eps)

        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.rd_weighting_strategy}"
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stability?
        if self.grad_clip_val is not None:
            grad = torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (rd_latents - grad).detach()

        if not is_dual:
            loss_asd = 0.5 * F.mse_loss(rd_latents, target, reduction="sum") / self.cfg.n_view
            return loss_asd, grad.norm()
        else:
            # split the grad into two parts
            loss_asd = torch.stack(
                [
                    0.5 * F.mse_loss(rd_latents[:view_batch_size], target[:view_batch_size], reduction="sum") / self.cfg.n_view,
                    0.5 * F.mse_loss(rd_latents[view_batch_size:], target[view_batch_size:], reduction="sum") / self.cfg.n_view,
                ]
            )
            grad_norm = torch.stack(
                [
                    grad[:view_batch_size].norm(),
                    grad[view_batch_size:].norm(),
                ]
            )
            return loss_asd, grad_norm

    def rd_grad_asd(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        normal_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        depth_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        **kwargs,
    ):
        if self.cfg.auto_grad:
            rgb_var = Variable(rgb, requires_grad=True)
            normal_var = Variable(normal, requires_grad=True)
            depth_var = Variable(depth, requires_grad=True)
            if rgb_2nd is not None:
                rgb_2nd_var = Variable(rgb_2nd, requires_grad=True)
                normal_2nd_var = Variable(normal_2nd, requires_grad=True)
                depth_2nd_var = Variable(depth_2nd, requires_grad=True)
                loss_asd, norm_asd = self._rd_grad_asd(
                    rgb_var, # change to rgb_var
                    normal_var,
                    depth_var,
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    c2w,
                    rgb_as_latents=rgb_as_latents,
                    fovy=fovy,
                    rgb_2nd=rgb_2nd_var, # change to rgb_2nd_var
                    normal_2nd=normal_2nd_var,
                    depth_2nd=depth_2nd_var,
                    **kwargs,
                )
                grad_rgb, grad_rgb_2nd, grad_normal, grad_normal_2nd, grad_depth, grad_depth_2nd = torch_grad(
                    loss_asd.sum(),
                    (rgb_var, rgb_2nd_var, normal_var, normal_2nd_var, depth_var, depth_2nd_var),
                    allow_unused=True,
                )                    
                loss_asd = torch.cat(
                    [
                        SpecifyGradient.apply(normal, grad_normal) + SpecifyGradient.apply(depth, grad_depth),
                        SpecifyGradient.apply(normal_2nd, grad_normal_2nd) + SpecifyGradient.apply(depth_2nd, grad_depth_2nd),
                    ],
                    dim=0
                )
                if grad_rgb is not None and grad_rgb_2nd is not None:
                    loss_asd = loss_asd + torch.cat(
                        [
                            SpecifyGradient.apply(rgb, grad_rgb),
                            SpecifyGradient.apply(rgb_2nd, grad_rgb_2nd),
                        ],
                        dim=0
                    )
                return loss_asd, norm_asd
            else:
                loss_asd, norm_asd = self._rd_grad_asd(
                    rgb_var, # change to rgb_var
                    normal_var,
                    depth_var,
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
                grad_rgb, grad_normal, grad_depth = torch_grad(
                    loss_asd,
                    (rgb_var, normal_var, depth_var),
                    allow_unused=True,
                )
                loss_asd = SpecifyGradient.apply(normal, grad_normal) + SpecifyGradient.apply(depth, grad_depth)
                if grad_rgb is not None:
                    loss_asd = loss_asd + SpecifyGradient.apply(rgb, grad_rgb)
                return loss_asd, norm_asd
        else:
            return self._rd_grad_asd(
                rgb,
                normal,
                depth,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                c2w,
                rgb_as_latents=rgb_as_latents,
                fovy=fovy,
                rgb_2nd=rgb_2nd,
                normal_2nd=normal_2nd,
                depth_2nd=depth_2nd,
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
        is_dual: bool = False,
    ):
        """
            Wrapper for the noise prediction of the stable diffusion model
        """
        img_batch_size = sd_latents.shape[0]
        use_t_plus = self.cfg.sd_plus_ratio > 0 # important for the 2nd diffusion model

        # prepare text embeddings ################################################################################################
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
        
        # prepare noisy input ################################################################################################
        num_repeat = 2
        latents_noisy = self.sd_scheduler.add_noise(sd_latents, sd_noise, t)
        
        latents_model_input = [
            latents_noisy,
        ] * num_repeat

        if use_t_plus:
            latents_noisy_second = self.sd_scheduler.add_noise(sd_latents, sd_noise, t_plus)
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
        ] * num_repeat

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
        noise_pred_text  = noise_pred[0 * img_batch_size: 1 * img_batch_size]
        noise_pred_uncond = noise_pred[1 * img_batch_size: 2 * img_batch_size]
        noise_pred_second = noise_pred[2 * img_batch_size: 3 * img_batch_size] if use_t_plus else noise_pred_text

        return noise_pred_text, noise_pred_uncond, noise_pred_second         

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
        # prepare latents
        sd_latents = self.sd_get_latents(
            rgb_BCHW,
            rgb_BCHW_2nd=rgb_2nd_BCHW if is_dual else None,
            rgb_as_latents=rgb_as_latents,
        )

        # prepare noisy input
        sd_noise = torch.randn_like(sd_latents)

        # prepare text embeddings
        _, text_embeddings = prompt_utils.get_text_embeddings(use_default_neg = True) # sdreamer uses the second text embeddings and the default negative text: ""
        text_batch_size = text_embeddings.shape[0] // 2

        # repeat the text embeddings for the number of views
        text_embeddings_vd = text_embeddings[0 * text_batch_size: 1 * text_batch_size].repeat_interleave(
            view_batch_size // text_batch_size, dim = 0
        )
        text_embeddings_uncond = text_embeddings[1 * text_batch_size: 2 * text_batch_size].repeat_interleave(
            view_batch_size // text_batch_size, dim = 0
        )

        with torch.no_grad():
            min_t, max_t = self.sd_min_step, self.sd_max_step
            t = torch.randint(
                min_t,
                max_t,
                [img_batch_size],
                dtype=torch.long,
                device=self.device,
            )

            # bigger timestamp, the following is specific to ASD
            t_plus = self.get_t_plus(t, module="sd")

            # perform noise prediction
            noise_pred_text, noise_pred_uncond, noise_pred_second = self._sd_noise_pred(
                sd_latents,
                sd_noise,
                t,
                t_plus,
                text_embeddings_vd,
                text_embeddings_uncond,
                text_embeddings_vd,
                is_dual=is_dual,
            )

            noise_pred_first = noise_pred_uncond + self.cfg.sd_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )


        if self.cfg.sd_weighting_strategy in ["sds", "uniform", "fantasia3d", "sds_sqrt"]:
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

            grad = (noise_pred_first - noise_pred_second) * w
        elif self.cfg.sd_weighting_strategy in ["dmd", "dmd+1", "dmd_mv", "dmd_mv+1"]:
            with torch.no_grad():
                alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
                sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
                latent_first = (sd_latents - sigma * noise_pred_first) / alpha
                latent_second = (sd_latents - sigma * noise_pred_second) / alpha
                # no difference between the two strategies for the single view
                if self.cfg.sd_weighting_strategy in ["dmd", "dmd_mv"]:
                    w = torch.abs(sd_latents - latent_first).mean(dim=(1, 2, 3), keepdim=True)
                elif self.cfg.sd_weighting_strategy in ["dmd+1", "dmd_mv+1"]:
                    _noise_pred_second =  noise_pred_uncond + self.cfg.sd_guidance_scale * (
                        noise_pred_second - noise_pred_uncond
                    )
                    _alpha = (self.alphas[t_plus] ** 0.5).view(-1, 1, 1, 1)
                    _sigma = ((1 - self.alphas[t_plus]) ** 0.5).view(-1, 1, 1, 1)
                    _latent_second = (sd_latents - _sigma * _noise_pred_second) / _alpha
                    w = torch.abs(sd_latents - _latent_second).mean(dim=(1, 2, 3), keepdim=True)
                    
                grad = (latent_second - latent_first) / (w + self.cfg.eps)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.sd_weighting_strategy}"
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stability?
        if self.grad_clip_val is not None:
            grad = torch.clamp(grad, -self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (sd_latents - grad).detach()
        if not is_dual:
            loss_asd = 0.5 * F.mse_loss(sd_latents, target, reduction="sum") #/ view_batch_size
            return loss_asd, grad.norm()
        else:
            # split the grad into two parts
            loss_asd = torch.stack(
                [
                    0.5 * F.mse_loss(sd_latents[:view_batch_size], target[:view_batch_size], reduction="sum"), #/ view_batch_size,
                    0.5 * F.mse_loss(sd_latents[view_batch_size:], target[view_batch_size:], reduction="sum"), #/ view_batch_size,
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


    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_distances_relative: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        rgb_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        normal_2nd: Optional[Float[Tensor, "B H W C"]] = None,
        depth_2nd: Optional[Float[Tensor, "B H W C"]] = None,
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
        if self.mv_weight > 0:
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
        # the following is specific to RichDreamer
        if self.cfg.cam_method == "rel_x2":
            camera_distances_input = camera_distances_relative * 2
        elif self.cfg.cam_method == "rel":
            camera_distances_input = camera_distances_relative
        elif self.cfg.cam_method == "abs":
            camera_distances_input = camera_distances
        else:
            raise ValueError(
                f"Unknown camera method: {self.cfg.cam_method}"
            )
        
        if self.rd_weight > 0:
            loss_rd, grad_rd = self.rd_grad_asd(
                rgb,
                normal,
                depth,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances_input, # the trick in the original implementation
                c2w,
                rgb_as_latents=rgb_as_latents,
                fovy=fovy,
                rgb_2nd=rgb_2nd,
                normal_2nd=normal_2nd,
                depth_2nd=depth_2nd,
                **kwargs,
            )
        else:
            loss_rd = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0],
                device=self.device
            )
            grad_rd = torch.tensor(
                0.0 if not is_dual else [0.0, 0.0],
                device=self.device
            )

        ################################################################################################
        # the following is specific to Stable Diffusion
        if self.cfg.sd_hq_only:
            assert self.cfg.sd_oppo_side, "sd_oppo_side must be True if sd_hq_only is True"
            # select only one rendering from n_view renderings
            idx = torch.randint(
                0, self.cfg.n_view,
                (rgb.shape[0] // self.cfg.n_view,), device=self.device, dtype=torch.long
            )
            _idx = torch.arange(0, rgb.shape[0], self.cfg.n_view, device=self.device, dtype=torch.long)
            if is_dual:
                idx_2nd = (idx + self.cfg.n_view // 2) % self.cfg.n_view # the 2nd rendering is the opposite view, we assume n_view is even
                idx = torch.cat([idx, idx_2nd], dim=0)
            idx += _idx.repeat_interleave(2 if is_dual else 1, dim=0)
        else: # only one rendering
            if self.cfg.sd_oppo_side:
                idx = torch.randint(
                    0, self.cfg.n_view,
                    (rgb.shape[0] // self.cfg.n_view,), device=self.device, dtype=torch.long
                )
                idx_2nd = (idx + self.cfg.n_view // 2) % self.cfg.n_view # the 2nd rendering is the opposite view, we assume n_view is even
                idx += torch.arange(0, rgb.shape[0], self.cfg.n_view, device=self.device, dtype=torch.long)
                idx_2nd += torch.arange(0, rgb.shape[0], self.cfg.n_view, device=self.device, dtype=torch.long)
            else:
                idx = torch.randint(
                    0, self.cfg.n_view,
                    (rgb.shape[0] // self.cfg.n_view,), device=self.device, dtype=torch.long
                )
                idx += torch.arange(0, rgb.shape[0], self.cfg.n_view, device=self.device, dtype=torch.long)
                if is_dual:
                    idx_2nd = torch.randint(
                        0, self.cfg.n_view,
                        (rgb_2nd.shape[0] // self.cfg.n_view,), device=self.device, dtype=torch.long
                    )
                    idx_2nd += torch.arange(0, rgb_2nd.shape[0], self.cfg.n_view, device=self.device, dtype=torch.long)
                    
        if self.sd_weight > 0:
            loss_sd, grad_sd = self.sd_grad_asd(
                rgb[idx],
                prompt_utils,
                elevation[idx],
                azimuth[idx],
                camera_distances[idx],
                c2w[idx],
                rgb_as_latents=rgb_as_latents,
                fovy=fovy,
                rgb_2nd=rgb_2nd[idx_2nd] if is_dual and not self.cfg.sd_hq_only else None,
                azimuth_2nd=azimuth[idx_2nd] if is_dual and not self.cfg.sd_hq_only else None,
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
                "loss_asd": self.rd_weight * loss_rd + self.mv_weight * loss_mv + self.sd_weight * loss_sd,
                "grad_norm_asd": self.rd_weight * grad_rd + self.mv_weight * grad_mv + self.sd_weight * grad_sd,
            }
        else:
            # return the loss and grad_norm for the 1st renderings
            loss = 0
            grad_norm = 0

            loss += self.rd_weight * loss_rd[0]
            grad_norm += self.rd_weight * grad_rd[0]
            loss += self.mv_weight * loss_mv[0]
            grad_norm += self.mv_weight * loss_mv[0]
            loss += self.sd_weight * loss_sd[0]
            grad_norm += self.sd_weight * loss_sd[0]

            guidance_1st =  {
                "loss_asd": loss,
                "grad_norm_asd": grad_norm,
            }

            # return the loss and grad_norm for the 2nd renderings
            loss = 0
            grad_norm = 0

            loss += self.rd_weight * loss_rd[1]
            grad_norm += self.rd_weight * grad_rd[1]
            loss += self.mv_weight * loss_mv[1]
            grad_norm += self.mv_weight * grad_mv[1]
            if not self.cfg.sd_hq_only: # otherwise, sd's grad for the 2nd rendering is not computed
                loss += self.sd_weight * loss_sd[1]
                grad_norm += self.sd_weight * loss_sd[1]
                                               
            guidance_2nd =  {
                "loss_asd": loss,
                "grad_norm_asd": grad_norm,

            }
            return guidance_1st, guidance_2nd


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)
        

        # update the weights and min/max step for each module ################################################################################################
        max_step = int(self.num_train_timesteps * C(self.cfg.max_step_percent, epoch, global_step))
        min_step = int(self.num_train_timesteps * C(self.cfg.min_step_percent, epoch, global_step))

        # stable diffusion
        self.sd_weight = C(self.cfg.sd_weight, epoch, global_step)
        self.sd_min_step = int(self.num_train_timesteps * C(self.cfg.sd_min_step_percent, epoch, global_step)) \
            if self.cfg.sd_min_step_percent is not None else min_step
        self.sd_max_step = int(self.num_train_timesteps * C(self.cfg.sd_max_step_percent, epoch, global_step)) \
            if self.cfg.sd_max_step_percent is not None else max_step
        # rich dreamer
        self.rd_weight = C(self.cfg.rd_weight, epoch, global_step)
        self.rd_min_step = int(self.num_train_timesteps * C(self.cfg.rd_min_step_percent, epoch, global_step)) \
            if self.cfg.rd_min_step_percent is not None else min_step
        self.rd_max_step = int(self.num_train_timesteps * C(self.cfg.rd_max_step_percent, epoch, global_step)) \
            if self.cfg.rd_max_step_percent is not None else max_step
        # mvdream
        self.mv_weight = C(self.cfg.mv_weight, epoch, global_step)
        self.mv_min_step = int(self.num_train_timesteps * C(self.cfg.mv_min_step_percent, epoch, global_step)) \
            if self.cfg.mv_min_step_percent is not None else min_step
        self.mv_max_step = int(self.num_train_timesteps * C(self.cfg.mv_max_step_percent, epoch, global_step)) \
            if self.cfg.mv_max_step_percent is not None else max_step

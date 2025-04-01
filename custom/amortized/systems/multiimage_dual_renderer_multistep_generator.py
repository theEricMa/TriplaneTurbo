import os
import shutil
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_rank, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from functools import partial

from tqdm import tqdm
from threestudio.utils.misc import barrier
from threestudio.models.mesh import Mesh

from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

from functools import partial


def sample_timesteps(
    all_timesteps: List,
    num_parts: int,
    batch_size: int = 1,
):
    # separate the timestep into num_parts_training parts
    timesteps = []

    for i in range(num_parts):
        length_timestep = len(all_timesteps) // num_parts
        timestep = all_timesteps[
            i * length_timestep : (i + 1) * length_timestep
        ]
        # sample only one from the timestep
        idx = torch.randint(0, len(timestep), (batch_size,))
        timesteps.append(timestep[idx])

    return timesteps

@threestudio.register("multiimage-dual-renderer-multistep-generator-system")
class MultiimageDualRendererMultiStepGeneratorSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):

        # validation related
        visualize_samples: bool = False

        # renderering related
        rgb_as_latents: bool = False

        # initialization related
        initialize_shape: bool = True

        # if the guidance requires training
        train_guidance: bool = False

        # added another renderer
        renderer_2nd_type: str = ""
        renderer_2nd: dict = field(default_factory=dict)

        # parallelly compute the guidance
        parallel_guidance: bool = False

        # scheduler path
        scheduler_dir: str = "pretrained/stable-diffusion-2-1-base"

        # the followings are related to the multi-step diffusion
        num_parts_training: int = 4

        num_steps_training: int = 50
        num_steps_sampling: int = 50

        timesteps_from_T: bool = True

        
        sample_scheduler: str = "ddpm" #any of "ddpm", "ddim"
        noise_scheduler: str = "ddim"

        specifiy_guidance_timestep: Optional[str] = None # any of None, v1;  control the guidance timestep

        gradient_accumulation_steps: int = 1

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        # set up the second renderer
        self.renderer_2nd = threestudio.find(self.cfg.renderer_2nd_type)(
            self.cfg.renderer_2nd,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

        if self.cfg.train_guidance: # if the guidance requires training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # Sampler for training
        self.noise_scheduler = self._configure_scheduler(self.cfg.noise_scheduler)

        # Sampler for inference
        self.sample_scheduler = self._configure_scheduler(self.cfg.sample_scheduler)

        # This property activates manual optimization.
        self.automatic_optimization = False 


    def _configure_scheduler(self, scheduler: str):
        assert scheduler in ["ddpm", "ddim", "dpm"]


        if scheduler == "ddpm":
            scheduler_returned = DDPMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        elif scheduler == "ddim":
            scheduler_returned = DDIMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        elif scheduler == "dpm":
            scheduler_returned = DPMSolverMultistepScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        return scheduler_returned


    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training

        if not self.cfg.train_guidance: # if the guidance does not require training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # initialize SDF
        if self.cfg.initialize_shape:
            # info
            if get_device() == "cuda_0": # only report from one process
                threestudio.info("Initializing shape...")
            
            # check if attribute exists
            if not hasattr(self.geometry, "initialize_shape"):
                threestudio.info("Geometry does not have initialize_shape method. skip.")
            else:
                self.geometry.initialize_shape()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # for gradient accumulation
        opt = self.optimizers()
        opt.zero_grad()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        # for gradient accumulation
        # update the weights with the remaining gradients
        opt = self.optimizers()
        try:
            opt.step()
            opt.zero_grad()
        except:
            pass

    # def on_train_batch_start(self, batch, batch_idx, unused=0):
    #     return super().on_train_batch_start(batch, batch_idx, unused)

    def forward_rendering(
        self,
        batch: Dict[str, Any],
    ):

        render_out = self.renderer(**batch, )
        render_out_2nd = self.renderer_2nd(**batch, )

        # decode the rgb as latents only in testing and validation
        if self.cfg.rgb_as_latents and not self.training: 
            # get the rgb
            if "comp_rgb" not in render_out:
                raise ValueError(
                    "comp_rgb is required for rgb_as_latents, no comp_rgb is found in the output."
                )
            else:
                out_image = render_out["comp_rgb"]
                out_image = self.guidance.decode_latents(
                    out_image.permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1) 
                render_out['decoded_rgb'] = out_image

                out_image_2nd = render_out_2nd["comp_rgb"]
                out_image_2nd = self.guidance.decode_latents(
                    out_image_2nd.permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1)
                render_out_2nd['decoded_rgb'] = out_image_2nd

        return {
            **render_out,
        }, {
            **render_out_2nd,
        }
        
    def compute_guidance_n_loss(
        self,
        out: Dict[str, Any],
        out_2nd: Dict[str, Any],
        idx: int,
        **batch,
    ) -> Dict[str, Any]:
        # guidance for the first renderer
        guidance_rgb = out["comp_rgb"]

        # specify the timestep range for the guidance
        if self.cfg.specifiy_guidance_timestep in [None]:
            timestep_range = None
        elif self.cfg.specifiy_guidance_timestep in ["v1"]:
            timestep_range = [
                (self.cfg.num_parts_training - idx - 1) / self.cfg.num_parts_training, # min
                (self.cfg.num_parts_training - idx) / self.cfg.num_parts_training # max
            ]
        elif self.cfg.specifiy_guidance_timestep in ["v2"]:
            timestep_range = [
                0, # min
                (self.cfg.num_parts_training - idx) / self.cfg.num_parts_training # max
            ]
        else:
            raise NotImplementedError

        # guidance for the second renderer
        guidance_rgb_2nd = out_2nd["comp_rgb"]

        # collect the guidance
        if "prompt_utils" not in batch:
            batch["prompt_utils"] = batch["guidance_utils"]

        if not self.cfg.parallel_guidance:
            # the guidance is computed in two steps
            guidance_out = self.guidance(
                guidance_rgb, 
                normal=out["comp_normal_cam_vis_white"] if "comp_normal_cam_vis_white" in out else None,
                depth=out["disparity"] if "disparity" in out else None,
                **batch, 
                rgb_as_latents=self.cfg.rgb_as_latents,
                timestep_range=timestep_range,
            )

            guidance_out_2nd = self.guidance(
                guidance_rgb_2nd, 
                normal=out_2nd["comp_normal_cam_vis_white"] if "comp_normal_cam_vis_white" in out_2nd else None,
                depth=out_2nd["disparity"] if "disparity" in out_2nd else None,
                **batch, 
                rgb_as_latents=self.cfg.rgb_as_latents,
                timestep_range=timestep_range,
            )
        else:
            # the guidance is computed in parallel
            guidance_out, guidance_out_2nd = self.guidance(
                guidance_rgb,
                normal=out["comp_normal_cam_vis_white"] if "comp_normal_cam_vis_white" in out else None,
                depth=out["disparity"] if "disparity" in out else None,
                **batch,
                rgb_as_latents=self.cfg.rgb_as_latents,
                rgb_2nd = guidance_rgb_2nd,
                normal_2nd = out_2nd["comp_normal_cam_vis_white"] if "comp_normal_cam_vis_white" in out_2nd else None,
                depth_2nd = out_2nd["disparity"] if "disparity" in out_2nd else None,
                timestep_range=timestep_range,
            )
        loss_dict = self._compute_loss(guidance_out, out, renderer="1st", step = idx, has_grad = guidance_rgb.requires_grad, **batch)
        loss_dict_2nd = self._compute_loss(guidance_out_2nd, out_2nd, renderer="2nd", step = idx, has_grad = guidance_rgb_2nd.requires_grad, **batch)

        return {
            "fidelity_loss": loss_dict["fidelity_loss"] + loss_dict_2nd["fidelity_loss"],
            "regularization_loss": loss_dict["regularization_loss"] + loss_dict_2nd["regularization_loss"],            
        }

    def _set_timesteps(
        self,
        scheduler,
        num_steps: int,
    ):
        scheduler.set_timesteps(num_steps)
        timesteps_orig = scheduler.timesteps
        if self.cfg.timesteps_from_T:
            timesteps_delta = scheduler.config.num_train_timesteps - 1 - timesteps_orig.max() 
            timesteps = timesteps_orig + timesteps_delta
            return timesteps
        else:
            return timesteps_orig


    def diffusion_reverse(
        self,
        batch: Dict[str, Any],
    ):


        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
           raise NotImplementedError
        else:
            # more general case
            cond_dict = prompt_utils.get_image_encodings()
        
        timesteps = self._set_timesteps(
            self.sample_scheduler,
            self.cfg.num_steps_sampling,
        )

        latents = batch.pop("noise")

        for i, t in enumerate(timesteps):

            # prepare inputs
            noisy_latent_input = self.sample_scheduler.scale_model_input(
                latents, 
                t
            )

            # predict the noise added
            pred = self.geometry.denoise(
                noisy_input = noisy_latent_input,
                timestep = t.to(self.device),
                **cond_dict,
            )

            latents_denoised = self.sample_scheduler.step(pred, t, latents).pred_original_sample
            latents = self.sample_scheduler.step(pred, t, latents).prev_sample


        # decode the latent to 3D representation
        space_cache = self.geometry.decode(
            latents = latents_denoised,
        )

        return space_cache

    def training_step(
        self,
        batch_list: List[Dict[str, Any]],
        batch_idx
    ):
        """
            Diffusion Forward Process
            but supervised by the 2D guidance
        """
        latent = batch_list[0]["noise"]
        # batch_size = batch_list[0]["prompt_utils"].get_global_text_embeddings().shape[0] # sort of hacky

        all_timesteps = self._set_timesteps(
            self.noise_scheduler,
            self.cfg.num_steps_training,
        )

        timesteps = sample_timesteps(
            all_timesteps,
            num_parts = self.cfg.num_parts_training, 
            batch_size=1, #batch_size,
        )

        # zero the gradients
        opt = self.optimizers()

        # load the coefficients to the GPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
    
        for i, (t, batch) in enumerate(zip(timesteps, batch_list)):

            # prepare the text embeddings as input
            prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
            if "prompt_target" in batch:
                raise NotImplementedError
            else:
                # more general case
                cond_dict = prompt_utils.get_image_encodings()
                batch["text_embed"] = cond_dict["text_embeddings_local"]

            # choose the noise to be added
            noise = torch.randn_like(latent)

            # add noise to the latent
            noisy_latent = self.noise_scheduler.add_noise(
                latent,
                noise,
                t,
            )

            noisy_latent = self.noise_scheduler.scale_model_input(
                noisy_latent,
                t
            )

            # predict the noise added
            noise_pred = self.geometry.denoise(
                noisy_input = noisy_latent,
                timestep = t.to(self.device),
                **cond_dict
            )

            #convert epsilon into x0
            denoised_latents = self.noise_scheduler.step(
                noise_pred, 
                t.to(self.device),
                noisy_latent).pred_original_sample
            # denoised_latents = torch.stack([
            #     self.noise_scheduler.step(_pred, t, _latent).pred_original_sample
            #     for _pred, t, _latent in zip(noise_pred, noisy_latent)
            # ])
            
            #self.noise_scheduler.step(noise_pred, t, latent).pred_original_sample

            # decode the latent to 3D representation
            batch["space_cache"] = self.geometry.decode(
                latents = denoised_latents,
            )

            # render the image and compute the gradients
            out, out_2nd = self.forward_rendering(batch)
            loss_dict = self.compute_guidance_n_loss(
                out, out_2nd, idx = i, **batch
            )
            fidelity_loss = loss_dict["fidelity_loss"]
            regularization_loss = loss_dict["regularization_loss"]

            weight_fide = 1.0 / self.cfg.num_parts_training
            weight_regu = 1.0 / self.cfg.num_parts_training

            # store gradients
            loss = weight_fide * fidelity_loss + weight_regu * regularization_loss
            self.manual_backward(loss / self.cfg.gradient_accumulation_steps)

            # prepare for the next iteration
            latent = denoised_latents.detach()
            
        # update the weights
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()

    def validation_step(self, batch, batch_idx):

        # prepare the text embeddings as input
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            cond_dict = prompt_utils.get_image_encodings()
            batch["text_embed"] = cond_dict["text_embeddings_local"]

        batch["space_cache"]  = self.diffusion_reverse(batch)
        out, out_2nd = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]

        for batch_idx in tqdm(range(batch_size), desc="Saving val images"):
            self._save_image_grid(batch, batch_idx, out, phase="val", render="1st")
            self._save_image_grid(batch, batch_idx, out_2nd, phase="val", render="2nd")
                
        if self.cfg.visualize_samples and hasattr(self.guidance, "sample"):
            if "prompt_utils" not in batch:
                batch["prompt_utils"] = batch["guidance_utils"]
            
            image_list = self.guidance.sample(
                **batch,
                seed=self.global_step
            )

            # save the image with the same name as the image_path
            phase = "val"
            if "name" in batch:
                name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
            else:
                name = batch['image_path'][0].replace(',', '').replace('.', '').replace(' ', '_')
            # specify the image name
            image_name  = f"it{self.true_global_step}-{phase}-sample/{name}.png"


            self.save_image_grid(
                image_name,
                image_list,
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx, return_space_cache = False, render_images = True):

        # prepare the text embeddings as input
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            cond_dict = prompt_utils.get_image_encodings()
            batch["text_embed"] = cond_dict["text_embeddings_local"]

        batch["space_cache"] = self.diffusion_reverse(batch)

        if render_images:
            out, out_2nd = self.forward_rendering(batch)
            batch_size = out['comp_rgb'].shape[0]

            for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
                self._save_image_grid(batch, batch_idx, out, phase="test", render="1st")
                self._save_image_grid(batch, batch_idx, out_2nd, phase="test", render="2nd")

        if return_space_cache:
            return batch["space_cache"]


    def _compute_loss(
        self,
        guidance_out: Dict[str, Any],
        out: Dict[str, Any],
        renderer: str = "1st",
        step: int = 0,
        has_grad: bool = True,
        **batch,
    ):
        
        assert renderer in ["1st", "2nd"]

        fide_loss = 0.0
        regu_loss = 0.0
        for name, value in guidance_out.items():
            if renderer == "1st":
                self.log(f"train/{name}_{step}", value)
                if name.startswith("loss_"):
                    fide_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            else:
                self.log(f"train/{name}_2nd_{step}", value)
                if name.startswith("loss_"):
                    fide_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_") + "_2nd"])

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_orient) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_orient_2nd) > 0):
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            if renderer == "1st":
                self.log(f"train/loss_orient_{step}", loss_orient)
                regu_loss += loss_orient * self.C(self.cfg.loss.lambda_orient)
            else:
                self.log(f"train/loss_orient_2nd_{step}", loss_orient)
                regu_loss += loss_orient * self.C(self.cfg.loss.lambda_orient_2nd)

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_sparsity) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_sparsity_2nd) > 0):
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            if renderer == "1st":
                self.log(f"train/loss_sparsity_{step}", loss_sparsity, prog_bar=False if step % 4 != 3 else True)
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            else:
                self.log(f"train/loss_sparsity_2nd_{step}", loss_sparsity, prog_bar=False if step % 4 != 3 else True)
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity_2nd)


        if (renderer == "1st" and self.C(self.cfg.loss.lambda_opaque) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_opaque_2nd) > 0):
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            if renderer == "1st":
                self.log(f"train/loss_opaque_{step}", loss_opaque)
                regu_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
            else:
                self.log(f"train/loss_opaque_2nd_{step}", loss_opaque)
                regu_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque_2nd)

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_z_variance) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_z_variance_2nd) > 0):
            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if 'z_variance' not in out:
                raise ValueError(
                    "z_variance is required for z_variance loss, no z_variance is found in the output."
                )
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            if renderer == "1st":
                self.log(f"train/loss_z_variance_{step}", loss_z_variance)
                regu_loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            else:
                self.log(f"train/loss_z_variance_2nd_{step}", loss_z_variance)
                regu_loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance_2nd)

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_sdf_abs) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_sdf_abs_2nd) > 0):
            if 'sdf' not in out:
                raise ValueError(
                    "sdf is required for sdf_abs loss, no sdf is found in the output."
                )
            if isinstance(out["sdf"], torch.Tensor):
                loss_sdf_abs = out["sdf"].abs().mean()
            else:
                loss_sdf_abs = 0
                for sdf in out["sdf"]:
                    loss_sdf_abs += sdf.abs().mean()
                loss_sdf_abs /= len(out["sdf"])

            if renderer == "1st":
                self.log(f"train/loss_sdf_abs_{step}", loss_sdf_abs)
                regu_loss += loss_sdf_abs * self.C(self.cfg.loss.lambda_sdf_abs)
            else:
                self.log(f"train/loss_sdf_abs_2nd_{step}", loss_sdf_abs)
                regu_loss += loss_sdf_abs * self.C(self.cfg.loss.lambda_sdf_abs_2nd)

        # sdf eikonal loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_eikonal) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_eikonal_2nd) > 0):
            if 'sdf_grad' not in out:
                raise ValueError(
                    "sdf is required for eikonal loss, no sdf is found in the output."
                )
            
            if isinstance(out["sdf_grad"], torch.Tensor):
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
            else:
                loss_eikonal = 0
                for sdf_grad in out["sdf_grad"]:
                    loss_eikonal += (
                        (torch.linalg.norm(sdf_grad, ord=2, dim=-1) - 1.0) ** 2
                    ).mean()
                loss_eikonal /= len(out["sdf_grad"])

            
            if renderer == "1st":
                self.log(f"train/loss_eikonal_{step}", loss_eikonal)
                regu_loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
            else:
                self.log(f"train/loss_eikonal_2nd_{step}", loss_eikonal)
                regu_loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal_2nd)

        # normal consistency loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_normal_consistency) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_normal_consistency_2nd) > 0):
            if 'mesh' in out:
                if not isinstance(out["mesh"], list):
                    out["mesh"] = [out["mesh"]]
                loss_normal_consistency = 0.0
                for mesh in out["mesh"]:
                    assert isinstance(mesh, Mesh), "mesh should be an instance of Mesh"
                    loss_normal_consistency += mesh.normal_consistency()
            else:
                raise ValueError(
                    "mesh is required for normal consistency loss, no mesh is found in the output."
                )

            if renderer == "1st":
                self.log(f"train/loss_normal_consistency_{step}", loss_normal_consistency)
                regu_loss += loss_normal_consistency * self.C(self.cfg.loss.lambda_normal_consistency)
            else:
                self.log(f"train/loss_normal_consistency_2nd_{step}", loss_normal_consistency)
                regu_loss += loss_normal_consistency * self.C(self.cfg.loss.lambda_normal_consistency_2nd)
        
        # laplacian loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_laplacian_smoothness_2nd) > 0):
            if 'mesh' in out:
                if not isinstance(out["mesh"], list):
                    out["mesh"] = [out["mesh"]]
                loss_laplacian = 0.0
                for mesh in out["mesh"]:
                    assert isinstance(mesh, Mesh), "mesh should be an instance of Mesh"
                    loss_laplacian += mesh.laplacian()

            else:
                raise ValueError(
                    "mesh is required for laplacian loss, no mesh is found in the output."
                )
            
            if renderer == "1st":
                self.log(f"train/loss_laplacian_smoothness_{step}", loss_laplacian)
                regu_loss += loss_laplacian * self.C(self.cfg.loss.lambda_laplacian_smoothness)
            else:
                self.log(f"train/loss_laplacian_smoothness_2nd_{step}", loss_laplacian)
                regu_loss += loss_laplacian * self.C(self.cfg.loss.lambda_laplacian_smoothness_2nd)
            
        # lambda_normal_smoothness_2d
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_normal_smoothness_2d_2nd) > 0):
            normal = out["comp_normal"]
            loss_normal_smoothness_2d = (
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean() +
                (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
            )
            if renderer == "1st":
                self.log(f"train/loss_normal_smoothness_2d_{step}", loss_normal_smoothness_2d)
                regu_loss += loss_normal_smoothness_2d * self.C(self.cfg.loss.lambda_normal_smoothness_2d)
            else:
                self.log(f"train/loss_normal_smoothness_2d_2nd_{step}", loss_normal_smoothness_2d)
                regu_loss += loss_normal_smoothness_2d * self.C(self.cfg.loss.lambda_normal_smoothness_2d_2nd)

        if "inv_std" in out:
            self.log("train/inv_std", out["inv_std"], prog_bar=True)

        # detach the loss if necessary
        if not has_grad:
            if hasattr(fide_loss, "requires_grad") and fide_loss.requires_grad:
                fide_loss = fide_loss.detach()
                
        if not has_grad:
            if hasattr(regu_loss, "requires_grad") and regu_loss.requires_grad:
                regu_loss = regu_loss.detach()

        return {
            "fidelity_loss": fide_loss,
            "regularization_loss": regu_loss,
        }


    def _save_image_grid(
        self, 
        batch,
        batch_idx,
        out,
        phase="val",
        render="1st",
    ):
        
        assert phase in ["val", "test"]

        # save the image with the same name as the image_path
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['image_path'][0].replace(',', '').replace('.', '').replace(' ', '_')
        # specify the image name
        image_name  = f"it{self.true_global_step}-{phase}-{render}/{name}/{str(batch['index'][batch_idx].item())}.png"
        # specify the verbose name
        verbose_name = f"{phase}_{render}_step"

        # normalize the depth
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())

        self.save_image_grid(
            image_name,
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][batch_idx] if not self.cfg.rgb_as_latents else out["decoded_rgb"][batch_idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_cam_vis_white"][batch_idx] if "comp_normal_cam_vis_white" in out else out["comp_normal"][batch_idx],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][batch_idx, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out['disparity'][batch_idx, :, :, 0] if 'disparity' in out else normalize(out["depth"][batch_idx, :, :, 0]),
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "depth" in out
                else []
            ),
            name=verbose_name,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        barrier() # wait until all GPUs finish rendering images
        filestems = [
            f"it{self.true_global_step}-val-{render}"
            for render in ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                files = os.listdir(os.path.join(self.get_save_dir(), filestem))
                files = [f for f in files if os.path.isdir(os.path.join(self.get_save_dir(), filestem, f))]
                for prompt in tqdm(
                    files,
                    desc="Generating validation videos",
                ):
                    try:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=10,
                            name="validation_epoch_end",
                            step=self.true_global_step,
                            multithreaded=True,
                        )
                    except:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=10,
                            name="validation_epoch_end",
                            step=self.true_global_step,
                            # multithreaded=True,
                        )

    def on_test_epoch_end(self):
        barrier() # wait until all GPUs finish rendering images
        filestems = [
            f"it{self.true_global_step}-test-{render}"
            for render in ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                files = os.listdir(os.path.join(self.get_save_dir(), filestem))
                files = [f for f in files if os.path.isdir(os.path.join(self.get_save_dir(), filestem, f))]
                for prompt in tqdm(
                    files,
                    desc="Generating validation videos",
                ):
                    try:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=30,
                            name="test",
                            step=self.true_global_step,
                            multithreaded=True,
                        )
                    except:
                        self.save_img_sequence(
                            os.path.join(filestem, prompt),
                            os.path.join(filestem, prompt),
                            "(\d+)\.png",
                            save_format="mp4",
                            fps=10,
                            name="validation_epoch_end",
                            step=self.true_global_step,
                            # multithreaded=True,
                        )


    def on_predict_start(self) -> None:
        self.exporter: Exporter = threestudio.find(self.cfg.exporter_type)(
            self.cfg.exporter,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

    def predict_step(self, batch, batch_idx):
        space_cache = self.test_step(batch, batch_idx, render_images=self.exporter.cfg.save_video, return_space_cache=True)
        # update the space_cache into the exporter
        exporter_output: List[ExporterOutput] = self.exporter(space_cache)

        # specify the name
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['image_path'][0].replace(',', '').replace('.', '').replace(' ', '_')

        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{name}/{out.save_name}", **out.params)

    def on_predict_epoch_end(self) -> None:
        if self.exporter.cfg.save_video:
            self.on_test_epoch_end()

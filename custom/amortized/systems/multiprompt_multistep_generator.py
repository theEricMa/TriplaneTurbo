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

@threestudio.register("multiprompt-multistep-generator-system")
class MultipromptMultiStepGeneratorSystem(BaseLift3DSystem):
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

        # parallelly compute the guidance
        parallel_guidance: bool = False

        # scheduler path
        scheduler_dir: str = "pretrained/stable-diffusion-2-1-base"

        # the followings are related to the multi-step diffusion
        noise_addition: str = "gaussian" # any of "gaussian", "zero", "pred"
        num_parts_training: int = 4

        num_steps_training: int = 50
        num_steps_sampling: int = 50

        
        sample_scheduler: str = "ddpm" #any of "ddpm", "ddim"
        noise_scheduler: str = "ddim"

        specifiy_guidance_timestep: Optional[str] = None # any of None, v1;  control the guidance timestep

        predition_type: str = "epsilon" # any of "epsilon", "sample"

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        if self.cfg.train_guidance: # if the guidance requires training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # Sampler for training
        self.noise_scheduler = self._configure_scheduler(self.cfg.noise_scheduler)
        self.is_training_odd = True if self.cfg.noise_scheduler == "ddpm" else False

        # Sampler for inference
        self.sample_scheduler = self._configure_scheduler(self.cfg.sample_scheduler)

        # This property activates manual optimization.
        self.automatic_optimization = False 


    def _configure_scheduler(self, scheduler: str):
        assert scheduler in ["ddpm", "ddim", "dpm"]
        assert self.cfg.predition_type in [
            "epsilon", "sample", "v_prediction",
            "sample_delta", "sample_delta_v2", "sample_delta_v3"
        ]
        
        # define the prediction type
        predition_type = self.cfg.predition_type
        if "sample" in predition_type: # special case for variance such as sample_delta
            predition_type = "sample"

        if scheduler == "ddpm":
            scheduler_returned = DDPMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
                prediction_type=predition_type,
            )
        elif scheduler == "ddim":
            scheduler_returned = DDIMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
                prediction_type=predition_type,
            )
        elif scheduler == "dpm":
            scheduler_returned = DPMSolverMultistepScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
                prediction_type=predition_type,
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

    def forward_rendering(
        self,
        batch: Dict[str, Any],
    ):

        render_out = self.renderer(**batch, )

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

        return render_out

    def compute_guidance_n_loss(
        self,
        out: Dict[str, Any],
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


        # collect the guidance
        if "prompt_utils" not in batch:
            batch["prompt_utils"] = batch["guidance_utils"]

            # the guidance is computed in parallel
            guidance_out = self.guidance(
                guidance_rgb,
                normal=out["comp_normal_cam_vis"] if "comp_normal_cam_vis" in out else None,
                depth=out["disparity"] if "disparity" in out else None,
                **batch,
                rgb_as_latents=self.cfg.rgb_as_latents,
                timestep_range=timestep_range,
            )
        loss_dict = self._compute_loss(guidance_out, out, renderer="1st", step = idx, **batch)
        return loss_dict

    def _set_timesteps(
        self,
        scheduler,
        num_steps: int,
    ):
        scheduler.set_timesteps(num_steps)
        timesteps_orig = scheduler.timesteps
        timesteps_delta = scheduler.config.num_train_timesteps - 1 - timesteps_orig.max() 
        timesteps = timesteps_orig + timesteps_delta
        return timesteps


    def diffusion_reverse(
        self,
        batch: Dict[str, Any],
    ):


        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
           raise NotImplementedError
        else:
            # more general case
            text_embed_cond = prompt_utils.get_global_text_embeddings()
            text_embed_uncond = prompt_utils.get_uncond_text_embeddings()
        
        if None:
            text_embed = torch.cat(
                [
                    text_embed_cond,
                    text_embed_uncond,
                ],
                dim=0,
            )
        else:
            text_embed = text_embed_cond

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

            if None:
                noisy_latent_input = torch.cat([noisy_latent_input] * 2, dim=0)

            # necessary coefficients
            alphas: Float[Tensor, "..."] = self.sample_scheduler.alphas_cumprod.to(
                self.device
            )
            alpha = (alphas[t] ** 0.5).view(-1, 1, 1, 1).to(self.device)
            sigma = ((1 - alphas[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)

            if self.cfg.predition_type in ["epsilon", "v_prediction"]:
                # predict the noise added
                pred = self.geometry.denoise(
                    noisy_input = noisy_latent_input,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )
                latents = self.sample_scheduler.step(pred, t, latents).prev_sample
            elif self.cfg.predition_type in ["sample", "sample_delta", "sample_delta_v2", "sample_delta_v3"]:
                output = self.geometry.denoise(
                    noisy_input = noisy_latent_input,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )
                if self.cfg.predition_type in ["sample"]:
                    denoised_latents = output
                elif self.cfg.predition_type in ["sample_delta"]:
                    denoised_latents = noisy_latent_input + output
                elif self.cfg.predition_type in ["sample_delta_v2"]:
                    denoised_latents = noisy_latent_input - output
                elif self.cfg.predition_type in ["sample_delta_v3"]:
                    denoised_latents = noisy_latent_input / alpha - output

                latents = self.sample_scheduler.step(denoised_latents, t, latents).prev_sample
            else:
                raise NotImplementedError

        # decode the latent to 3D representation
        space_cache = self.geometry.decode(
            latents = latents,
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
        opt.zero_grad()
    
        for i, (t, batch) in enumerate(zip(timesteps, batch_list)):

            # prepare the text embeddings as input
            prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
            if "prompt_target" in batch:
                raise NotImplementedError
            else:
                # more general case
                cond = prompt_utils.get_global_text_embeddings()
                uncond = prompt_utils.get_uncond_text_embeddings()
                batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
                batch["text_embed"] = cond

            # choose the noise to be added
            if self.cfg.noise_addition == "gaussian":
                noise = torch.randn_like(latent)
            elif self.cfg.noise_addition == "zero":
                noise = torch.zeros_like(latent)
            elif self.cfg.noise_addition == "pred": # use the network to predict the noise
                noise = noise_pred.detach() if i > 0 else torch.randn_like(latent)

            # add noise to the latent
            noisy_latent = self.noise_scheduler.add_noise(
                latent,
                noise,
                t,
            )
 

            # prepare the text embeddings as input
            text_embed = cond
            # if torch.rand(1) < 0: 
            #     text_embed = uncond

            # necessary coefficients
            alphas: Float[Tensor, "..."] = self.noise_scheduler.alphas_cumprod.to(
                self.device
            )
            alpha = (alphas[t] ** 0.5).view(-1, 1, 1, 1).to(self.device)
            sigma = ((1 - alphas[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)


            # predict the noise added
            if self.cfg.predition_type in ["epsilon"]:
                noise_pred = self.geometry.denoise(
                    noisy_input = noisy_latent,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )

                # convert epsilon into x0
                denoised_latents = (noisy_latent - sigma * noise_pred) / alpha
            elif self.cfg.predition_type in ["v_prediction"]:
                v_pred = self.geometry.denoise(
                    noisy_input = noisy_latent,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )
                denoised_latents = alpha * noisy_latent - sigma * v_pred
            elif self.cfg.predition_type in ["sample", "sample_delta", "sample_delta_v2", "sample_delta_v3"]:
                output = self.geometry.denoise(
                    noisy_input = noisy_latent,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )
                if self.cfg.predition_type in ["sample"]:
                    denoised_latents = output
                elif self.cfg.predition_type in ["sample_delta"]:
                    denoised_latents = noisy_latent + output
                elif self.cfg.predition_type in ["sample_delta_v2"]:
                    denoised_latents = noisy_latent - output
                elif self.cfg.predition_type in ["sample_delta_v3"]:
                    denoised_latents = noisy_latent / alpha - output
            else:
                raise NotImplementedError

            # decode the latent to 3D representation
            batch["space_cache"] = self.geometry.decode(
                latents = denoised_latents,
            )

            # render the image and compute the gradients
            out = self.forward_rendering(batch)
            loss_dict = self.compute_guidance_n_loss(
                out, idx = i, **batch
            )
            fidelity_loss = loss_dict["fidelity_loss"]
            regularization_loss = loss_dict["regularization_loss"]

            if hasattr(self.cfg.loss, "weighting_strategy"):
                if self.cfg.loss.weighting_strategy in ["v1"]:
                    weight_fide = 1.0 / self.cfg.num_parts_training
                    weight_regu = 1.0 / self.cfg.num_parts_training
                elif self.cfg.loss.weighting_strategy in ["v2"]:
                    weight_fide = (alpha / sigma).mean() # mean is for converting the batch to a scalar
                    weight_regu = (alpha / sigma).mean() 
                elif self.cfg.loss.weighting_strategy in ["v2-2"]:
                    weight_fide = 1.0 / self.cfg.num_parts_training
                    weight_regu = (alpha / sigma).mean()
                elif self.cfg.loss.weighting_strategy in ["v3"]:
                    # follow SDS
                    weight_fide = (sigma**2).mean()
                    weight_regu = (sigma**2).mean() 
                elif self.cfg.loss.weighting_strategy in ["v3-2"]:
                    weight_fide = 1.0 / self.cfg.num_parts_training
                    # follow SDS
                    weight_regu = (sigma**2).mean()
                elif self.cfg.loss.weighting_strategy in ["v4"]:
                    weight_fide = (sigma / alpha).mean()
                    weight_regu = (sigma / alpha).mean()
                else:
                    raise NotImplementedError
            else:
                weight_fide = 1.0 / self.cfg.num_parts_training
                weight_regu = 1.0 / self.cfg.num_parts_training

            # store gradients
            loss = weight_fide * fidelity_loss + weight_regu * regularization_loss
            self.manual_backward(loss)

            # prepare for the next iteration
            latent = denoised_latents.detach()
            
        # update the weights
        opt.step()

    def validation_step(self, batch, batch_idx):

        # prepare the text embeddings as input
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            batch["text_embed"] = prompt_utils.get_global_text_embeddings()
            batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
    
        batch["space_cache"]  = self.diffusion_reverse(batch)
        out  = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]

        for batch_idx in tqdm(range(batch_size), desc="Saving val images"):
            self._save_image_grid(batch, batch_idx, out, phase="val", render="1st")
                
        if self.cfg.visualize_samples:
            raise NotImplementedError

    def test_step(self, batch, batch_idx):

        # prepare the text embeddings as input
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            batch["text_embed"] = prompt_utils.get_global_text_embeddings()
            batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
    
        batch["space_cache"] = self.diffusion_reverse(batch)
        out = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]

        for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
            self._save_image_grid(batch, batch_idx, out, phase="test", render="1st")

    def _compute_loss(
        self,
        guidance_out: Dict[str, Any],
        out: Dict[str, Any],
        renderer: str = "1st",
        step: int = 0,
        **batch,
    ):
        
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
                self.log(f"train/loss_sparsity_{step}", loss_sparsity)
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            else:
                self.log(f"train/loss_sparsity_2nd_{step}", loss_sparsity)
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

        # sdf loss
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_eikonal) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_eikonal_2nd) > 0):
            if 'sdf_grad' not in out:
                raise ValueError(
                    "sdf is required for eikonal loss, no sdf is found in the output."
                )
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            
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

        return {"fidelity_loss": fide_loss, "regularization_loss": regu_loss}


    def _save_image_grid(
        self, 
        batch,
        batch_idx,
        out,
        phase="val",
        render="1st",
    ):
        
        assert phase in ["val", "test"]

        # save the image with the same name as the prompt
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['prompt'][0].replace(',', '').replace('.', '').replace(' ', '_')
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
            for render in ["1st"]
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
            for render in ["1st"]
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
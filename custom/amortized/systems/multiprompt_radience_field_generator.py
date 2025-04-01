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

@threestudio.register("multiprompt-radience-field-generator-system")
class MultipromptRadienceFieldGeneratorSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture', 'coarse+geometry']
        stage: str = "coarse"

        # validation related
        visualize_samples: bool = False
        validation_via_video: bool = False

        # renderering related
        rgb_as_latents: bool = False

        # initialization related
        initialize_shape: bool = True

        # if the guidance requires training
        train_guidance: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        if self.cfg.train_guidance: # if the guidance requires training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)


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

        # initialize the prompt processor after dist init
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )

    # in case the prompt_processor is not initialized in the fit_start
    def on_predict_start(self) -> None:
        super().on_predict_start()

        # initialize the prompt processor after dist init
        if not hasattr(self, "prompt_processor"):
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
    
    # in case the prompt_processor is not initialized in the fit_start
    def on_test_start(self) -> None:
        super().on_test_start()

        # initialize the prompt processor after dist init
        if not hasattr(self, "prompt_processor"):
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.prompt_utils = self.prompt_processor(prompt = batch["prompt"])

        if "prompt_target" in batch:
            # for the case of interpolation
            self.prompt_utils_target = self.prompt_processor(prompt = batch["prompt_target"])
            ratio = batch["ratio"]
            batch["text_embed"] = ratio * self.prompt_utils.get_global_text_embeddings() + (1 - ratio) * self.prompt_utils_target.get_global_text_embeddings()
        else:
            # more general case
            batch["text_embed"] = self.prompt_utils.get_global_text_embeddings()
    
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
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

        return {
            **render_out,
        }

    def training_step(self, batch, batch_idx):
        out = self(batch)

        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
        else:
            guidance_inp = out["comp_rgb"]

        guidance_out = self.guidance(
            guidance_inp, 
            self.prompt_utils, 
            **batch, 
            rgb_as_latents=self.cfg.rgb_as_latents,
        )

        loss = 0.0
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if "coarse" in self.cfg.stage: # i.e. coarse or coarse+geometry
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
                # helps reduce floaters and produce solid geometry
                if 'z_variance' not in out:
                    raise ValueError(
                        "z_variance is required for z_variance loss, no z_variance is found in the output."
                    )
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            # sdf loss
            if hasattr(self.cfg.loss, 'lambda_eikonal')  and self.C(self.cfg.loss.lambda_eikonal) > 0:
                if 'sdf_grad' not in out:
                    raise ValueError(
                        "sdf is required for eikonal loss, no sdf is found in the output."
                    )
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
            
            if "inv_std" in out:
                self.log("train/inv_std", out["inv_std"], prog_bar=True)

        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}") # TODO

        # addition loss for geometry stage
        if self.cfg.stage == "coarse+geometry":
            # use the geometry_lr to control the contribution of geometry
            guidance_inp = torch.nan_to_num(out["comp_normal"], nan=0.0, posinf=0.0, neginf=0.0)
            guidance_out = self.guidance(
                guidance_inp, 
                self.prompt_utils, 
                **batch, 
                rgb_as_latents=False,
            )
            lambda_geo = 0.2 # hard-coded lambda
            for name, value in guidance_out.items():
                self.log(f"train/shape_{name}", value)
                if name.startswith("loss_"):
                    loss += lambda_geo * value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        # print("\n current device is:{}".format(loss.device))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)

        batch_size = out['comp_rgb'].shape[0]

        # save the image with the same name as the prompt
        name = batch['prompt'][0].replace(',', '').replace('.', '').replace(' ', '_')

        for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
            # visualize the depth
            if 'depth' in out:
                render_depth = True
                depth = out["depth"][batch_idx, :, :, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            else:
                render_depth = False
                
            self.save_image_grid(
                f"it{self.true_global_step}-val/{name}/{str(batch['index'][batch_idx].item())}.png"
                    if self.cfg.validation_via_video
                    else f"it{self.true_global_step}/{name}/{str(batch['index'][batch_idx].item())}.png",
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
                            "img": out["comp_normal"][batch_idx],
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
                            "img": depth,
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        },
                    ]
                    if render_depth
                    else []
                ),

                name=f"validation_step_batchidx_{batch_idx}" 
                    if self.cfg.validation_via_video
                    else "validation_step",
                step=self.true_global_step,
            )

        if self.cfg.visualize_samples:
            raise NotImplementedError

    def on_validation_epoch_end(self):
        if self.cfg.validation_via_video:
            filestem = f"it{self.true_global_step}-val"
            if get_rank() == 0: # only remove from one process
                for prompt in tqdm(
                    os.listdir(os.path.join(self.get_save_dir(), filestem)),
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
                        threestudio.info('cannot save {} at step {}'.format(prompt, self.true_global_step))
                    # shutil.rmtree(
                    #     os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
                    # )


    def test_step(self, batch, batch_idx):
        out = self(batch)

        batch_size = out['comp_rgb'].shape[0]

        # save the image with the same name as the prompt
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['prompt'][0].replace(',', '').replace('.', '').replace(' ', '_')

        for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
            # visualize the depth
            depth = out["depth"][batch_idx, :, :, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            self.save_image_grid(
                f"it{self.true_global_step}-test/{name}/{str(batch['index'][batch_idx].item())}.png",
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
                            "img": out["comp_normal"][batch_idx],
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
                            "img": depth,
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        },
                    ]
                    if 'depth' in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        filestem = f"it{self.true_global_step}-test"
        if get_rank() == 0: # only remove from one process
            for prompt in tqdm(
                os.listdir(os.path.join(self.get_save_dir(), filestem)),
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
                    threestudio.info('cannot save {} at step {}'.format(prompt, self.true_global_step))
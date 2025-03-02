import os
import re
import json
from tqdm import tqdm

import torch
from typing import *
from dataclasses import dataclass, field
from diffusers import StableDiffusionPipeline

from .base import Pipeline
from ..models.geometry import StableDiffusionTriplaneDualAttention
from ..utils.mesh_exporter import isosurface, colorize_mesh, DiffMarchingCubeHelper

from diffusers.loaders import AttnProcsLayers
from ..models.networks import get_activation

@dataclass
class TriplaneTurboTextTo3DPipelineConfig:
    """Configuration for TriplaneTurboTextTo3DPipeline"""
    # Basic pipeline settings
    base_model_name_or_path: str = "pretrained/stable-diffusion-2-1-base"

    num_inference_steps: int = 4
    num_results_per_prompt: int = 1
    latent_channels: int = 4
    latent_height: int = 64
    latent_width: int = 64
    
    # Training/sampling settings
    num_steps_sampling: int = 4
    
    # Geometry settings
    radius: float = 1.0
    normal_type: str = "analytic"
    sdf_bias: str = "sphere"
    sdf_bias_params: float = 0.5
    rotate_planes: str = "v1"
    split_channels: str = "v1"
    geo_interpolate: str = "v1"
    tex_interpolate: str = "v2"
    n_feature_dims: int = 3

    sample_scheduler: str = "ddim" # any of "ddpm", "ddim"
    
    # Network settings
    mlp_network_config: dict = field(
        default_factory=lambda: {
            "otype": "VanillaMLP",
            "activation": "ReLU",
            "output_activation": "none",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
    )

    # Adapter settings
    space_generator_config: dict = field(
        default_factory=lambda: {
            "training_type": "self_lora_rank_16-cross_lora_rank_16-locon_rank_16" ,
            "output_dim": 64,  # 32 * 2 for v1
            "self_lora_type": "hexa_v1",
            "cross_lora_type": "vanilla",
            "locon_type": "vanilla_v1",
            "prompt_bias": False,
            "vae_attn_type": "basic",  # "basic", "vanilla"
        }
    )

    isosurface_deformable_grid: bool = True
    isosurface_resolution: int = 160
    color_activation: str = "sigmoid-mipnerf"

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "TriplaneTurboTextTo3DPipelineConfig":
        """Load config from pretrained path"""
        config_path = os.path.join(pretrained_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        else:
            print(f"No config file found at {pretrained_path}, using default config")
            return cls()  # Return default config if no config file found

class TriplaneTurboTextTo3DPipeline(Pipeline):
    """
    A pipeline for converting text to 3D models using triplane representation.
    """
    config_name = "config.json"

    def __init__(
        self,
        geometry: StableDiffusionTriplaneDualAttention,
        material: Callable,
        base_pipeline: StableDiffusionPipeline,
        sample_scheduler: Callable,
        isosurface_helper: Callable,
        **kwargs,
    ):
        super().__init__()
        self.geometry = geometry
        self.material = material

        self.base_pipeline = base_pipeline

        self.sample_scheduler = sample_scheduler
        self.isosurface_helper = isosurface_helper


        self.models = {
            "geometry": geometry,
            "base_pipeline": base_pipeline,
        }
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        """
        Load pretrained adapter weights, config and update pipeline components.

        Args:
            pretrained_model_name_or_path: Path to pretrained adapter weights
            base_pipeline: Optional base pipeline instance
            **kwargs: Additional arguments to override config values

        Returns:
            pipeline: Updated pipeline instance
        """
        # Load config from pretrained path
        config = TriplaneTurboTextTo3DPipelineConfig.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

        # load base pipeline
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            config.base_model_name_or_path,
            **kwargs,
        )

        # load sample scheduler
        if config.sample_scheduler == "ddim":   
            from diffusers import DDIMScheduler
            sample_scheduler = DDIMScheduler.from_pretrained(
                config.base_model_name_or_path,
                subfolder="scheduler",
            )
        else:
            raise ValueError(f"Unknown sample scheduler: {config.sample_scheduler}")

        # load geometry
        geometry = StableDiffusionTriplaneDualAttention(
                config=config,
                vae=base_pipeline.vae,
                unet=base_pipeline.unet,
            )

        # no gradient for geometry
        for param in geometry.parameters():
            param.requires_grad = False

        # and load adapter weights
        if pretrained_model_name_or_path.endswith(".pth"):
            state_dict = torch.load(pretrained_model_name_or_path)
            new_state_dict = state_dict
            _, unused = geometry.load_state_dict(new_state_dict, strict=False)
            if len(unused) > 0:
                print(f"Unused keys: {unused}")
        else:
            raise ValueError(f"Unknown pretrained model name or path: {pretrained_model_name_or_path}")


        # load material, convert to int
        # material = lambda x: (256 * get_activation(config.color_activation)(x)).int()
        material = get_activation(config.color_activation)

        # Load geometry model
        pipeline = cls(
            base_pipeline=base_pipeline,
            geometry=geometry,
            sample_scheduler=sample_scheduler,
            material=material,
            isosurface_helper=DiffMarchingCubeHelper(
                resolution=config.isosurface_resolution,
            ),
            **kwargs,
        )
        return pipeline


    def encode_prompt(
        self, 
        prompt: Union[str, List[str]],
        device: str,
        num_results_per_prompt: int = 1,
    ) -> torch.FloatTensor:
        """
        Encodes the prompt into text encoder hidden states.
        
        Args:
            prompt: The prompt to encode.
            device: The device to use for encoding.
            num_results_per_prompt: Number of results to generate per prompt.
            do_classifier_free_guidance: Whether to use classifier-free guidance.
            negative_prompt: The negative prompt to encode.
            
        Returns:
            text_embeddings: Text embeddings tensor.
        """
        # Use base_pipeline to encode prompt
        text_embeddings = self.base_pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_results_per_prompt,
            do_classifier_free_guidance=False,
            negative_prompt=None
        )
        return text_embeddings
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 4,
        num_results_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        colorize: bool = True,
        **kwargs,
    ):
        # Implementation similar to Zero123Pipeline
        # Reference code from: https://github.com/zero123/zero123-diffusers
        
        # Validate inputs
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Prompt must be a string or list of strings, got {type(prompt)}")
        
        # Get the device from the first available module

        # Generate latents if not provided
        if latents is None:
            latents = torch.randn(
                (batch_size * 6, 4, 32, 32), # hard-coded for now
                generator=generator,
                device=self.device,
            )

        # Process text prompt through geometry module
        text_embed, _ = self.encode_prompt(prompt, self.device, num_results_per_prompt)
        
        # Run diffusion process
        # Set up timesteps for sampling
        timesteps = self._set_timesteps(
            self.sample_scheduler,
            num_inference_steps
        )


        with torch.no_grad():
            # Run diffusion process
            for i, t in tqdm(enumerate(timesteps)):
                # Scale model input
                noisy_latent_input = self.sample_scheduler.scale_model_input(
                    latents, 
                    t
                )

                # Predict noise/sample
                pred = self.geometry.denoise(
                    noisy_input=noisy_latent_input,
                    text_embed=text_embed,
                    timestep=t.to(self.device),
                )

                # Update latents
                results = self.sample_scheduler.step(pred, t, latents)
                latents = results.prev_sample
                latents_denoised = results.pred_original_sample

            # Use final denoised latents
            latents = latents_denoised
            
            # Generate final 3D representation
            space_cache = self.geometry.decode(latents)

            # Extract mesh from space cache
            mesh_list = isosurface(
                space_cache,
                self.geometry.forward_field,
                self.isosurface_helper,
            )

            if colorize:
                mesh_list = colorize_mesh(
                    space_cache,
                    self.geometry.export,
                    mesh_list,
                    activation=self.material,
                )

        # decide output type based on return_dict
        if return_dict:
            return {
                "space_cache": space_cache,
                "latents": latents,
                "mesh": mesh_list,
            }
        else:
            return mesh_list

    def _set_timesteps(
        self,
        scheduler,
        num_steps: int,
    ):
        """Set up timesteps for sampling.
        
        Args:
            scheduler: The scheduler to use for timestep generation
            num_steps: Number of diffusion steps
            
        Returns:
            timesteps: Tensor of timesteps to use for sampling
        """
        scheduler.set_timesteps(num_steps)
        timesteps_orig = scheduler.timesteps
        # Shift timesteps to start from T
        timesteps_delta = scheduler.config.num_train_timesteps - 1 - timesteps_orig.max()
        timesteps = timesteps_orig + timesteps_delta
        return timesteps


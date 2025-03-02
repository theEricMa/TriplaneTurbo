import os
import torch
import argparse
from typing import *
from diffusers import StableDiffusionPipeline

from triplaneturbo_executable.utils.mesh_exporter import export_obj
from triplaneturbo_executable import TriplaneTurboTextTo3DPipeline, TriplaneTurboTextTo3DPipelineConfig



# Initialize configuration and parameters
prompt = "a beautiful girl"
output_dir = "examples/output"
adapter_name_or_path = "pretrained/triplane_turbo_sd_v1.pth"
num_results_per_prompt = 1
seed = 42
device = "cuda"

# Initialize the TriplaneTurbo pipeline
triplane_turbo_pipeline = TriplaneTurboTextTo3DPipeline.from_pretrained(adapter_name_or_path)
triplane_turbo_pipeline.to(device)

# Run the pipeline
output = triplane_turbo_pipeline(
    prompt=prompt,
    num_results_per_prompt=num_results_per_prompt,
    generator=torch.Generator(device=device).manual_seed(seed),
    device=device,
)

# Save mesh
os.makedirs(output_dir, exist_ok=True)
for i, mesh in enumerate(output["mesh"]):
    name = f"{prompt.replace(' ', '_')}_{i}"
    save_paths = export_obj(mesh, f"{output_dir}/{name}.obj")
    print(f"Saved mesh to: {save_paths}")
    
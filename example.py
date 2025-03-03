import os
import torch
import argparse
from typing import *
from diffusers import StableDiffusionPipeline
from collections import deque

from triplaneturbo_executable.utils.mesh_exporter import export_obj
from triplaneturbo_executable import TriplaneTurboTextTo3DPipeline, TriplaneTurboTextTo3DPipelineConfig



# Initialize configuration and parameters
prompt = "a beautiful girl"
output_dir = "examples/output"
adapter_name_or_path = "pretrained/triplane_turbo_sd_v1.pth"
num_results_per_prompt = 1
seed = 42
device = "cuda"
max_obj_files = 100

# download pretrained models if not exist
if not os.path.exists(adapter_name_or_path):
    print(f"Downloading pretrained models from huggingface")
    os.system(
        f"huggingface-cli download --resume-download ZhiyuanthePony/TriplaneTurbo \
        --include \"triplane_turbo_sd_v1.pth\" \
        --local-dir ./pretrained \
        --local-dir-use-symlinks False"
        )


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

# Initialize a deque with maximum length of 100 to store obj file paths
obj_file_queue = deque(maxlen=max_obj_files)

# Save mesh
os.makedirs(output_dir, exist_ok=True)
for i, mesh in enumerate(output["mesh"]):
    vertices = mesh.v_pos
    
    # 1. First rotate -90 degrees around X-axis to make the model face up
    vertices = torch.stack([
        vertices[:, 0],           # x remains unchanged
        vertices[:, 2],           # y = z
        -vertices[:, 1]           # z = -y
    ], dim=1)
    
    # 2. Then rotate 90 degrees around Y-axis to make the model face the observer
    vertices = torch.stack([
        -vertices[:, 2],          # x = -z
        vertices[:, 1],           # y remains unchanged
        vertices[:, 0]            # z = x
    ], dim=1)
    
    mesh.v_pos = vertices
    
    # If mesh has normals, they need to be rotated in the same way
    if mesh.v_nrm is not None:
        normals = mesh.v_nrm
        # 1. Rotate -90 degrees around X-axis
        normals = torch.stack([
            normals[:, 0],
            normals[:, 2],
            -normals[:, 1]
        ], dim=1)
        # 2. Rotate 90 degrees around Y-axis
        normals = torch.stack([
            -normals[:, 2],
            normals[:, 1],
            normals[:, 0]
        ], dim=1)
        mesh._v_nrm = normals
    
    # Save obj file and add its path to the queue
    name = f"{prompt.replace(' ', '_')}_{seed}_{i}"
    save_paths = export_obj(mesh, f"{output_dir}/{name}.obj")
    obj_file_queue.append(save_paths[0])
    
    # If an old file needs to be removed (queue is at max length)
    # and the file exists, delete it
    if len(obj_file_queue) == max_obj_files and os.path.exists(obj_file_queue[0]):
        old_file = obj_file_queue[0]
        try:
            os.remove(old_file)
        except OSError as e:
            print(f"Error deleting file {old_file}: {e}")


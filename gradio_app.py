import os
import torch
import gradio as gr
from typing import *
from diffusers import StableDiffusionPipeline

from triplaneturbo_executable import TriplaneTurboTextTo3DPipeline
from triplaneturbo_executable.utils.mesh_exporter import export_obj

# Initialize global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_PATH = "pretrained/triplane_turbo_sd_v1.pth"
PIPELINE = None  # Will hold our pipeline instance

def download_model():
    """Download the pretrained model if not exists"""
    if not os.path.exists(ADAPTER_PATH):
        print("Downloading pretrained models from huggingface")
        os.system(
            f"huggingface-cli download --resume-download ZhiyuanthePony/TriplaneTurbo \
            --include \"triplane_turbo_sd_v1.pth\" \
            --local-dir ./pretrained \
            --local-dir-use-symlinks False"
        )

def initialize_pipeline():
    """Initialize the pipeline once and keep it in memory"""
    global PIPELINE
    if PIPELINE is None:
        print("Initializing pipeline...")
        PIPELINE = TriplaneTurboTextTo3DPipeline.from_pretrained(ADAPTER_PATH)
        PIPELINE.to(DEVICE)
        print("Pipeline initialized!")
    return PIPELINE

def generate_3d_mesh(prompt: str, seed: int = 42) -> str:
    """Generate 3D mesh from text prompt"""
    global PIPELINE
    
    # Use the global pipeline instance
    pipeline = initialize_pipeline()
    
    # Generate mesh
    output = pipeline(
        prompt=prompt,
        num_results_per_prompt=1,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
    )
    
    # Save mesh
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    mesh_path = None
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
        
        name = f"{prompt.replace(' ', '_')}_{i}"
        save_paths = export_obj(mesh, f"{output_dir}/{name}.obj")
        mesh_path = save_paths[0]
        
    return mesh_path

def main():
    # Download model if needed
    download_model()
    
    # Initialize pipeline at startup
    initialize_pipeline()
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=generate_3d_mesh,
        inputs=[
            gr.Textbox(label="Text Prompt", placeholder="Enter your text description..."),
            gr.Number(label="Random Seed", value=42)
        ],
        outputs=gr.Model3D(
            label="Generated 3D Mesh",
            camera_position=(90, 90, 3),  # alpha=180° rotates to back, beta=0° for horizontal view, radius=2 units
            clear_color=(0.5, 0.5, 0.5, 1),
        ),
        title="Text to 3D Mesh Generation",
        description="Generate 3D meshes from text descriptions using TriplaneTurbo",
        examples=[
            ["a beautiful girl", 42],
            ["a cute cat", 123],
            ["an elegant vase", 456],
        ]
    )
    
    # Launch the interface
    iface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Specify port
        share=True,             # Create public link
    )

if __name__ == "__main__":
    main()
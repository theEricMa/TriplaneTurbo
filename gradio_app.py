import os
import torch
import gradio as gr
from typing import *
from collections import deque
from diffusers import StableDiffusionPipeline

from triplaneturbo_executable import TriplaneTurboTextTo3DPipeline
from triplaneturbo_executable.utils.mesh_exporter import export_obj

# Initialize global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_PATH = "pretrained/triplane_turbo_sd_v1.pth"
PIPELINE = None  # Will hold our pipeline instance
OBJ_FILE_QUEUE = deque(maxlen=100)  # Queue to store OBJ file paths

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

def generate_3d_mesh(prompt: str) -> Tuple[str, str]:
    """Generate 3D mesh from text prompt"""
    global PIPELINE, OBJ_FILE_QUEUE
    
    # Use the global pipeline instance
    pipeline = initialize_pipeline()
    
    # Use fixed seed value
    seed = 42
    
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
        
        name = f"{prompt.replace(' ', '_')}"
        save_paths = export_obj(mesh, f"{output_dir}/{name}.obj")
        mesh_path = save_paths[0]
        
        # Add new file path to queue
        OBJ_FILE_QUEUE.append(mesh_path)
        
        # If queue is at max length, remove oldest file
        if len(OBJ_FILE_QUEUE) == OBJ_FILE_QUEUE.maxlen:
            old_file = OBJ_FILE_QUEUE[0]  # Get oldest file (will be automatically removed from queue)
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except OSError as e:
                    print(f"Error deleting file {old_file}: {e}")
        
    return mesh_path, mesh_path  # Return the path twice - once for 3D preview, once for download

def main():
    # Download model if needed
    download_model()
    
    # Initialize pipeline at startup
    initialize_pipeline()
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=generate_3d_mesh,
        inputs=[
            gr.Textbox(label="Text Prompt", placeholder="Enter your text description...")
        ],
        outputs=[
            gr.Model3D(
                label="Generated 3D Mesh",
                camera_position=(90, 90, 3),
                clear_color=(0.5, 0.5, 0.5, 1),
            ),
            gr.File(label="Download OBJ file")
        ],
        title="Text to 3D Mesh Generation with TriplaneTurbo",
        description="Demo of the paper Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation beyond 3D Training Data [CVPR 2025] <br><a href='https://github.com/theEricMa/TriplaneTurbo' style='color: #2196F3;'>https://github.com/theEricMa/TriplaneTurbo</a>",
        examples=[
            ["Armor dress style of outsiderzone fantasy helmet"],
            ["Gandalf the grey riding a camel in a rock concert, victorian newspaper article, hyperrealistic"],
            ["A DSLR photo of a bald eagle"],
            ["A goblin riding a lawnmower in a hospital, victorian newspaper article, 4k hd"],
            ["An imperial stormtrooper, highly detailed"],
        ],
        allow_flagging="never",
    )
    
    # Launch the interface
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )

if __name__ == "__main__":
    main()
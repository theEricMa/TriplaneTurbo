from diffusers import DiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
pipeline.save_pretrained("./pretrained/stable-diffusion-2-1-base")

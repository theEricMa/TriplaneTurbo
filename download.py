from diffusers import DiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
pipeline.save_pretrained("./pretrained/stable-diffusion-2-1-base")

# repo_id = "stabilityai/stable-diffusion-2-1"
# pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
# pipeline.save_pretrained("./pretrained/stable-diffusion-2-1")

repo_id = "stabilityai/sd-turbo"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
pipeline.save_pretrained("./pretrained/sd-turbo")

from diffusers import StableDiffusionPipeline

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
pipe.save_pretrained("./pretrained/stable-diffusion-v1-5")

import os

cmd = "wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt?download=true -O ./pretrained/sd-v2.1-base-4view.pt"
os.system(cmd)

# download pickapic scripts

# from diffusers import StableDiffusionXLPipeline
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = StableDiffusionXLPipeline.from_pretrained(model_id, use_safetensors=True)
# pipe.save_pretrained("./pretrained/stable-diffusion-xl-base-1.0")


# cmd = "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./pretrained/sam_vit_h_4b8939.pth"
# os.system(cmd)

# cmd = "wget https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt -O ./pretrained/stable_zero123.ckpt"
# os.system(cmd)


# from diffusers import StableUnCLIPImg2ImgPipeline

# repo_id = "stabilityai/stable-diffusion-2-1-unclip"
# pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(repo_id, use_safetensors=True)
# pipeline.save_pretrained("./pretrained/stable-diffusion-2-1-unclip")

cmd = "wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd_mv_ema.ckpt -O ./pretrained/nd_mv_ema.ckpt"
os.system(cmd)

# repo_id = "runwayml/stable-diffusion-v1-5"
# pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
# pipeline.save_pretrained("./pretrained/stable-diffusion-v1-5")

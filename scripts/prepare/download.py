from diffusers import DiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
pipeline.save_pretrained("./pretrained/stable-diffusion-2-1-base")


from diffusers import StableDiffusionPipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
pipe.save_pretrained("./pretrained/stable-diffusion-v1-5")

import os
cmd = "wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt?download=true -O ./pretrained/sd-v2.1-base-4view.pt"
os.system(cmd)


cmd = "wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd_mv_ema.ckpt -O ./pretrained/nd_mv_ema.ckpt"
os.system(cmd)

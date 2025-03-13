CUDA_VISIBLE_DEVICES=2 python launch.py \
    --config configs/v1_mem-48G_iters-20k.yaml  \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
    --config configs/group_1-1/3DTopia__base_step_4__asd_mv+rd+sd_volsdf+diffmc__triple_32_vanilla_32_bias_lora_prompt.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/group_1-1/3DTopia__base_step_4__asd_mv+rd+sd_volsdf+diffmc__triple_32_vanilla_32_bias_lora_prompt.yaml \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 
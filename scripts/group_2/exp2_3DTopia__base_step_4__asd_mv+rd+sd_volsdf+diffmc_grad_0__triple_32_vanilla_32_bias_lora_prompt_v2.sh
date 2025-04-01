CUDA_VISIBLE_DEVICES=4,5,6,7  python launch.py \
    --config configs/group_2/3DTopia__base_step_4__asd_mv+rd+sd_volsdf+diffmc_grad_0__triple_32_vanilla_32_bias_lora_prompt_v2.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/group_2/3DTopia__base_step_4__asd_mv+rd+sd_volsdf+diffmc_grad_0__triple_32_vanilla_32_bias_lora_prompt_v2.yaml \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 
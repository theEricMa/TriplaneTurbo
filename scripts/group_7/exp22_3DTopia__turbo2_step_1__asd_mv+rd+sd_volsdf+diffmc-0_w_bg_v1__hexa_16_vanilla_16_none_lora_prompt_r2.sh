CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python launch.py \
    --config configs/group_7/3DTopia__turbo2_step_1__asd_mv+rd+sd_volsdf+diffmc-0_w_bg_v1__hexa_16_vanilla_16_none_lora_prompt_r2.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

CUDA_VISIBLE_DEVICES=0,1,2,3  python launch.py \
    --config configs/group_9/3DTopia__base2_step_2__asd_mv+rd+sd_volsdf+diffmc-0_h_v_16_v_16_bias_lora_dmd_v4.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

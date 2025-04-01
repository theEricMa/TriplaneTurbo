CUDA_VISIBLE_DEVICES=0,1,2,3  python launch.py \
    --config configs/group_7/3DTopia__base2_step_4__asd_mv+rd_volsdf+diffmc-01_wo_bg_v1__hexa_16_vanilla_16_none_lora_prompt_r1.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

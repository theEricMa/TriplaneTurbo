CUDA_VISIBLE_DEVICES=0,1,2,3  python launch.py \
    --config configs/group_8-1/exp6_3DTopia__base2_step_4__asd_mv+rd+sd_volsdf+diffmc-0_hexa_16_vanilla_16_wobias_lora_prompt_v2.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

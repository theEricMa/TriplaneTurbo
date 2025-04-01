CUDA_VISIBLE_DEVICES=4,5,6,7 python launch.py \
    --config configs/group_10/Cap3D__base2_step_4__asd_mv+rd+sd_volsdf+diffmc-0_h_v_16_v_16_none_lora_dmd_v1_eik1.yaml \
    --train \
    data.prompt_library="Cap3D_306k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_Cap3D_306k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_Cap3D_306k" 

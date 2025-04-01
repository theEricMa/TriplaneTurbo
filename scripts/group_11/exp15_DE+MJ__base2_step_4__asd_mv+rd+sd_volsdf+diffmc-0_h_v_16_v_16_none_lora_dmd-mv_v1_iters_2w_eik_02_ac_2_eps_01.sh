CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/group_11/DE+MJ__base2_step_4__asd_mv+rd+sd_volsdf+diffmc-0_h_v_16_v_16_none_lora_dmd-mv_v1_iters_2w_eik_02_ac_2_eps_01.yaml  \
    --train \
    data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 

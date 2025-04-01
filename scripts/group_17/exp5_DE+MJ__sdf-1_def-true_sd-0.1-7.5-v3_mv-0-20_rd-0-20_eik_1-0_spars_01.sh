CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/group_17/DE+MJ__sdf-1_def-true_sd-0.1-7.5-v3_mv-0-20_rd-0-20_eik_1-0_spars_01.yaml \
    --train \
    data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 

      

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/group_18/DE+MJ__vae-fix__ns_100_def=0.01-1_nobg__sd-0.1-7.5-v1_mv-10-v1_rd-20-v1__eik_1-0_spars_1-0_eps_1.yaml \
    --train \
    data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 

      

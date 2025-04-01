CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/group_12/DE+MJ__diffmc-001_dmd-mv_v1_iters_3w_eik_02_ac_1_eps_01.yaml \
    --train \
    data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 

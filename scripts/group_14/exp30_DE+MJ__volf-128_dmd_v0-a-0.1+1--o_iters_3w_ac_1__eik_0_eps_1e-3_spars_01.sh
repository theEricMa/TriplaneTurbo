CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/group_14/DE+MJ__volf-128_dmd_v0-a-0.1+1--o_iters_3w_ac_1__eik_0_eps_1e-3_spars_01.yaml \
    --train \
    data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DALLE_Midjourney" 

      

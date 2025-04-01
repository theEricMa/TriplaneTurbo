NUM_TRAINERS=8

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    launch.py \
        --config configs/group_13/DE+MJ__diffmc-001_dmd_v3_iters_3w_ac_1__eik_0_eps_1e-3_spars_10-50.yaml \
        --train \
        --gpu 0,1,2,3,4,5,6,7 \
        data.prompt_library="DALLE_Midjourney_1313928_prompt_library" \
        data.condition_processor.cache_dir="/dfs/ai-storage/Scaledreamer/.threestudio_cache/text_embeddings_DALLE_Midjourney" \
        data.guidance_processor.cache_dir="/dfs/ai-storage/Scaledreamer/.threestudio_cache/text_embeddings_DALLE_Midjourney" 

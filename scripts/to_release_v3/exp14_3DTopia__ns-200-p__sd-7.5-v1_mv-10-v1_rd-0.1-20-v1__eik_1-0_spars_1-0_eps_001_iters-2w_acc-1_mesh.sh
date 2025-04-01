CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/to_release_3/3DTopia__ns-200-p__sd-7.5-v1_mv-10-v1_rd-0.1-20-v1__eik_1-0_spars_1-0_eps_001_iters-2w_acc-1_mesh.yaml  \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 


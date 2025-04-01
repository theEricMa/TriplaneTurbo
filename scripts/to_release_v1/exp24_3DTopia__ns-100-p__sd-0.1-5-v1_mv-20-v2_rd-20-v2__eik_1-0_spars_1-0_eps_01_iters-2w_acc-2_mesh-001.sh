CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config  configs/to_release_1/3DTopia__ns-100-p__sd-0.1-5-v1_mv-20-v2_rd-20-v2__eik_1-0_spars_1-0_eps_01_iters-2w_acc-2_mesh-001.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 


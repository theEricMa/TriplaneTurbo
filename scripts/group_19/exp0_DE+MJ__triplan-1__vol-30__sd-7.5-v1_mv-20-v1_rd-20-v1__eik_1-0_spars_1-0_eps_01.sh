CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/group_19/DE+MJ__triplan-1__vol-30__sd-7.5-v1_mv-20-v1_rd-20-v1__eik_1-0_spars_1-0_eps_01.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k"

      
# CUDA_VISIBLE_DEVICES=0 python launch.py \
#     --config configs/group_19/DE+MJ__triplan-1__vol-30__sd-7.5-v1_mv-20-v1_rd-20-v1__eik_1-0_spars_1-0_eps_01.yaml \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.batch_size=4
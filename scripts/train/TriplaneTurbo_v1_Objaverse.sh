# # 8 GPUs with 98 GB memory each
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
#     --config configs/TriplaneTurbo_v1.yaml  \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     system.parallel_guidance=true
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

# # 8 GPUs with 80 GB memory each
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
#     --config configs/TriplaneTurbo_v1.yaml  \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

# 8 GPUs with 48 GB memory each
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
#     --config configs/TriplaneTurbo_v1_acc-2.yaml  \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
#     data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 

CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/TriplaneTurbo_v1_acc-2.yaml  \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia_361k" 
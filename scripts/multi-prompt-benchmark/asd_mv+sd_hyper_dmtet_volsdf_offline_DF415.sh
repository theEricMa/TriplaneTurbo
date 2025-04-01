# CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv+sd_hyper_dmtet_volsdf_10k_offline.yaml \
#     --train \
#     system.prompt_processor.prompt_library="dreamfusion_415_prompt_library"

CUDA_VISIBLE_DEVICES=0  python launch.py \
    --config configs/multi-prompt_benchmark/asd_mv+sd_hyper_dmtet_volsdf_10k_offline.yaml \
    --train \
    system.prompt_processor.prompt_library="magic3d_15_prompt_library"

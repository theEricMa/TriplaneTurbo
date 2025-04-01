CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
    --config configs/multi-prompt_benchmark/asd_mv_hyper_volsdf_10k.yaml \
    --train \
    system.prompt_processor.prompt_library="dreamfusion_415_prompt_library"

# CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv_hyper_volsdf_10k.yaml \
#     --train \
#     system.prompt_processor.prompt_library="dreamfusion_415_prompt_library"


# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv_hyper_volsdf_10k.yaml \
#     --train \
#     system.prompt_processor.prompt_library="dreamfusion_415_prompt_library"

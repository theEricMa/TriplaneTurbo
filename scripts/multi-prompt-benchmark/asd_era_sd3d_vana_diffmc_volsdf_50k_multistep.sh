# CUDA_VISIBLE_DEVICES=7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"

# CUDA_VISIBLE_DEVICES=7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml  \
#     --train \
#      data.image_library="sdxl_3d_animation_v1_7_image_library" \
#      data.image_root_dir="datasets/sdxl_3d_animation_v1_7"


# CUDA_VISIBLE_DEVICES=5,6  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml  \
#     --train \
#      data.image_library="sdxl_3d_animation_v1_7_image_library" \
#      data.image_root_dir="datasets/sdxl_3d_animation_v1_7"

# CUDA_VISIBLE_DEVICES=4,7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml  \
#     --train \
#      data.image_library="sdxl_3d_animation_v1_7_image_library" \
#      data.image_root_dir="datasets/sdxl_3d_animation_v1_7" \
#      name="asd_era_sd3d_diffmc_volsdf_50k_multistep_exp2"


# CUDA_VISIBLE_DEVICES=3  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml \
#     --train \
#      data.image_library="sdxl_3d_animation_v2_2056_image_library" \
#      data.image_root_dir="datasets/sdxl_3d_animation_v2_2056" \
#      name="asd_era_sd3d_diffmc_volsdf_50k_multistep_exp2"


# CUDA_VISIBLE_DEVICES=3  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml \
#     --train \
#      data.image_library="layerdiffuse_v0_1_image_library" \
#      data.image_root_dir="datasets/layerdiffuse_v1_7488" \
#      trainer.val_check_interval=1 \
#      resume="outputs_13/asd_era_sd3d_hexa_diffmc_volsdf_50k_multistep/layerdiffuse_v1_7488_image_library_v12_wasted/ckpts/epoch=1-step=3500.ckpt"
#     #  resume="outputs_13/asd_era_sd3d_hexa_diffmc_volsdf_50k_multistep/layerdiffuse_v1_7488_image_library_v9_wasted/ckpts/epoch=0-step=1000.ckpt"
#     #  system.visualize_samples=true \

#     #  trainer.val_check_interval=1

# CUDA_VISIBLE_DEVICES=3  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml \
#     --test \
#      data.image_library="layerdiffuse_v0_1_image_library" \
#      data.image_root_dir="datasets/layerdiffuse_v1_7488" \
#      trainer.val_check_interval=1 \
#      resume="outputs_13/asd_era_sd3d_hexa_diffmc_volsdf_50k_multistep/layerdiffuse_v1_7488_image_library_v12_wasted/ckpts/epoch=1-step=3500.ckpt" \
#     # system.exporter_type="multiprompt-mesh-exporter" \
#     # system.exporter.fmt=obj

CUDA_VISIBLE_DEVICES=5,6 python launch.py \
    --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml \
    --train \
     data.image_library="layerdiffuse_v1_7488_image_library" \
     data.image_root_dir="datasets/layerdiffuse_v1_7488"

# CUDA_VISIBLE_DEVICES=5 python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_vana_diffmc_volsdf_50k_multistep.yaml \
#     --train \
#      data.image_library="layerdiffuse_v1_7488_image_library" \
#      data.image_root_dir="datasets/layerdiffuse_v1_7488"
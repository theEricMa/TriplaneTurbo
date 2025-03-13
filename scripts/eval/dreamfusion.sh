
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python launch.py \
    --config configs/v1_mem-48G_iters-20k.yaml \
    --export \
    system.exporter_type="multiprompt-mesh-exporter" \
    system.weights="pretrained/triplane_turbo_sd_v1.pth" \
    data.prompt_library="dreamfusion_415_prompt_library" \
    system.exporter.fmt=obj

SAVE_DIR=outputs/TriplaneTurbo_v1_mem-48G_iters-20k/dreamfusion_415_prompt_library/save/it0-4views

# run in single GPU, single thread
python evaluation/mesh_visualize.py \
    outputs/TriplaneTurbo_v1_mem-48G_iters-20k/dreamfusion_415_prompt_library/save/it0-export \
    --save_dir $SAVE_DIR \
    --gpu 0 \
    --debug True

# # can specify more GPUs, e.g., 0,1,2,3,4,5,6,7 with all the threads
# python evaluation/mesh_visualize.py \
#     outputs/TriplaneTurbo_v1_mem-48G_iters-20k/dreamfusion_415_prompt_library/save/it0-export \
#     --save_dir $SAVE_DIR \
#     --gpu 0,1,2,3,4,5,6,7
    
python evaluation/clipscore/compute.py \
    --result_dir $SAVE_DIR
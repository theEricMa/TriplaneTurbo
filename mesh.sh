
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python launch.py \
    --config ??? \
    --export \
    system.exporter_type="multiprompt-mesh-exporter" \
    resume=??? \
    data.prompt_library="dreamfusion_415_prompt_library" \
    system.exporter.fmt=obj

SAVE_DIR=???
python evaluation/mesh_visualize.py\
    ??? \
    --save_dir $SAVE_DIR \
    --gpu 1,2,3,4,5,6,7

python evaluation/clipscore/compute.py \
    --result_dir $SAVE_DIR
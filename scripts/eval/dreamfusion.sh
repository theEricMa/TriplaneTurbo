#!/bin/bash
export NCCL_P2P_DISABLE=1 # disable P2P, do not affect the performance
# Default GPU setting
GPUS="0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash scripts/eval/dreamfusion.sh [--gpu GPU_IDS]"
      echo "Examples:"
      echo "  Single GPU: bash scripts/eval/dreamfusion.sh --gpu 0"
      echo "  Multiple GPUs: bash scripts/eval/dreamfusion.sh --gpu 0,1,2,3"
      exit 1
      ;;
  esac
done

echo "Using GPUs: $GPUS"

# Generate meshes
CUDA_VISIBLE_DEVICES=$GPUS python launch.py \
    --config configs/TriplaneTurbo_v1_acc-2.yaml \
    --export \
    system.exporter_type="multiprompt-mesh-exporter" \
    system.weights="pretrained/triplane_turbo_sd_v1.pth" \
    data.prompt_library="dreamfusion_415_prompt_library" \
    system.exporter.fmt=obj

SAVE_DIR=outputs/TriplaneTurbo_v1_mem-48G_iters-20k/dreamfusion_415_prompt_library/save/it0-4views
OBJ_DIR=outputs/TriplaneTurbo_v1_mem-48G_iters-20k/dreamfusion_415_prompt_library/save/it0-export

# Run visualization using the specified GPUs
python evaluation/mesh_visualize.py \
    $OBJ_DIR \
    --save_dir $SAVE_DIR \
    --gpu $GPUS \
    --debug $([ "$GPUS" == "0" ] && echo "True" || echo "False")

# Compute metrics
python evaluation/clipscore/compute.py \
    --result_dir $SAVE_DIR
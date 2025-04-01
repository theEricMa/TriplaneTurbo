<div align="center">
<h1>Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data</h1>

<div style="background-color: #EEEEEE; color: #333333; padding: 15px; margin: 20px 0; border-radius: 5px; border: 1px solid #999999; font-weight: bold;">
⚠️ WARNING: This branch contains unfiltered original codebase! Not recommended for production use! ⚠️
</div>



</div>

> **Note:** This branch is specifically maintained to verify how the same evaluation code performs across different codebases. It allows us to analyze the impact of different implementation environments on evaluation metrics while keeping the evaluation code itself consistent.

## 🛠️ Official Installation

For installation instructions, please refer to the [main branch](https://github.com/theEricMa/TriplaneTurbo/tree/main) of our repository.

## 📊 Evaluation

If you only want to run the evaluation without training, follow these steps:

```sh
# Download the model from HuggingFace
huggingface-cli download --resume-download ZhiyuanthePony/TriplaneTurbo \
    --include "triplane_turbo_sd_v1.pth" \
    --local-dir ./pretrained \
    --local-dir-use-symlinks False

# Download evaluation assets
python scripts/prepare/download_eval_only.py

# Run evaluation script
bash scripts/eval/dreamfusion.sh --gpu 0,1 # You can use more GPUs (e.g. 0,1,2,3,4,5,6,7). For single GPU usage, please check the script for required modifications
```

Our evaluation metrics include:
- CLIP Similarity Score
- CLIP Recall@1

For detailed evaluation results, please refer to our paper.

If you want to evaluate your own model, use the following script:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config <path_to_your_exp_config> \
    --export \
    system.exporter_type="multiprompt-mesh-exporter" \
    resume=<path_to_your_ckpt> \
    data.prompt_library="dreamfusion_415_prompt_library" \
    system.exporter.fmt=obj
```

After running the script, you will find generated OBJ files in `outputs/<your_exp>/dreamfusion_415_prompt_library/save/<itXXXXX-export>`. Set this path as `<OBJ_DIR>`, and set `outputs/<your_exp>/dreamfusion_415_prompt_library/save/<itXXXXX-4views>` as `<VIEW_DIR>`. Then run:

```sh
SAVE_DIR=<VIEW_DIR>
python evaluation/mesh_visualize.py \
    <OBJ_DIR> \
    --save_dir $SAVE_DIR \
    --gpu 0,1,2,3,4,5,6,7

python evaluation/clipscore/compute.py \
    --result_dir $SAVE_DIR
```
The evaluation results will be displayed in your terminal once the computation is complete.

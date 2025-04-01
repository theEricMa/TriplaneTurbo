<img src="assets/Showcase_v4.drawio.png" width="100%" align="center">
<div align="center">
<h1>Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data</h1>
<div>
    <a href='https://scholar.google.com/citations?user=F15mLDYAAAAJ&hl=en' target='_blank'>Zhiyuan Ma</a>&emsp;
    <a href='https://scholar.google.com/citations?user=R9PlnKgAAAAJ&hl=en' target='_blank'>Xinyue Liang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=en' target='_blank'>Rongyuan Wu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=1rbNk5oAAAAJ&hl=zh-CN' target='_blank'>Xiangyu Zhu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=cuJ3QG8AAAAJ&hl=en' target='_blank'>Zhen Lei</a>&emsp;
    <a href='https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en' target='_blank'>Lei Zhang</a>
</div>

<div>
<a href="https://arxiv.org/abs/2503.21694"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://theericma.github.io/TriplaneTurbo/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/ZhiyuanthePony/TriplaneTurbo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
<a href='https://theericma.github.io/TriplaneTurbo/static/pdf/main.pdf'><img src='https://img.shields.io/badge/Slides-Presentation-orange?logo=microsoftpowerpoint&logoColor=white' alt='Presentation Slides'></a>
</div>


---

</div>

<!-- Updates -->
## ⏩ Updates

- **2025-04-01**: Presentation slides are now available for download.
- **2025-03-27**: The paper is now available on Arxiv.
- **2025-03-03**: Gradio and HuggingFace Demos are available.
- **2025-02-27**: TriplaneTurbo is accepted to CVPR 2025.

<!-- Features -->
## 🌟 Features
- **Fast Inference 🚀**: Our code excels in inference efficiency, capable of outputting textured mesh in around 1 second.
- **Text Comprehension 🆙**: It demonstrates strong understanding capabilities for complex text prompts, ensuring accurate generation according to the input.
- **3D-Data-Free Training 🙅‍♂️**: The entire training process doesn't rely on any 3D datasets, making it more resource-friendly and adaptable.


## 🤖 Start local inference in 3 minutes
If you only wish to set up the demo locally, use the following code for the inference. Otherwise, for training and evaluation, use the next section of instructions for environment setup.

```python
python -m venv venv
source venv/bin/activate
bash setup.sh
python gradio_app.py
```

## 🛠️ Official Installation

Create a virtual environment:
```sh
conda create -n triplaneturbo python=3.10
conda activate triplaneturbo
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
(Optional, Recommended) Install xFormers for attention acceleration:
```sh
conda install xFormers -c xFormers
```
(Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions
```sh
pip install ninja
```
Install major dependencies
```sh
pip install -r requirements.txt
```
Install iNGP
```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
If you encounter errors while installing iNGP, it is recommended to check your gcc version. Follow these steps to change the gcc version within your -cconda environment. After that, return to the project directory and reinstall iNGP and NerfAcc:
```sh
conda install -c conda-forge gxx=9.5.0
cd  $CONDA_PREFIX/lib
ln -s  /usr/lib/x86_64-linux-gnu/libcuda.so ./
cd <your project directory>
```

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

## 🚀 Training Options

### 1. Download Required Pretrained Models and Datasets
Use the provided download script to get all necessary files:
```sh
python scripts/prepare/download_full.py
```

This will download:
- Stable Diffusion 2.1 Base
- Stable Diffusion 1.5
- MVDream 4-view checkpoint
- RichDreamer checkpoint
- Text prompt datasets (3DTopia and DALLE+Midjourney)

### 2. Training Options

#### Option 1: Train with 3DTopia Text Prompts
```sh
# Single GPU
CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/TriplaneTurbo_v0_acc-2.yaml \
    --train \
    data.prompt_library="3DTopia_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia"
```

For multi-GPU training:
```sh
# 8 GPUs with 48GB+ memory each
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/TriplaneTurbo_v1_acc-2.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_3DTopia"
```

#### Option 2: Train with DALLE+Midjourney Text Prompts
Choose the appropriate command based on your GPU configuration:

```sh
# Single GPU
CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/TriplaneTurbo_v0_acc-2.yaml \
    --train \
    data.prompt_library="DALLE_Midjourney_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DE+MJ" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DE+MJ"
```

For multi-GPU training (higher performance):
```sh
# 8 GPUs with 48GB+ memory each
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/TriplaneTurbo_v1_acc-2.yaml \
    --train \
    data.prompt_library="DALLE_Midjourney_prompt_library" \
    data.condition_processor.cache_dir=".threestudio_cache/text_embeddings_DE+MJ" \
    data.guidance_processor.cache_dir=".threestudio_cache/text_embeddings_DE+MJ"
```

### 3. Configuration Notes
- **Memory Requirements**: 
  - v1 configuration: Requires GPUs with 48GB+ memory
  - v0 configuration: Works with GPUs that have less memory (46GB+) but with reduced performance
  
- **Acceleration Options**:
  - Use `_acc-2.yaml` configs for gradient accumulation to reduce memory usage

- **Advanced Options**:
  - For highest quality, use `configs/TriplaneTurbo_v1.yaml` with `system.parallel_guidance=true` (requires 98GB+ memory GPUs)
  - To disable certain guidance components: add `guidance.rd_weight=0 guidance.sd_weight=0` to the command





<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:
```
@article{ma2025progressive,
  title={Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data},
  author={Ma, Zhiyuan and Liang, Xinyue and Wu, Rongyuan and Zhu, Xiangyu and Lei, Zhen and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2025}
}
```


<!-- Acknowledgement -->
## 🙏 Acknowledgement
Our code is heavily based on the following works
- [ThreeStudio](https://github.com/threestudio-project/threestudio): A clean and extensible codebase for 3D generation via Score Distillation.
- [MVDream](https://github.com/bytedance/MVDream): Used as one of our multi - view teachers.
- [RichDreamer](https://github.com/bytedance/MVDream): Serves as another multi - view teacher for normal and depth supervision
- [3DTopia](https://github.com/3DTopia/3DTopia): Its text caption dataset is applied in our training and comparison.
- [DiffMC](https://github.com/SarahWeiii/diso): Our solution uses its differentiable marching cube for mesh rasterization.
- [NeuS](https://github.com/Totoro97/NeuS): We implement its SDF - based volume rendering for dual rendering in our solution


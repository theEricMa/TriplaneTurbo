<div align="center">
<h1>Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data</h1>

<div style="background-color: #EEEEEE; color: #333333; padding: 15px; margin: 20px 0; border-radius: 5px; border: 1px solid #999999; font-weight: bold;">
⚠️ WARNING: This branch contains unfiltered original codebase! Not recommended for production use! ⚠️
</div>

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
</div>


---

</div>

<!-- Updates -->
## ⏩ Updates

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

### Dependencies and Installation

Create a virtual environment:
```sh
conda create -n triplaneturbo python=3.10
conda activate triplaneturbo
```

Install PyTorch:
```sh
# Prefer using the latest version of CUDA and PyTorch
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

(Optional, Recommended) Install [xFormers](https://github.com/facebookresearch/xformers) for attention acceleration:
```sh
conda install xformers -c xformers
```

(Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:
```sh
pip install ninja
```

Install major dependencies:
```sh
pip install -r requirements.txt
```

Install [iNGP](https://github.com/NVlabs/instant-ngp) and [NerfAcc](https://github.com/nerfstudio-project/nerfacc):
```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
```

If you encounter errors while installing iNGP, it is recommended to check your gcc version. Follow these instructions to change the gcc version within your conda environment. Then return to the repository directory to install iNGP and NerfAcc ⬆️ again.
```sh
conda install -c conda-forge gxx=9.5.0
cd  $CONDA_PREFIX/lib
ln -s  /usr/lib/x86_64-linux-gnu/libcuda.so ./
cd <your repo directory>
```

### Download 2D Diffusion Priors

Save [SD-v2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and [MVDream](https://mv-dream.github.io/) to the local directory `pretrained`:
```
python scripts/download_pretrained_models.py
```

### Download additional diffusion prior
```sh
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd_mv_ema.ckpt -O ./pretrained/nd_mv_ema.ckpt
```

Make sure the files in `pretrained` directory are as follows:
```
pretrained
├── sd-turbo
├── stable-diffusion-v1-5
├── stable-diffusion-2-1-base
├── nd_mv_ema.ckpt
├── sd-v2.1-base-4view
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
Our code is heavily based on the following works:
- [ThreeStudio](https://github.com/threestudio-project/threestudio): A clean and extensible codebase for 3D generation via Score Distillation.
- [MVDream](https://github.com/bytedance/MVDream): Used as one of our multi-view teachers.
- [RichDreamer](https://github.com/bytedance/MVDream): Serves as another multi-view teacher for normal and depth supervision.
- [3DTopia](https://github.com/3DTopia/3DTopia): Its text caption dataset is applied in our training and comparison.
- [DiffMC](https://github.com/SarahWeiii/diso): Our solution uses its differentiable marching cube for mesh rasterization.
- [NeuS](https://github.com/Totoro97/NeuS): We implement its SDF-based volume rendering for dual rendering in our solution.
- [ScaleDreamer](https://github.com/huggingface/ScaleDreamer): The current code is based on ScaleDreamer architecture for improved capabilities.

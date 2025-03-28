
<div align="center">
<h1>Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation beyond 3D Training Data</h1>

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
<a href='https://huggingface.co/spaces/ZhiyuanthePony/TriplaneTurbo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</div>


---

</div>


## ⏩ Updates

- **2025-03-13**: The source code and pretrained models are released.
- **2025-03-03**: Gradio and HuggingFace Demos are available.
- **2025-02-27**: TriplaneTurbo is accepted to CVPR 2025.

<!-- Features -->
## 🌟 Features
- **Fast Inference 🚀**: Our code excels in inference efficiency, capable of outputting textured mesh in around 1 second.
- **Text Comprehension 🆙**: It demonstrates strong understanding capabilities for complex text prompts, ensuring accurate generation according to the input.
- **3D-Data-Free Training 🙅‍♂️**: The entire training process doesn't rely on any 3D datasets, making it more resource-friendly and adaptable.

<!-- Updates -->

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
Install iNGP and NerfAcc
```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindingstorch
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
```
If you encounter errors while installing iNGP, it is recommended to check your gcc version. Follow these steps to change the gcc version within your -cconda environment. After that, return to the project directory and reinstall iNGP and NerfAcc:
```sh
conda install -c conda-forge gxx=9.5.0
cd  $CONDA_PREFIX/lib
ln -s  /usr/lib/x86_64-linux-gnu/libcuda.so./
cd <your project directory>
```

The README is currently under development. Please stay tuned for further updates.


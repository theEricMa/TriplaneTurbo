#!/bin/bash
# create a virtual environment
# 先卸载旧版本（确保完全清理）
pip uninstall -y setuptools pkg_resources

# 强制安装最新版 setuptools 和 pip
pip install --upgrade --force-reinstall "setuptools==68.2.2" "pip==24.0"

pip install "omegaconf==2.3.0"
pip install "opencv-python"
pip install "imageio>=2.28.0"
pip install "wandb"

echo "Installing PyTorch with CUDA support..."
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 \
--extra-index-url https://download.pytorch.org/whl/cu121
pip install "pytorch-lightning==2.0.0"

echo "Installing diffusers..."
# pip install "diffusers==0.29"
pip install "diffusers<0.20"

echo "Installing jaxtyping..."
pip install jaxtyping

echo "Installing typeguard..."
pip install typeguard

echo "Installing trimesh..."
pip install trimesh[easy]

echo "Installing networkx..."
pip install networkx

echo "Installing pysdf..."
pip install pysdf

echo "Installing PyMCubes..."
pip install PyMCubes

echo "Installing transformers..."
# pip install "transformers" 
pip install "transformers==4.36.1"


echo "Installing gradio..."
pip install "gradio==3.0.11"

echo "Installing einops..."
pip install einops

echo "Installing DISO..."
git clone https://github.com/SarahWeiii/diso
cd diso
python setup.py install
cd ..
rm -rf diso


echo "Installing huggingface_hub..."
pip install "huggingface_hub==0.20.0"


echo "Installing numpy..."
pip install "numpy==1.26.4"

echo "Finished installing all dependencies!"
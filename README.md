# TriplaneTurbo: Instant Text-to-3D Generator from Multiple 2D Diffusion Priors
## Paper | Project Page

## ⚙️ Dependencies and Installation

<details>
<summary> Follow threestudio to set up the conda environment, or use our provided instructions as below. </summary>
 
- Create a virtual environment:

```sh
conda create -n scaledreamer python=3.10
conda activate scaledreamer
```
- Add plugin

```sh
mkdir custom
cd custom
git clone https://github.com/87003697/amortized
cd ..
```

- Install PyTorch
```sh
# Prefer using the latest version of CUDA and PyTorch 
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
- (Optional, Recommended) Install [xFormers](https://github.com/facebookresearch/xformers) for attention acceleration.
```sh
conda install xformers -c xformers
```
- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install major dependencies:

```sh
pip install -r requirements.txt
```
- Install [iNGP](https://github.com/NVlabs/instant-ngp) and [NerfAcc](https://github.com/nerfstudio-project/nerfacc):

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
</details>

<details>
<summary> Download 2D Diffusion Priors. </summary>
 
- Save [SD-v2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and [MVDream](https://mv-dream.github.io/) to the local directory `pretrained`.
 
```
python scripts/download_pretrained_models.py
```
</details>

## 🕹️ Experiments

### Git update

Use the following code to update the code to the latest version
```sh
git pull
cd custom/amortized
git pull
cd ../..
```

### Group-0 08.08.2024
```sh
sh scripts/group_0/exp1_3DTopia_step_scheduler_4_triple_16_vanilla_16_bias_lora_prompt_60k.sh
```

```sh
sh scripts/group_0/exp2_3DTopia_step_4_triple_16_vanilla_16_bias_lora_prompt_60k.sh
```
Each experiment lasts for around 5 days.
Results are saved in `outputs_group_0` directory.

### Group-1 09.05.2024

Update with the following step:

1. Please git update first. Run `git pull` in two directories, as explained ⬆️.

2. Please prepare sd-turbo and sd-v1.5 into `pretrained` directory.
  
3. Run the following script to download another diffusion prior
```sh
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd_mv_ema.ckpt -O ./pretrained/nd_mv_ema.ckpt
```
Now the files in `pretrained` directory is as follows. A little bit messy but will be refined in the future.
```
pretrained
├── sd-turbo
├── stable-diffusion-v1-5
├── stable-diffusion-2-1-base
├── nd_mv_ema.ckpt
├── sd-v2.1-base-4view
```

Two experiments are as follows.

```sh
sh scripts/group_1/exp1_3DTopia__base_step_4__asd_mv+rd+sd_volsdf+cpumc__triple_16_vanilla_16_bias_lora_prompt.sh
```

```sh
sh scripts/group_1/exp2_3DTopia__turbo_step_4__asd_mv+rd+sd_volsdf+cpumc__triple_16_vanilla_16_bias_lora_prompt.sh
```
Each experiment lasts for around 7 days.
Results are saved in `outputs_group_1` directory.


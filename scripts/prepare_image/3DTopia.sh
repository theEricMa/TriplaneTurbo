
export CUDA_VISIBLE_DEVICES=$1
group_total=$2
group_index=$3

python load/sdxl_animation_img_generation_v2.py \
    --prompt_libary /home/zhiyuan_ma/code/ScaleDreamer_v1/load/3DTopia_361k_prompt_library.json \
    --group_total $group_total \
    --group_index $group_index \
    --save_dir ./datasets_real/Images_3DStyle_3DTopia \
    --num_repeat 2
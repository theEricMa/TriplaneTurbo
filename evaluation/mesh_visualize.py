
import os
import glob
import argparse
from multiprocessing import Pool

def process_file(args):
    file, gpu_id, save_dir, num_views, requires_normal = args
    name = file.split('/')[-2]
    save_path = os.path.join(save_dir, name)

    if os.path.exists(save_path):
        os.system(f'rm -rf {save_path}')

    if requires_normal:
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} kire {file} --front_dir '\\+y' "
            f"--save {save_path} --wogui --mode normal --num_azimuth {num_views} --H 512 --W 512 --elevation '-15' --force_cuda_rast"
        )
        os.system(command)
        # rename all imgs in the directory
        # make a temp directory
        temp_normal_dir = os.path.join(save_path, 'temp_normal')
        os.makedirs(temp_normal_dir, exist_ok=True)
        # move all imgs to the temp directory
        os.system(f'mv {save_path}/*.png {temp_normal_dir}')
        # rename all imgs in the temp directory
        all_imgs = glob.glob(f'{temp_normal_dir}/*')
        all_imgs.sort()
        for i, img in enumerate(all_imgs):
            num_view = int(
                # int(img.split('_')[-1].split('.')[0]) \
                (int(img.split('_')[-1].split('.')[0]) - 90) % 360 \
                    / 360 * num_views
            )
            os.system(f'mv {img} {temp_normal_dir}/normal_{num_view}.png')

    command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} kire {file} --front_dir '\\+y' "
        f"--save {save_path} --wogui --num_azimuth {num_views} --H 512 --W 512 --elevation '-15' --force_cuda_rast"
    )
    os.system(command)
    # rename all imgs in the directory
    # make a temp directory
    temp_color_dir = os.path.join(save_path, 'temp_color')
    os.makedirs(temp_color_dir, exist_ok=True)
    # move all imgs to the temp directory, except for the normal imgs
    os.system(f'mv {save_path}/*.png {temp_color_dir}')
    # rename all imgs in the temp directory
    all_imgs = glob.glob(f'{temp_color_dir}/*')
    all_imgs.sort()
    for i, img in enumerate(all_imgs):
        num_view = int(
            # int(img.split('_')[-1].split('.')[0]) \
            (int(img.split('_')[-1].split('.')[0]) - 90) % 360 \
                / 360 * num_views
        )
        os.system(f'mv {img} {temp_color_dir}/rgb_{num_view}.png')

    # merge the two directories
    if requires_normal:
        os.system(f'mv {temp_normal_dir}/*.png {save_path}')
        os.system(f'rm -r {temp_normal_dir}')
    os.system(f'mv {temp_color_dir}/*.png {save_path}')
    os.system(f'rm -r {temp_color_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', default='workspace', type=str)
    parser.add_argument('--gpus', default='0', type=str, help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--save_dir', default='ours_dreamfusion_objs', type=str)
    parser.add_argument('--num_views', default=4, type=int)
    parser.add_argument('--requires_normal', default=False, type=bool)
    args = parser.parse_args()

    files = glob.glob(f'{args.dir}/*/*.obj')
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse GPU IDs
    gpu_ids = args.gpus.split(',')

    # Create a list of arguments for each file, cycling through the GPU IDs
    tasks = [(file, gpu_ids[i % len(gpu_ids)], args.save_dir, args.num_views, args.requires_normal) for i, file in enumerate(files)]

    # # for debugging
    # for task in tasks:
    #     process_file(task)

    with Pool() as pool:
        pool.map(process_file, tasks)
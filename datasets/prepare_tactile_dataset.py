import os
import glob
import pandas as pd

import numpy as np
from PIL import Image
import imageio

import logging
log = logging.getLogger(__name__)

def load_seq_from_file(csvfile):

    seq_data = pd.read_csv(csvfile) if os.path.exists(csvfile) else None

    if seq_data is not None:
        seq_data = None if (len(seq_data) < 2) else seq_data

    return seq_data

def main():

    src_base_dir = '/home/paloma/code/fair_ws/tacto'
    dst_base_dir = '/home/paloma/code/lib_ws/pytorch-CycleGAN-and-pix2pix'
    
    dataset_types = ['train', 'test']
    dataset_name = 'tacto_rolling_die_0.45'

    dir_A = f"{dst_base_dir}/local/datasets/{dataset_name}/A/"
    dir_B = f"{dst_base_dir}/local/datasets/{dataset_name}/B/"
    dir_AB = f"{dst_base_dir}/local/datasets/{dataset_name}/AB/"

    # iterate over train/test data
    for dataset_type in dataset_types:

        os.makedirs(f"{dir_A}/{dataset_type}", exist_ok=True)
        os.makedirs(f"{dir_B}/{dataset_type}", exist_ok=True)
        os.makedirs(f"{dir_AB}/{dataset_type}", exist_ok=True)

        seq_dirs = sorted(glob.glob(f"{src_base_dir}/local/datasets/{dataset_name}/{dataset_type}/*"))
        dst_img_idx = 0

        # iterate over contact sequences
        for seq_dir in seq_dirs:
            seq_data = load_seq_from_file(f"{seq_dir}/poses_imgs.csv")

            if seq_data is None:
                continue

            img_color_locs = seq_data[f"img_top_color_loc"]
            img_normal_locs = seq_data[f"img_top_normal_loc"]

            # iterate over images within each sequence
            for img_idx in range(0, len(img_color_locs)):
                img_color = Image.open(f"{src_base_dir}/{img_color_locs[img_idx]}")
                img_normal = Image.open(f"{src_base_dir}/{img_normal_locs[img_idx]}")

                imageio.imwrite(f"{dir_A}/{dataset_type}/{dst_img_idx}.png", img_color)
                imageio.imwrite(f"{dir_B}/{dataset_type}/{dst_img_idx}.png", img_normal)

                dst_img_idx = dst_img_idx + 1

    os.system(f"python {dst_base_dir}/datasets/combine_A_and_B.py --fold_A {dir_A} --fold_B {dir_B} --fold_AB {dir_AB}")

    log.info(f"Created tactile dataset of {dst_img_idx} images at {dst_base_dir}/local/datasets/{dataset_name}.")

if __name__=='__main__':
    main()
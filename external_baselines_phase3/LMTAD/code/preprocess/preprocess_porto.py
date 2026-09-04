"""
This code is based on the following github: https://github.com/liuyiding1993/ICDE2020_GMVSAE/blob/master/preprocess/preprocess.py
"""
import os
from collections import defaultdict
import json
import argparse


import numpy as np
import pandas as pd
from tqdm import tqdm

def height2lat(height):
    return height / 110.574


def width2lng(width):
    return width / 111.320 / 0.99974


def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']


def main(
        data_file_path,
        out_dir,
        grid_height,
        grid_width,
        boundary,
        min_traj_length,
        override=False,
):
    
    """
    preprocess the proto dataset
    """
    lat_size, lng_size = height2lat(grid_height), width2lng(grid_width)

    lat_grid_num = int((boundary['max_lat'] - boundary['min_lat']) / lat_size) + 1
    lng_grid_num = int((boundary['max_lng'] - boundary['min_lng']) / lng_size) + 1
    
    # import ipdb
    # ipdb.set_trace()
    trajectories = pd.read_csv(data_file_path, header=0, index_col="TRIP_ID")

    processed_trajectories = defaultdict(list)

    shortest, longest = 20, 1200
    valid_trajectory_count = 0
    
    for i, (idx, traj) in enumerate(tqdm(trajectories.iterrows(), total=trajectories.shape[0], desc="process trajectories")):

        current_traj_size = 0
        grid_seq = []
        valid = True
        polyline = eval(traj["POLYLINE"])
        if shortest <= len(polyline) <= longest:
            for lng, lat in polyline:
                if in_boundary(lat, lng, boundary):
                    grid_i = int((lat - boundary['min_lat']) / lat_size)
                    grid_j = int((lng - boundary['min_lng']) / lng_size)
                    token = grid_i * lng_grid_num + grid_j
                    grid_seq.append(token)
                else:
                    valid = False
                    break

            if valid:
                s, d = grid_seq[0], grid_seq[-1]
                processed_trajectories[(s, d)].append(grid_seq)
                valid_trajectory_count +=1


    print("Valid trajectory num:", valid_trajectory_count)
    print("Grid size:", (lat_grid_num, lng_grid_num))

    os.makedirs(out_dir, exist_ok=True)
    fout = open(f"{out_dir}/porto_processed.csv", "w")

    saved_trajectory_count = 0
    max_trajectory_size = 0
    for trajs in tqdm(processed_trajectories.values(), desc="filter and save valid trajectories"):
        if len(trajs) >= min_traj_length:
            for traj in trajs:
                fout.write(f"{traj}\n")
                saved_trajectory_count += 1

                if len(grid_seq) > max_trajectory_size:
                    max_trajectory_size = len(grid_seq)
    
    print("saved trajectory num:", saved_trajectory_count)
    print(f"max trajectory size: {max_trajectory_size}")

    if not os.path.exists(f"{out_dir}/vocab.json") or override:
        vocab = {}
        maximum_token_number = lat_grid_num * lng_grid_num + lng_grid_num
        for label in range(0, maximum_token_number + 1):
            vocab[str(label)] = label

        vocab["PAD"] = len(vocab)
        vocab["EOT"] = len(vocab) #end of trajectory
        vocab["SOT"] = len(vocab) #start of trajectory
        
        with open(f"{out_dir}/vocab.json", "w", encoding="utf-8") as fp:
            json.dump(vocab, fp)

    # generate outliers
    from datasets import PortoConfig, PortoDataset

    """
    outlier_level: int = 3
    outlier_prob: float = 0.3
    outlier_ratio: float = 0.05
    """
    outlier_configs = [
        [0.05, 3, 0.1],
        [0.05, 3, 0.3],
        [0.05, 5, 0.1]
    ]

    for outlier_config in outlier_configs:

        dataset_config = PortoConfig(
            outlier_ratio=outlier_config[0],
            outlier_level=outlier_config[1],
            outlier_prob=outlier_config[2],
            include_outliers=False
        )
        dataset = PortoDataset(dataset_config)
        dataset.generate_outliers()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--min_traj_length", type=int, default=25)
    parser.add_argument("--grid_height", type=float, default=0.1) # 100m
    parser.add_argument("--grid_width", type=float, default=0.1) # 100m
    parser.add_argument("--out_dir", type=str, default="./data/porto")

    args = parser.parse_args()

    BOUNDARY = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}

    main(
        data_file_path= args.data_dir,
        out_dir=args.out_dir,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        boundary=BOUNDARY,
        min_traj_length=args.min_traj_length,
        override=False
    )

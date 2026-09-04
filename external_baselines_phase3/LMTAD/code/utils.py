import random
import numpy as np
import torch

def seed_all(seed):
    """set the random seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def log(s, path, print_=True):
    """loging functionalitoes"""
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')

# language model training

def save_file_name_pattern_of_life(features):
    """get the save file name for the pattern of life dataset"""
    
    file_name = "features"
    if "gps" in features:
        file_name += "_gps"
    if "distance" in features:
        file_name += "_distance"
    if "duration" in features:
        file_name += "_duration"
    if "place" in features:
        file_name += "_place"

    return file_name

def save_file_name_trial0(features):
    """get the save file name for the trial0 dataset"""
    
    file_name = "features"
    if "gps" in features:
        file_name += "_gps"
    if "distance" in features:
        file_name += "_distance"
    if "duration" in features:
        file_name += "_duration"
    if "agent_id" in features:
        file_name += "_user_id"

    return file_name

def save_file_new_datset(features):
    """get the save file name for the pattern of life dataset"""
    
    file_name = f"features"
    if "user_id" in features:
        file_name += "_user_id"
    if "dayofweek" in features:
        file_name += "_dayofweek"
    if "gps" in features:
        file_name += "_gps"
    if "distance" in features:
        file_name += "_distance"
    if "duration" in features:
        file_name += "_duration"
    if "place" in features:
        file_name += "_place"

    return file_name


def save_file_name_porto(config):
    """get the save file name for the porto dataset"""
    
    if config.include_outliers:
         file_name = f"outliers_{'_'.join(config.outliers_list)}_ratio_{config.outlier_ratio}_level_{config.outlier_level}_prob_{config.outlier_prob}"
    else:
        file_name = ""

    return file_name
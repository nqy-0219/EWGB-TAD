import os
from dataclasses import dataclass, field
from typing import Any, List, Union
import json
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import log

class VocabDictionary(object):
    """
    dictionary to map trajectory semantics to tokens
    """

    def __init__(self, vocab_file_path) -> None:

        with open(vocab_file_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.reverse_map_vocab = {value:item for item, value in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def encode(self, trajectory:Union[List[str], List[int]]):
        """
        encode a trajectory into token
        """
        tokens = []

        # pdb.set_trace()
        for string in trajectory:
            tokens.append(self.vocab[str(string)])
            

        return tokens
    
    def decode(self, tokens:List[int]):
        """
        decode a trajectory into token
        """
        trajectory = []
        for token in tokens:
            trajectory.append(self.reverse_map_vocab[token])

        return trajectory

    def pad(self):
        return "PAD"
    def eot(self):
            return "EOT"
    def pad_token(self):
        return self.vocab[self.pad()]
    def eot_token(self):
        return self.vocab[self.eot()]


@dataclass
class POLConfig:
    """
    dataclass for semantic trajectory
    """
    data_dir: str = "./data/work-outliers/checkin-atl"
    file_name: str = "data"
    features: List = field(default_factory=lambda: ["gps", "distance", "duration", "place"])
    block_size: int = 64 # length of maximum daily trajectories of a user
    grid_leng: int = 25 # the size of a cell in a grid
    include_outliers: bool = False
    outlier_days: int= 14
    # include_outliers: bool= False
    log_file: str = ""
    logging: bool = True
    start_time: int= 4

class POLDataset(Dataset):
    """
    semantic trajectory datset
    """
    def __init__(self, config: POLConfig) -> None:
        super().__init__()

        self.config = config

        dictionary_path = self.get_dictionary_path()

        self.dictionary = VocabDictionary(dictionary_path)
        file_path = os.path.join(self.config.data_dir, f"{self.config.file_name}_grouped.tsv")

        self.data, self.outliers = self.get_data(file_path)

        # pdb.set_trace()
        
    def get_data(self, file_path):
        """
        get all the data
        """

        message=f"loading the data..."
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        data = pd.read_csv(file_path, delimiter="\t")
        
        data["date_formated"] = pd.to_datetime(data["date"])
       
        outlier_list = [546, 644, 347, 62, 551, 992, 554, 949, 900, 57] #TO DO can load this data from a file

        message=f"outliers: {outlier_list}"
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        data["user_id_int"] = data["user_id"].str.split("_").str[-1].astype(int)
        # grouped = data.groupby(by=["user_id", "date"]).agg(list).reset_index()
        outliers = data[(data["date_formated"] > (data["date_formated"].max() - pd.DateOffset(self.config.outlier_days)))].copy()
        outliers = outliers.loc[outliers['user_id_int'].isin(outlier_list)][["user_id_int", "date"]]
        outliers["outlier"] = "outlier"

        data= pd.merge(data, outliers,  how='left', left_on=['user_id_int','date'], right_on = ['user_id_int','date'])
        data['outlier'] = np.where(data.outlier.notnull(), "outlier", "non outlier")

        if not self.config.include_outliers:
            data = data[data["outlier"] != "outlier"]

        message=f"inlcude outliers: {self.config.include_outliers}"

        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        # times len(self.config.features) because we may have more than one feature to include in the vector
        # plus 3 to account for the user_id, dayofweek and EOT
        self.config.block_size = (data.token.str.len().max() * len(self.config.features)) + 3

        message=f"context size: {self.config.block_size }"

        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        return data, outliers
    
    def get_dictionary_path(self):
        """get vocab file name"""
        file_name = "vocab"
        if "gps" in self.config.features:
            file_name += "_gps"
        if "distance" in self.config.features:
            file_name += "_distance"
        if "duration" in self.config.features:
            file_name += "_duration"
        if "place" in self.config.features:
            file_name += "_place"

        file_name += ".json"

        path = os.path.join(self.config.data_dir, file_name)

        return path
    def get_outliers(self):
        return self.outliers
    
    def partition_dataset(self, proportion=0.9, seed=123):
        np.random.seed(seed)
        train_num = int(len(self) * proportion)
        indices = np.random.permutation(len(self))
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices
     
    def __len__(self):
        return len(self.data)
    
    def get_samples_for_user(self, user_id):
        """
        get the samples of a particular user given their user_id
        """
        indices = self.data[self.data["user_id"] == f"user_{user_id}"].index.tolist()
        samples = []
        for i in indices:
            samples.append(self.__getitem__(i))

        return samples

    def get_feature_vector(self, sample:pd.core.series.Series):
        """
        generate a feature vector for a given sample
        """

        daily_trajectory_feature = [sample["user_id"], eval(sample["dayofweek"])[0]]

        # [user_id, dayofweek, place, token, duration_bucket, distance, place, token, duration_bucket, distance, ..., EOT]
        # [user_35, day_4, Workplace, 88, 0-60, near, Restaurant, 88, 0-60, near, ..., EOT]

        places = eval(sample.place)
        tokens = eval(sample.token)
        duration_buckets = eval(sample.duration_bucket)
        distances = eval(sample.distance_label)

        for place, token, duration_bucket, distance in zip(places, tokens, duration_buckets, distances):
            current_seq = []

            if "place" in self.config.features:
                current_seq.append(place)

            if "gps" in self.config.features:
                current_seq.append(str(token))

            if "duration" in self.config.features:
                current_seq.append(duration_bucket)

            if "distance" in self.config.features:
                current_seq.append(distance)
            
            daily_trajectory_feature.extend(current_seq)
        
        daily_trajectory_feature.append("EOT")
        return daily_trajectory_feature
    
    def get_all_data(self):

        data = defaultdict(list)
        for i in tqdm(range(self.data.shape[0])):

            sample = self.data.iloc[i]
            daily_trajectory_feature = self.get_feature_vector(sample)

            data["user_id_int"].append(sample.user_id_int)
            data["date"].append(sample.date)
            data["feature"].append(daily_trajectory_feature)

            # pdb.set_trace()

        file_name = "data_with_features"
        if "gps" in self.config.features:
            file_name += "_gps"
        if "distance" in self.config.features:
            file_name += "_distance"
        if "duration" in self.config.features:
            file_name += "_duration"
        if "place" in self.config.features:
            file_name += "_place"

        file_name += ".tsv"

        data_df = pd.DataFrame(data)
        data_df.to_csv(f"{self.config.data_dir}/{file_name}", sep="\t", index=False)

    def __getitem__(self, index) -> Any:

        # pdb.set_trace()
        sample  = self.data.iloc[index]
        daily_trajectory_feature = self.get_feature_vector(sample)   
        tokens = self.dictionary.encode(daily_trajectory_feature)
        metadata = [sample.user_id, sample.date, sample.outlier]

        # ipdb.set_trace()
        return (metadata,  tokens)
    
    def collate(self, data):
        """
        collate function
        """
        # max_length = len(max(token_lists, key=len))
        masks = []
        token_lists = []
        all_metadata = []
        
        # start_time = time.time()
        max_lenth = max([len(item[-1]) for item in data])
        for metadata, tokens in data:

            mask = [1] * len(tokens) + [0] * (max_lenth - len(tokens))
            tokens_ = tokens +  (max_lenth - len(tokens)) * [self.dictionary.pad_token()]
            token_lists.append(tokens_)
            masks.append(mask)
            all_metadata.append(metadata)

        token_lists = torch.tensor(token_lists)
        masks = torch.tensor(masks)

        # print(f"to batchify it took {time.time() - start_time}")
        return {
            "data" : token_lists,
            "mask": masks,
            "metadata": all_metadata
        }
  

@dataclass
class PortoConfig:
    """
    dataclass for the porto taxi dataset
    """
    data_dir: str = "./data/porto"
    file_name: str = "porto_processed"
    grip_size: List = field(default_factory=lambda: (51, 158))
    data_split: str = None
    block_size: int = 1186 # length of maximum trajectory
    outlier_level: int = 3
    outlier_prob: float = 0.3
    outlier_ratio: float = 0.05
    outliers_list: List = field(default_factory=lambda: ["route_switch"])
    include_outliers: bool = True

class PortoDataset(Dataset):
    """
    semantic trajectory datset
    """
    def __init__(self, config: PortoConfig) -> None:
        super().__init__()

        self.config = config
        dictionary_path = os.path.join(self.config.data_dir, "vocab.json")
        self.dictionary = VocabDictionary(dictionary_path)
        file_path = os.path.join(self.config.data_dir, f"{self.config.file_name}.csv")

        self.data, self.metadata = self.get_data(file_path)
        # pdb.set_trace()
        
    def get_data(self, file_path):
        """
        get all the data
        """

        print(f"loading the dataset ...")
        # pdb.set_trace()
        trajectories = []
        labels = []
        sizes = []
        i = 0
        for traj in tqdm(open(file_path, 'r').readlines()):
            traj = eval(traj)
            trajectories.append(traj)
            labels.append("non outlier")
            sizes.append(len(traj))

            # if i > 2000:
            #     break
            # i+=1
        
        sizes = np.array(sizes)
        self.config.block_size = sizes.max() + 2 # to account for EOT and SOT 
        # pdb.set_trace()
        outlier_counts = 0
        skipped_long_trajectories = 0
        if self.config.include_outliers:
            print("loading outliers")
            # add outliers
            for key, values in self.get_outliers().items():
                label =""
                if key == "route_switch":
                    label = "route switch outlier"
                elif key == "detour":
                    label =  "detour outlier"

                for traj in values:

                    if len(traj) <= self.config.block_size - 2:
                        trajectories.append(traj)
                        labels.append(label)
                        outlier_counts += 1
                    else:
                        skipped_long_trajectories += 1

        # sizes.sort()
        
        print(f"total number of outliers: {outlier_counts}")
        print(f"number of spkipped trajectories: {skipped_long_trajectories}")
        print(f"context size {self.config.block_size}")

        # pdb.set_trace()
        sorted([trajectories], key=lambda k: len(k))
        return trajectories, labels
    def get_outliers(self):
        """
        load saved outliers
        """
        outliers = {}
        
        for outlier_type  in self.config.outliers_list:
          
            file = f"{self.config.data_dir}/outliers/{outlier_type}_ratio_{self.config.outlier_ratio}_level_{self.config.outlier_level}_prob_{self.config.outlier_prob}.csv"

            try:
                route_switched_outliers = open(file, 'r').readlines()
                outliers[outlier_type] = [eval(traj) for traj in route_switched_outliers]
            except Exception as e:
                raise Exception(f"the file {file} cannot be found")
            print(f"loaded {outlier_type} outliers")
        return outliers
        
    def generate_outliers(self):

        """

        TO DO: move this in the data preprocessing step
        generated outliers
        """
        outliers = {}
        trajectory_count = len(self)

        # route swithing outliers
        np.random.seed(0)
        route_swithing_idx = np.random.randint(0, trajectory_count, size=int(trajectory_count * self.config.outlier_ratio))
        # [199340,  43567, 173685, ..., 150926, 233238, 224962]
        outliers["route_switch"] = self.get_route_switch_outliers(
            [self.data[idx] for idx in route_swithing_idx],level=self.config.outlier_level, prob=self.config.outlier_prob)
        
        np.random.seed(10)
        detour_idx = np.random.randint(0, trajectory_count, size=int(trajectory_count * self.config.outlier_ratio))
        # [ 83209, 236669,  94735, ...,  97329, 173664,  83412]
        outliers["detour"] = self.get_detour_outliers([self.data[idx] for idx in detour_idx],
                                      level=self.config.outlier_level, prob=self.config.outlier_prob, vary=False)
        # pdb.set_trace()

        save_dir = f"{self.config.data_dir}/outliers"
        os.makedirs(f"{save_dir}", exist_ok=True)
        for key, values in outliers.items():
            current_save_dir = \
                f"{save_dir}/{key}_ratio_{self.config.outlier_ratio}_level_{self.config.outlier_level}_prob_{self.config.outlier_prob}.csv"
            
            print(f"saved outlier file {current_save_dir}")
            
            fout = open(current_save_dir, "w")
            for traj in values:
                fout.write(f"{traj}\n")
        

        return outliers
        
    def get_route_switch_outliers(self, batch_x, level, prob):
        """
        get route swithing outliers
        """
        outliers = []
        for traj in batch_x:
            outliers.append([traj[0]] + [self._perturb_point(p, level)
                                 if not p == 0 and np.random.random() < prob else p
                                 for p in traj[1:-1]] + [traj[-1]])
        return outliers
    
    def _perturb_point(self, point, level, offset=None):
        """
        -
        """
        map_size = self.config.grip_size
        x, y = int(point // map_size[1]), int(point % map_size[1])
        if offset is None:
            offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
            x += x_offset * level
            y += y_offset * level
        return int(x * map_size[1] + y)
    
    def get_detour_outliers(self, batch_x, level, prob, vary=False):
        map_size = self.config.grip_size
        outliers = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for traj in batch_x:
            anomaly_len = int((len(traj) - 2) * prob)
            anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            offset = [int(traj[anomaly_st_loc] // map_size[1]) - int(traj[anomaly_ed_loc] // map_size[1]),
                      int(traj[anomaly_st_loc] % map_size[1]) - int(traj[anomaly_ed_loc] % map_size[1])]
            if offset[0] == 0: div0 = 1
            else: div0 = abs(offset[0])
            if offset[1] == 0: div1 = 1
            else: div1 = abs(offset[1])

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            outliers.append(traj[:anomaly_st_loc] +
                                 [self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])
        return outliers
    
    def partition_dataset(self, proportion=0.9, seed=123):
        np.random.seed(seed)
        train_num = int(len(self) * proportion)
        indices = np.random.permutation(len(self))
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices
     
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:

        sample  = self.data[index]
        sample = ["SOT"] + sample + ["EOT"]
        metadata = self.metadata[index]
        return sample, metadata
    
    def collate(self, data):
        """
        collate function
        """
        masks = []
        token_lists = []
        metadatas = []
        
        max_lenth = max([len(item[0]) for item in data])
        for tokens_, metadata in data: 
            
            mask = [1] * len(tokens_) + [0] * (max_lenth - len(tokens_))
            tokens = self.dictionary.encode(tokens_ + [self.dictionary.pad()] * (max_lenth - len(tokens_)))

            # pdb.set_trace()
            token_lists.append(tokens)

            masks.append(mask)
            metadatas.append(metadata)

        token_lists = torch.tensor(token_lists)
        masks = torch.tensor(masks)
        return {
            "data" : token_lists,
            "mask": masks,
            "metadata": metadatas
        }


"""
preprocessing code for the pattern of life dataset
"""
import os
from collections import defaultdict
import json
from typing import List
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

# def save_outlier_user_id(file_name:str, out_dir:str) -> None:
#     """
#     save the ouliers user ids
#     """
#     out_file_name = f"{out_dir}/{file_name.split('/')[-2]}.tsv"

#     if not os.path.exists(f"{out_dir}/{file_name}"):
#         with open(file_name, "r", encoding="utf-8") as file:
#                 info = file.read().splitlines()

#         info = [item.strip() for item in info if item.strip() != ""]

#         outliers = {}

#         for item in info[-3:]:            
#             outlier_severity, outlier_ids = item.split(":")
#             outlier_ids = [user_id.strip() for user_id in outlier_ids.split(",")]
#             outliers[outlier_severity] = outlier_ids
        
#         df_to_save = pd.DataFrame(outliers)
#         df_to_save.to_csv(
#             f"{out_dir}/{file_name.split('/')[-2]}.tsv",
#             index=False,
#             sep="\t"
#         )
#     else:
#         outliers = pd.read_csv(outliers, sep="\t")

#     return outliers

# def save_vocab(data:pd.DataFrame, out_dir:str, outlier_type:str):
#     """
#     save the vocabulary
#     """
#     # create vocabulary
#     vocab = {}
#     vocab["PAD"] = 0
#     vocab["EOT"] = 1 #end of trajectory
    
#     # tokens tokens
#     for t_idx in range(2, data["token"].max() + 1):
#         vocab[t_idx] = t_idx
    
#     # day of week tokens
#     current_idx = len(vocab)#next staring
#     for day in data["CheckinTime"].dt.dayofweek.unique():
#         vocab[f"day_{day}"] = current_idx
#         current_idx += 1

#     # agent token
#     current_idx = len(vocab)#next staring
#     for user_id in data["UserId"].unique():
#         vocab[user_id] = current_idx
#         current_idx += 1
    
#     # venue type tokens
#     current_idx = len(vocab)
#     for venue in data["VenueType"].unique():
#         vocab[venue] = current_idx
#         current_idx += 1

#     with open(f"{out_dir}/{outlier_type}_vocab.json", "w", encoding="utf-8") as fp:
#         json.dump(vocab, fp)

def generate_vocabulary(data:pd.DataFrame, out_dir, features:List):
    """
    generate the vocabulary
    """
    # TODO parametarize the well
    gps=True
    place=True
    distance=True
    duration=True

    # pdb.set_trace()

    # create vocabulary
    vocab = {}
    vocab["PAD"] = 0
    vocab["EOT"] = 1 #end of trajectory
    file_name = "vocab"
    if "gps" in features:
        file_name = f"{file_name}_gps"
        # tokens tokens
        for t_idx in range(2, data["token"].max() + 1):
            vocab[str(t_idx)] = t_idx
    
    # day of week tokens
    current_idx = len(vocab)#next staring
    for day in data["dayofweek"].unique():
        vocab[day] = current_idx
        current_idx += 1

    # agent token
    current_idx = len(vocab)#next staring
    for user_id in data["user_id"].unique():
        vocab[str(user_id)] = current_idx
        current_idx += 1
    
    if "distance" in features:
        file_name = f"{file_name}_distance"
        # distance type tokens
        current_idx = len(vocab)
        for d in data["distance_label"].unique():
            vocab[d] = current_idx
            current_idx += 1

    if "duration" in features:
        file_name = f"{file_name}_duration"
        # duration type tokens
        current_idx = len(vocab)
        for place in data["duration_bucket"].unique():
            vocab[place] = current_idx
            current_idx += 1
    # pdb.set_trace()
    if "place" in features:
        file_name = f"{file_name}_place"
        # place tokens
        current_idx = len(vocab)
        for d in data["place"].unique():
            vocab[d] = current_idx
            current_idx += 1

    # pdb.set_trace()    
    with open(f"{out_dir}/{file_name}.json", "w", encoding="utf-8") as fp:
        json.dump(vocab, fp)

    # pdb.set_trace()

def pattern_life_file_preprocess(data_dir:str,
                                outlier_type:str,
                                file_name:str,
                                grid_length:int,
                                out_dir:str,
                                features:List,
                                train_ratio:int=0.9,
                                override:bool=False):
    """
    preprocess file from the pattern of life simulartion simulation data
    """

    file_path = os.path.join(data_dir, outlier_type, file_name)
    info_file = os.path.join(data_dir, outlier_type, "info.txt")
    out_dir = os.path.join(out_dir, outlier_type, file_name.split(".")[0])
    os.makedirs(out_dir, exist_ok=True)
    
    data = pd.read_csv(file_path, delimiter="\t")
    data['CheckinTime'] = pd.to_datetime(data['CheckinTime'])
    # scalling up X_d and Y_d so that the lowest token will be 2.
    # this is done in order to free up positions 0 and 1 in our vocab
    # we reserved these positions for <PAD> and <EOS> tokens
    data["X_d"] =  ((data["X"] - data["X"].min()) // grid_length).astype('int64') + 1
    data["Y_d"] =  ((data["Y"] - data["Y"].min()) // grid_length).astype('int64') + 1
    data["token"] = data["X_d"]  + data["Y_d"]

    # pdb.set_trace()
    # override = True
    if not os.path.exists(f"{out_dir}/data.tsv") or override:
        
        # group daily trajectories
        data_dict = defaultdict(list)

        # idx = 0
        # split by agent
        i = 0
        for user_id, user_group in tqdm(data.groupby('UserId')):
            # sort by datetime
            user_group = user_group.sort_values(by='CheckinTime')

            for date, group in user_group.groupby(user_group.CheckinTime.dt.date):
                new_group = group.reset_index(drop=True).copy()
                new_group["to"] = pd.concat([group.CheckinTime[1:], group.CheckinTime[-1:]]).reset_index(drop=True).copy()
                new_group["to_X"] = pd.concat([group.X[1:], group.X[-1:]]).reset_index(drop=True).copy()
                new_group["to_Y"] = pd.concat([group.Y[1:], group.Y[-1:]]).reset_index(drop=True).copy()
                new_group["duration"] = ((new_group.to - new_group.CheckinTime) / pd.Timedelta(minutes=1)).astype(int)
                new_group["distance"] = np.sqrt(np.square(new_group.to_X - new_group.X) + np.square(new_group.to_Y - new_group.Y))
                
                for row in new_group.iterrows():

                    data_dict["user_id"].append(f"user_{user_id}")
                    data_dict["user_id_int"].append(user_id)
                    data_dict["date"].append(str(date))
                    data_dict["dayofweek"].append(f"day_{group.iloc[0].CheckinTime.dayofweek}")
                    data_dict["place"].append(row[1]["VenueType"])
                    data_dict["X"].append(row[1]["X"])
                    data_dict["Y"].append(row[1]["Y"])
                    data_dict["X_d"].append(row[1]["X_d"])
                    data_dict["Y_d"].append(row[1]["Y_d"])
                    data_dict["token"].append(row[1]["X_d"]  + row[1]["Y_d"])
                    data_dict["duration"].append( row[1]["duration"])
                    data_dict["distance"].append(row[1]["distance"])
                    data_dict["CheckinTime"].append(row[1]["CheckinTime"])

                    # pdb.set_trace()
            # if i == 5:
            #     break
            # i += 1
        
        df_to_save = pd.DataFrame(data_dict)

        mean_distance = df_to_save.distance.sort_values().unique().mean()
        df_to_save["distance_label"] = np.where(df_to_save["distance"] < mean_distance, "near", "far")

        df_to_save["duration_label"] = df_to_save["duration"].apply(get_duration_range, label="label")
        df_to_save["duration_bucket"] = df_to_save["duration"].apply(get_duration_range, label="bucket")

        df_to_save["CheckinTime"] = pd.to_datetime(df_to_save["CheckinTime"])
        df_to_save["time"] = df_to_save['CheckinTime'].dt.time
        data_filtered = df_to_save[df_to_save["time"] >=  pd.to_datetime([f"4:00:00"]).time[0]].copy()
        
        print(f"{df_to_save.shape[0] - data_filtered.shape[0]} daily trajectories were filtered out")
        grouped = data_filtered.groupby(by=["user_id", "date"]).agg(list).reset_index()

        # pdb.set_trace()
        df_to_save.to_csv(
            f"{out_dir}/data.tsv",
            index=False,
            sep="\t"
        )

        grouped.to_csv(
            f"{out_dir}/data_grouped.tsv",
            index=False,
            sep="\t"
        )
    else:
        df_to_save = pd.read_csv(f"{out_dir}/data.tsv", delimiter="\t")

    generate_vocabulary(df_to_save, out_dir, features)
    return df_to_save


def get_duration_range(x, label):

    
    # result = (x // interval_length) + 1
    # result = f"time_bucket_{result}"
    if label=="label":
        if x <= 60:
            return "time_bucket_1"
        elif x <= 120:
            return "time_bucket_2"
        elif x <= 180:
            return "time_bucket_3"
        elif x <= 240:
            return "time_bucket_4"
        elif x <= 300:
            return "time_bucket_5"
        else:
            return "time_bucket_6"
    elif label=="bucket":
        if x <= 60:
            return "0-60"
        elif x <= 120:
            return "61-120"
        elif x <= 180:
            return "121-180"
        elif x <= 240:
            return "181-240"
        elif x <= 300:
            return "241-300"
        else:
            return ">300"

    raise Exception(f"label {label} not supported")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--outlier_type", type=str, default="work-outliers")
    parser.add_argument("--file_name", type=str, default="checkin-atl.tsv")
    parser.add_argument("--grid_length", type=int, default=25)
    parser.add_argument("--out_dir", type=str, default="./data/pol")
    parser.add_argument("--features", type=str, default="place")

    args = parser.parse_args()

    features = args.features.split(",")   
    # reported the results for the atlanta dataset
    files = [f"{args.data_dir}/{args.outlier_type}/checkin-atl.tsv"]

    for file in files:
        df_to_save = pattern_life_file_preprocess(args.data_dir, 
                                    args.outlier_type, 
                                    file.split("/")[-1], 
                                    args.grid_length,
                                    args.out_dir,
                                    features,
                                    override=True )
        
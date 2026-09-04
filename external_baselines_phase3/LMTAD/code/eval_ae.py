

import argparse
import os
from collections import  defaultdict
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import (POLConfig, 
                      POLDataset,
                      PortoConfig,
                      PortoDataset)

from models import DAE, VAE, AEConfig, GMSVAE, GMSVAEConfig
from meter import AverageMeter
from utils import log, seed_all

from plot_utils import (plot_agent_surprisal_rate, 
                        plot_metrics_pattern_of_life, 
                        plot_agent_perlexity_over_date)

from plot_utils import (plot_metrics, 
                        load_tsvs)


def get_parser():
    """argparse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()
    return args

@torch.no_grad()
def eval_porto(dataset_config, model, model_path, device, model_type, dataloader=None):
    """eval on the porto dataset"""
    # ratio, level, prob
    
    if not dataloader:
        dataset = PortoDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)
    
    results = defaultdict(list)
    for data in tqdm(dataloader, desc="normal trajectories", total=len(dataloader)):
        # inputs = data["data"][:, :].T.contiguous().to(device)
        # losses = model.autoenc(inputs, inputs, is_train=False) # just reconstruct the input
        # losses['rec'] = model.loss(losses)

        if model_type in ["dae", "vae"]:
            inputs = data["data"][:, :].T.contiguous().to(device)
            losses = model.autoenc(inputs, inputs, is_train=False) 
        elif model_type == "gmvae":
            inputs = data["data"][:, :].contiguous().to(device)
            masks = data["mask"].contiguous().to(device)
            losses = model.autoenc(inputs, masks, is_train=False)
        
        # ipdb.set_trace()
        rec_losses = losses["rec"].tolist()
        outlier = data["metadata"]
        seq_length = (data["mask"].sum(-1)).tolist()
        # trajectories = inputs.T.tolist()
        # ipdb.set_trace()

        results["rec_loss"].extend(rec_losses)
        results["outlier"].extend(outlier)
        results["seq_length"].extend(seq_length)
        # results["trajectory"].extend(trajectories)

        # ipdb.set_trace()
    if dataset_config.include_outliers:
        output_file_name = f"outliers_config_ratio_{dataset_config.outlier_ratio }_level_{dataset_config.outlier_level}_prob_{dataset_config.outlier_prob}"
    else:
        output_file_name = f"{Path(model_path).stem}_outliers_config_ratio_{dataset_config.outlier_ratio }_level_{dataset_config.outlier_level}_prob_{dataset_config.outlier_prob}"

    return results, output_file_name

@torch.no_grad()
def eval_pattern_of_life(dataset_config, model, model_path, device, model_type, dataloader=None):
    """eval on the porto dataset"""
    # ratio, level, prob

    # ipdb.set_trace()

    # dataset_config = checkpoint["dataset_config"]
    if not dataloader:
        dataset_config.logging = False
        dataset = POLDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)

    results = defaultdict(list)

    # ipdb.set_trace()
    for data in tqdm(dataloader, desc="evaluating trajectories", total=len(dataloader)):

        if model_type in ["dae", "vae"]:
            inputs = data["data"][:, :].T.contiguous().to(device)
            losses = model.autoenc(inputs, inputs, is_train=False) 
        elif model_type == "gmvae":
            inputs = data["data"][:, :].contiguous().to(device)
            masks = data["mask"].contiguous().to(device)
            losses = model.autoenc(inputs, masks, is_train=False)

        # inputs = data["data"][:, :].T.contiguous().to(device)

        # losses = model.autoenc(inputs, inputs, is_train=False) # just reconstruct the input
        # losses['rec'] = model.loss(losses)

        rec_losses = losses["rec"].tolist()
        seq_length = (data["mask"].sum(-1)).tolist()
        # trajectories = inputs.T.tolist()
        # ipdb.set_trace()

        user_ids = []
        dates = []
        outliers = []
        for item in data["metadata"]:
            user_ids.append(item[0])
            dates.append(item[1])
            outliers.append(item[2])

        results["user_id"].extend(user_ids)
        results["date"].extend(dates)
        results["rec_loss"].extend(rec_losses)
        results["outlier"].extend(outliers)
        results["seq_length"].extend(seq_length)
        # results["trajectory"].extend(trajectories)

        # ipdb.set_trace()
    output_file_name = f"{'_'.join(Path(model_path).stem.split('_')[1:])}"

    return results, output_file_name

def save_results(results, out_dir, output_file_name):
    df_to_save = pd.DataFrame(results)
    df_to_save.to_csv(
            f"{out_dir}/{output_file_name}.tsv", #'_').
            index=False,
            sep="\t"
        )    
    return df_to_save

def main(eval_args):
    """eval main method"""

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_paths = eval_args.model_file_path.split(",")
    
    for model_path in tqdm(model_paths, desc="model"):
        if model_path == "":
            continue

        print(f"model path: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        args = checkpoint["args"]
        model_config = checkpoint["model_config"]
        dataset_config = checkpoint["dataset_config"]

        if args.model_type in ["dae", "vae"]:
            model = {'dae': DAE, 'vae': VAE}[args.model_type](model_config).to(device)
        elif args.model_type == "gmvae":
            model = GMSVAE(model_config).to(device)
        
        # model = {'dae': DAE, 'vae': VAE}[args.model_type](model_config).to(device)
        
        # ipdb.set_trace()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)
        
        out_dir = f'./eval_results/{"/".join(args.out_dir.split("/")[2:])}'
        print(f"outdir {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        if eval_args.dataset == "porto":
            
            dataset_config = checkpoint['dataset_config']

            if dataset_config.include_outliers:
                
                out_dir += "/include_outliers"
                os.makedirs(out_dir, exist_ok=True)
                print(f"updated outdir {out_dir}")

                results, output_file_name = eval_porto(dataset_config, model, model_path, device, args.model_type)
                save_results(results, out_dir, output_file_name)

            else:
                outlier_parameters = [
                    [0.05, 3, 0.1],
                    [0.05, 3, 0.3],
                    [0.05, 5, 0.1]
                ]

                
                for outlier_parameter in outlier_parameters:
                    
                    dataset_config.outlier_ratio = outlier_parameter[0]
                    dataset_config.outlier_level = outlier_parameter[1]
                    dataset_config.outlier_prob = outlier_parameter[2]
                    dataset_config.outliers_list = ["route_switch", "detour"]
                    dataset_config.include_outliers = True 
                
                    results, output_file_name = eval_porto(dataset_config, model, model_path, device, args.model_type)

                    df_to_save = save_results(results, out_dir, output_file_name)

            dfs = load_tsvs(out_dir)
            plot_metrics(dfs, "rec_loss", out_dir)

        elif eval_args.dataset == "pol":
            results, output_file_name = eval_pattern_of_life(dataset_config, model, model_path, device, args.model_type )

            df_to_save = save_results(results, out_dir, output_file_name)

            df_to_save["date"] = pd.to_datetime(df_to_save["date"])
            df_to_save["id"] = df_to_save.user_id.str.split("_").str[1].astype(int)
            df_to_save["outlier_label"] =   np.where(df_to_save["outlier"] == "non outlier", 0, 1)
            results = plot_metrics_pattern_of_life(df_to_save, "rec_loss", out_dir)
            filtered_data = plot_agent_perlexity_over_date(df_to_save, "rec_loss", out_dir)

if __name__ == "__main__":
    seed_all(123)
    eval_args = get_parser()
    
    main(eval_args)
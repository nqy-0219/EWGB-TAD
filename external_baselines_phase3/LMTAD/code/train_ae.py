
import collections
import argparse
import os
import time
from collections import  defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, auc, precision_recall_curve
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import (POLConfig, 
                      POLDataset,
                      PortoConfig,
                      PortoDataset)

from models import DAE, VAE, AEConfig, GMSVAE, GMSVAEConfig
from meter import AverageMeter
from utils import log, seed_all, save_file_name_pattern_of_life
from eval_ae import eval_porto, eval_pattern_of_life
from metrics import get_metrics, get_pattern_of_life_metrics, get_per_user_metrics


def get_parser():
    """parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_file_name', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--grid_leng', type=int, default=25)
    parser.add_argument('--dataset', type=str, default="porto")
    parser.add_argument('--include_outliers', action='store_true', required=False)
    parser.add_argument('--outlier_days', type=int, default=14)
    # parser.add_argument('--only_outliers', type=bool, default=False)
    parser.add_argument('--features', type=str, default="place")

    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--output_file_name', type=str, default='')
    parser.add_argument('--log_interval', type=int, default=100)

    parser.add_argument('--dim_emb', type=int, default=512)
    parser.add_argument('--dim_h', type=int, default=1024)
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--dim_z', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)


    parser.add_argument('--model_type', default='vae', metavar='M',
                    choices=['dae', 'vae', 'gmvae'])

    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_all_ckpts', action='store_true', required=False)

    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', action='store_true', required=False)

    args = parser.parse_args()
    return args

def eval(model, dataloader, device, model_type):
    """evaluate the model on the dev set"""
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for data in dataloader:

             # was throughing some erros if not contiguous
            # targets = data["data"][:, 1:].T.contiguous().to(device)

            if model_type in ["dae", "vae"]:
                inputs = data["data"][:, :].T.contiguous().to(device)
                losses = model.autoenc(inputs, inputs) # just reconstruct the input
            elif model_type == "gmvae":
                inputs = data["data"][:, :].contiguous().to(device)
                masks = data["mask"].contiguous().to(device)
                losses = model.autoenc(inputs, masks) # just reconstruct the input

            # losses = model.autoenc(inputs, inputs)
            for k, v in losses.items():
                meters[k].update(v.mean().item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def main(args):
    """train ae/vae/gmvae for sequence data"""

    # ipdb.set_trace()
    seed_all(args.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset == "pol":
        args.include_outliers = True
        args.features = args.features.split(",")
        features_folder_name = save_file_name_pattern_of_life(args.features)

        args.out_dir = f"{args.out_dir}/outlier_{args.include_outliers}/{features_folder_name}/dim_h_{args.dim_h}_dim_z_{args.dim_z}_dim_emb_{args.dim_emb}"
        os.makedirs(f"{args.out_dir}", exist_ok=True)

        log_file = f"{args.out_dir}/log.txt"
        args.log_file = log_file
        args.output_file_name = ""
        # override the current log file
        with open(log_file, "w") as f:
            f.write("")
        
        dataset_args = dict(data_dir=args.data_dir, 
                                file_name=args.data_file_name,
                                features=args.features,
                                block_size=args.block_size,
                                grid_leng=args.grid_leng,
                                include_outliers=args.include_outliers,
                                outlier_days=args.outlier_days,
                                logging=True,
                                log_file=args.log_file)
        
        dataset_config = POLConfig(**dataset_args)
        dataset = POLDataset(dataset_config)
        
        args.block_size = dataset_config.block_size

        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate)
        
        
    elif args.dataset == "porto":
        args.include_outliers = False
        dataset_config = PortoConfig()
        # config.file_name = "porto_processed_max_length_300"
            
        dataset_config.file_name = args.data_file_name

        """
        No outlier: 
        [0.05, 3, 0.1], 
        [0.05, 3, 0.3], 
        [0.05, 5, 0.1], 
        """

        dataset_config.outlier_level = 5
        dataset_config.outlier_prob = 0.1
        dataset_config.outlier_ratio = 0.05
        dataset_config.outliers_list = ["route_switch", "detour"]
        # config.outliers_list = ["detour"]
        dataset_config.include_outliers = args.include_outliers
        dataset = PortoDataset(dataset_config)


        if args.debug and args.dataset == "pol":
            pass
        elif args.debug and args.dataset == "porto":
            test_dataset_config = PortoConfig(**vars(dataset_config))
            test_dataset_config.include_outliers = True
            test_dataset_config.outlier_level = 3
            test_dataset_config.outlier_prob = 0.1
            test_dataset_config.outlier_ratio = 0.05
            test_dataset_config.outliers_list = ["route_switch", "detour"]

            test_dataset = PortoDataset(test_dataset_config)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=dataset.collate)
        
        # args.out_dir = f"./results/{args.model_type}/porto/outlier_{config.include_outliers}"

        if dataset_config.include_outliers:
            args.output_file_name = f"ckpt_outliers_{'_'.join(dataset_config.outliers_list)}_ratio_{dataset_config.outlier_ratio}_level_{dataset_config.outlier_level}_prob_{dataset_config.outlier_prob}"
        else:
            args.output_file_name = "ckpt"

        args.out_dir = f"{args.out_dir}/outlier_{dataset_config.include_outliers}_dim_h_{args.dim_h}_dim_z_{args.dim_z}_dim_emb_{args.dim_emb}"

        # ipdb.set_trace()
        os.makedirs(f"{args.out_dir}", exist_ok=True)

        log_file = os.path.join(args.out_dir, f'log_{args.output_file_name}.txt')
        args.log_file = log_file
        #override existing data in the file
        with open(log_file, 'w') as f:
            f.write("")
        # pdb.set_trace()
        log(f"output file name: {args.output_file_name}", args.log_file)

        args.block_size = dataset_config.block_size 
        args.features = []


    log(f"# of vocabulary size: {len(dataset.dictionary)}", args.log_file)
    train_indices, val_indices = dataset.partition_dataset()
    # temp = sorted(list(val_indices))[:20]
    # [0, 4, 25, 45, 47, 55, 58, 59, 71, 78, 99, 107, 119, 140, 158, 162, 167, 168, 175, 181]

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate, sampler=SubsetRandomSampler(train_indices))
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate, sampler=SubsetRandomSampler(val_indices))

    # ipdb.set_trace()

    model_args = dict(dim_emb=args.dim_emb, dim_h=args.dim_h, dropout=args.dropout, nlayers=args.nlayers,
                  dim_z=args.dim_z, lr=args.lr, vocab_size=len(dataset.dictionary), pad_token=dataset.dictionary.pad_token())
    
    if args.model_type in ["dae", "vae"]:
        model_config = AEConfig(**model_args)
        model = {'dae': DAE, 'vae': VAE}[args.model_type](model_config).to(device)

    elif args.model_type == "gmvae":
        model_config = GMSVAEConfig()
        model_config.dim_emb = args.dim_emb
        model_config.dim_z = args.dim_z
        model_config.dim_h = args.dim_h
        model_config.vocab_size = len(dataset.dictionary)
        model = GMSVAE(model_config).to(device)

    # ipdb.set_trace()
    log(f"# of model parameters: {model.get_num_params():.3f}M", args.log_file)

    best_val_loss = None
    stop_train_cumulation=0
    stop_training_count = 3

    train_losses = []
    valid_losses = []
    metric_results = defaultdict(list)
    for epoch in range(args.max_iters):

        start_time = time.time()
        meters = collections.defaultdict(lambda: AverageMeter())
        log('-' * 85, args.log_file)
        model.train()

        current_train_losses = []
        for batch_id, data in enumerate(train_dataloader):

            if args.model_type in ["dae", "vae"]:

                inputs = data["data"][:, :].T.contiguous().to(device) # was throughing some erros if not contiguous
                losses = model.autoenc(inputs, inputs, is_train=True) # just reconstruct the input
            elif args.model_type == "gmvae":
                inputs = data["data"][:, :].contiguous().to(device) # was throughing some erros if not contiguous
                masks = data["mask"].contiguous().to(device)
                losses = model.autoenc(inputs, masks, is_train=True) # just reconstruct the input
            # targets = data["data"][:, 1:].T.contiguous().to(device)

            # ipdb.set_trace()

            losses['loss'] = model.loss(losses)
            model.step(losses)

            for k, v in losses.items():
                meters[k].update(v.item())

            if (batch_id + 1) % args.log_interval == 0:
                log_output = f"|epoch {epoch+1}/{args.max_iters} |\t batch {batch_id+1}/{len(train_dataloader)}|"
                for k, meter in meters.items():
                    log_output += f"\t{k}: {meter.avg:.3f}|"

                    if k == "loss":
                        current_train_losses.append(meter.avg)

                    meter.clear()

                log(log_output, args.log_file)

            # break
        
        valid_meters = eval(model, val_dataloader, device, args.model_type)
        log('-' * 85, args.log_file)
        log_output = f"| end of epoch {epoch+1}/{args.max_iters} | time {(time.time() - start_time):.3f}s | valid:"
        
        for k, meter in valid_meters.items():
            log_output += f"{k}: {meter.avg:.3f}|"
        # if not best_val_loss: #or valid_meters['loss'].avg < best_val_loss:
        log_output += ' | saving model'
        ckpt = {"args": args, "dataset_config": dataset_config, "model_config": model_config, "model": model.state_dict()}

        if args.save_all_ckpts:
            save_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt")
        else:
            save_path = os.path.join(args.out_dir, f"{args.output_file_name}.pt")
        
        torch.save(ckpt, save_path)
        best_val_loss = valid_meters['loss'].avg

        stop_train_cumulation = 0
        # else:
        #     stop_train_cumulation +=1

        
        if args.debug and args.dataset == "pol":

            model.eval()
            results, _ = eval_pattern_of_life(dataset_config, model, args.output_file_name, device, args.model_type, test_dataloader)
            model.train()
            df_results = pd.DataFrame(results)

            red_outliers = [546, 644, 347, 62, 551, 992, 554, 949, 900, 57] # TODO parametirize this. Now we only care about the red outliers
            df_results = get_per_user_metrics(df_results, red_outliers, "rec_loss")

            # ipdb.set_trace()

            log_output += f"\n| red agent results -> avarage f1: {df_results.f1.mean():.3f} | avarage pr_auc: {df_results.pr_auc.mean():.3f}"

            metric_results["epoch"].append(epoch)
            metric_results["avg_f1"].append(df_results.f1.mean())
            metric_results["avg_pr_auc"].append(df_results.pr_auc.mean())

        elif args.debug and args.dataset == "porto":            

            # if test_dataloader is None, then create dataset from the test_dataset_config
            model.eval()
            results, _ = eval_porto(test_dataset_config, model, args.output_file_name, device, args.model_type, test_dataloader)
            model.train()
            df_results = pd.DataFrame(results)            
            
            (_, _, _, _, f1, pr_auc), _ = get_metrics(df_results[df_results["outlier"] != "detour outlier"], "rec_loss")
            log_output += f"\n| route switching outliers -> f1: {f1:.3f} | pr_auc: {pr_auc:.3f}"

            metric_results["epoch"].append(epoch)
            metric_results["f1_rs"].append(f1.mean())
            metric_results["pr_auc_rs"].append(pr_auc.mean())

            (_, _, _, _, f1, pr_auc), _ = get_metrics(df_results[df_results["outlier"] != "route switch outlier"], "rec_loss")
            log_output += f"| detour outliers -> f1: {f1:.3f} | pr_auc: {pr_auc:.3f}"
            
            metric_results["f1_detour"].append(f1.mean())
            metric_results["pr_auc_detour"].append(pr_auc.mean())

        
        train_losses.append(np.sum(current_train_losses).mean())
        valid_losses.append(valid_meters['loss'].avg)

        
        log(log_output, args.log_file)
        

    losses_dict = {"train": train_losses, "val":valid_losses}
    losses_dict = pd.DataFrame(losses_dict)
    losses_dict.to_csv(f"{args.out_dir}/losses.tsv", sep="\t")

    metric_results_df = pd.DataFrame(metric_results)          
    metric_results_df.to_csv(
            f"{args.out_dir}/metrics_results.tsv", 
            index=False,
            sep="\t"
        )
    
    log('Done training', args.log_file)
    # ipdb.set_trace()
if __name__ == "__main__":
    args = get_parser()
    main(args)
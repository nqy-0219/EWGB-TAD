
import time
import os
import math
import argparse
from contextlib import nullcontext
from typing import List
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from datasets import (VocabDictionary, 
                      POLConfig,
                      POLDataset,
                      PortoConfig,
                      PortoDataset)

from models import (
    LMTADConfig,
    LMTAD,
    AEConfig,
    DAE,
    VAE,
    GMSVAEConfig,
    GMSVAE
)
from utils import (log, seed_all, 
                   save_file_name_pattern_of_life, 
                   save_file_new_datset, 
                   save_file_name_porto,
                   save_file_name_trial0)

from eval_porto import eval_porto
from eval_lm import eval_pattern_of_life
from metrics import get_metrics, get_per_user_metrics


def get_parser():
    """argparse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_file_name', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--grid_leng', type=int, default=25)
    parser.add_argument('--dataset', type=str, default="pol", choices=["porto", "pol"])
    parser.add_argument('--include_outliers', action='store_true', required=False)
    parser.add_argument('--outlier_days', type=int, default=14)
    # parser.add_argument('--skip_gps', type=bool, default=True)
    parser.add_argument('--features', type=str, default="place")

    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--output_file_name', type=str, default="")

    parser.add_argument('--eval_interval', type=int, default=250)
    # parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_file', type=str, default="")

    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    # use simple poe (1, 2, 3), instead of the sin and cosine
    parser.add_argument('--integer_poe', action='store_true', required=False)

    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--lr_decay_iters', type=int, default=600000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--min_lr', type=float, default=6e-6)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_iters', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--decay_lr', type=bool, default=True)

    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--compile', action='store_false')
    parser.add_argument('--debug', action='store_true', required=False)

    args = parser.parse_args()
    return args


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, args):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr - args.min_lr)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(args, model, test_dataloader, device, ctx):
    out = {}
    model.eval()
    # pdb.set_trace()
    # for split in ['train', 'val']:
    losses = torch.zeros(len(test_dataloader))
    for batch, data in enumerate(test_dataloader):
        X = data["data"][:, :-1].contiguous().to(device) # was throughing some erros if not contiguous
        Y = data["data"][:, 1:].contiguous().to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses[batch] = loss.item()
        # out[split] = losses.mean()
    model.train()
    return losses.mean()

@torch.no_grad()
def model_eval(args,
         epoch,
         iter_num, 
         model, 
         optimizer, 
         model_conf, 
         dataset_config, 
         test_dataloader, 
         device, 
         ctx, 
         best_val_loss, 
         metric_results,
         **kwargs):
    """run evaluation on the eval set"""
    # evaluate the loss on train/val sets and write checkpoints   

    model.eval()
    val_loss = estimate_loss(args, model, test_dataloader, device, ctx)
    # pdb.set_trace()
    log(f"|step {iter_num}:  val loss {val_loss:.4f}|", args.log_file)
    saved_model_eval = False
    # if val_loss < best_val_loss: #or always_save_checkpoint:
        
    best_val_loss = val_loss
    if iter_num > 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_config': model_conf,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'dataset_config': dataset_config,
            "args": args
        }

        log_output = f"\nsaving checkpoint to {args.out_dir}"
        if args.output_file_name != "":
            ckpt_name = f"ckpt_{args.output_file_name}"
        else:
            ckpt_name = f"ckpt"

        ckpt_name += f"epoch_{epoch}_batch_{iter_num}.pt"

        torch.save(checkpoint, os.path.join(args.out_dir, ckpt_name))

        # ipdb.set_trace()
        if args.debug and args.dataset == "pol":
            model.eval()
            results = eval_pattern_of_life(kwargs["test_dataset_config"], model, device, kwargs["dictionary"], kwargs["dataloader"])
            model.train()
            df_results = pd.DataFrame(results)

            red_outliers = [546, 644, 347, 62, 551, 992, 554, 949, 900, 57] # TODO parametirize this. Now we only care about the red outliers
            df_results = get_per_user_metrics(df_results, red_outliers, "log_perplexity")

            # ipdb.set_trace()

            log_output += f"\n| red agent results -> avarage f1: {df_results.f1.mean():.3f} | avarage pr_auc: {df_results.pr_auc.mean():.3f}"

            metric_results["epoch"].append(epoch)
            metric_results["avg_f1"].append(df_results.f1.mean())
            metric_results["avg_pr_auc"].append(df_results.pr_auc.mean())

        elif args.debug and args.dataset == "porto":   
            results = eval_porto(model=model, device=device, dataloader=kwargs["dataloader"])
            df_results = pd.DataFrame(results)            
            
            (_, _, _, _, f1, pr_auc), treshold = get_metrics(df_results[df_results["outlier"] != "detour outlier"], "log_perplexity")

            metric_results["model_number"].append(iter_num)
            metric_results["f1_rs"].append(f1)
            metric_results["pr_rs"].append(pr_auc)

            log_output += f"\n| route switching outliers -> f1: {f1:.3f} | pr_auc: {pr_auc:.3f}"

            (_, _, _, _, f1, pr_auc), treshold = get_metrics(df_results[df_results["outlier"] != "route switch outlier"], "log_perplexity")
            metric_results["f1_detour"].append(f1)
            metric_results["pr_auc_detour"].append(pr_auc)
            log_output += f"| detour outliers -> f1: {f1:.3f} | pr_auc: {pr_auc:.3f} |\n"
        
        log(log_output, args.log_file)
    
    saved_model_eval = True
    model.train()
    return best_val_loss, saved_model_eval

def main(args):
    """train orchastration"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    if args.dataset == "pol":
        args.features = args.features.split(",")
        args.include_outliers = True

        features_folder_name = save_file_name_pattern_of_life(args.features)
        args.out_dir = f"{args.out_dir}/outlier_{args.include_outliers}/{features_folder_name}/n_layer_{args.n_layer}_n_head_{args.n_head}_n_embd_{args.n_embd}_lr_{args.lr}_integer_poe_{args.integer_poe}"

        os.makedirs(f"{args.out_dir}", exist_ok=True)

        output_file_name = save_file_name_pattern_of_life(args.features)
        log_file = f"{args.out_dir}/log.txt"
        args.log_file = log_file
        args.output_file_name = ""
        # override the current log file
        with open(args.log_file, "w") as f:
            f.write("")

        # ipdb.set_trace()
        
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

    elif args.dataset == "porto":
        
        """
        No outlier: 
        [0.05, 3, 0.1], Yes-RS | No-D | No-RS-D
        [0.05, 3, 0.3], Yes-RS | No-D | No-RS-D 
        [0.05, 5, 0.1], No-RS | No-D | No-RS-D
        """
        args.include_outliers = False

        dataset_config = PortoConfig()
        dataset_config.file_name = args.data_file_name
        dataset_config.outlier_level = 5
        dataset_config.outlier_prob = 0.1
        dataset_config.outlier_ratio = 0.05
        dataset_config.outliers_list = ["route_switch", "detour"]
        # config.outliers_list = ["detour"]
        dataset_config.include_outliers = args.include_outliers

        args.out_dir = f"{args.out_dir}/outlier_{dataset_config.include_outliers}/n_layer_{args.n_layer}_n_head_{args.n_head}_n_embd_{args.n_embd}_lr_{args.lr}_integer_poe_{args.integer_poe}"

        # ipdb.set_trace()
        
        os.makedirs(f"{args.out_dir}", exist_ok=True)

        output_file_name = save_file_name_porto(dataset_config)
        log_file = f"{args.out_dir}/log_{output_file_name}.txt"
        args.log_file = log_file
        args.output_file_name = output_file_name

        # override the current log file
        with open(args.log_file, "w") as f:
            f.write("")

        dataset = PortoDataset(dataset_config)
        
        log(f"output file name: {args.output_file_name}", args.log_file)

        args.block_size = dataset_config.block_size # for SOT and EOT``
        args.features = []

    train_indices, val_indices = dataset.partition_dataset()
    

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate, sampler=SubsetRandomSampler(train_indices))
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate, sampler=SubsetRandomSampler(val_indices))

    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size, log_file=args.log_file,
                  bias=False, vocab_size=len(dataset.dictionary), dropout=args.dropout, pad_token=dataset.dictionary.pad_token(), logging=True, integer_poe=args.integer_poe)
    
    model_conf = LMTADConfig(**model_args)
    model = LMTAD(model_conf)

    model = model.to(device) # TODO maybe compile the model to speed up the training process? 
    model.train()
    # compile the model
    if not args.compile:
        print("")
        log(f"compiling the model... (takes a ~minute)", args.log_file)
        model = torch.compile(model) # requires PyTorch 2.0

    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))    
    
    best_val_loss = 1e9
    t0 = time.time()
    train_losses = []
    valid_losses = []
    cumul_train_loses = 0
    cumulation = 1
    save_model_count = 3
    metric_results = defaultdict(list)

    # ipdb.set_trace()
    eval_outliers_kwargs = None

    if args.debug and args.dataset == "pol":

        eval_outliers_kwargs = {}

        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate)
        eval_outliers_kwargs["dataloader"] = test_dataloader
        eval_outliers_kwargs["test_dataset_config"] = dataset_config
        eval_outliers_kwargs["dictionary"]  = dataset.dictionary
        
    elif args.debug and args.dataset == "porto":    
        
        eval_outliers_kwargs = {}

        test_dataset_config = PortoConfig(**vars(dataset_config))
        test_dataset_config.include_outliers = True
        test_dataset_config.outlier_level = 3
        test_dataset_config.outlier_prob = 0.1
        test_dataset_config.outlier_ratio = 0.05
        test_dataset_config.outliers_list = ["route_switch", "detour"]

        log('loading the metrics test dataset', args.log_file)
        test_dataset = PortoDataset(test_dataset_config)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=dataset.collate)

        eval_outliers_kwargs["dataloader"] = test_dataloader
        eval_outliers_kwargs["test_dataset_config"] = test_dataset_config
        
    iter_num = 0
    for epoch in range(args.max_iters):
        save_model = False
        log('-' * 85, args.log_file)
        for batch_id, data in enumerate(train_dataloader):

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num, args) if args.decay_lr else args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            inputs = data["data"][:, :-1].contiguous().to(device) # was throughing some erros if not contiguous
            targets = data["data"][:, 1:].contiguous().to(device)

            if batch_id % (len(train_dataloader) - 1) == 0:
                best_val_loss, saved_model_eval = model_eval(args,
                                                       epoch, 
                                                       batch_id, 
                                                       model, 
                                                       optimizer, 
                                                       model_conf, 
                                                       dataset_config, 
                                                       val_dataloader, 
                                                       device, 
                                                       ctx, 
                                                       best_val_loss,
                                                       metric_results,
                                                       **eval_outliers_kwargs)

                if saved_model_eval:
                    save_model = saved_model_eval

                train_losses.append(cumul_train_loses/cumulation)
                valid_losses.append(best_val_loss.item())
                cumulation = 1
                cumul_train_loses = 0

            # ipdb.set_trace()
            with ctx:
                logits, loss = model(inputs, targets)

            scaler.scale(loss).backward()
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            cumul_train_loses += loss.item()
            if batch_id % args.log_interval == 0:
                log(f"|epoch {epoch+1}/{args.max_iters} | batch {batch_id+1}/{len(train_dataloader)}: loss {loss.item():.4f} \t| time {dt*1000:.2f}ms|", args.log_file)

            cumulation += 1
            iter_num +=1 

        if not save_model:
            save_model_count +=1
        else:
            save_model_count = 0

        if save_model_count >= 5+1:
            log(f'stopped training becaue the loss did not improve {save_model_count-1} times', args.log_file)
            # break # stopp training

        log('-' * 85, args.log_file)
        log(f"|save_model_count: {save_model_count} | lr: {lr}",  args.log_file)
    # res = model(train_dataset[:1, :-1], train_dataset[:1, 1:])

    losses_dict = {"train": train_losses, "val":valid_losses}
    losses_dict = pd.DataFrame(losses_dict)
    losses_dict.to_csv(f"{args.out_dir}/losses_{args.output_file_name}.tsv", sep="\t")

    metric_results_df = pd.DataFrame(metric_results)          
    metric_results_df.to_csv(
            f"{args.out_dir}/metrics_results.tsv", 
            index=False,
            sep="\t"
        )

if __name__ == "__main__":
    args = get_parser()

    main(args)
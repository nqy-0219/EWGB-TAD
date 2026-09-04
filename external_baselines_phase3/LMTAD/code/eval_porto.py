

import os
import math
import pdb
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models import LMTADConfig, LMTAD
from datasets import (VocabDictionary, 
                      PortoConfig, 
                      PortoDataset)

from plot_utils import (plot_metrics, 
                        load_tsvs)

def get_parser():
    """argparse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_path", type=str, default="")

    args = parser.parse_args()
    return args

def get_perplexity_slow(input, model, eot_token, ctx,device, bedug=True):
    """
    this method calculates the perplexity of of each trajectory given the model.
    This method is slow since it loops through each input and tokens
    input: 
        input: (B, T) -> trajectories potentially padded
        eot_token: int 

    return:
        perplexities: (B,), the perplexity of each trajectory given by the model
    """

    probs = []
    for current_input in input:
        current_input = current_input.unsqueeze(0)
        # pdb.set_trace()
        current_probs = []
        for i in range(0, current_input.size(1) - 1):
            # if we encounter the eot token, just stop
            if current_input[:, i].item() == eot_token:
                # pdb.set_trace()
                if bedug:
                    print("break the loop")
                break
            with ctx:
                logits, _ = model(current_input[:, :i+1]) 

            logits = logits[:, [-1], :] # get the last token
            all_probs = F.softmax(logits, dim=-1)
            current_probs.append(all_probs[0, 0, current_input[:, i+1].item()].item())

        probs.append(current_probs)

    log_perplexities = []
    for prob in probs:
        log_perplexities.append((np.log(prob).sum() / len(prob)) * -1)

    log_perplexities = torch.tensor(log_perplexities).float().to(device)

    return log_perplexities

    
def get_perplexity_fast(input, model, mask):
    """
    this method calculates the perplexity of of each trajectory given the model.
    This method is fast compared tot the get_perplexity_slow
    input: 
        input: (B, T) -> trajectories potentially padded

    return:
        perplexities: (B,), the perplexity of each trajectory given by the model
    """

    # [SOT, 23, 553, 23, EOT, PAD, PAD] -> [23, 553, 23, EOT, PAD, PAD, PAD] (ideally)
    logits, _ = model(input[:, :-1]) # (B, T (T-1), C)
    all_probs = F.softmax(logits, dim=-1) # (B, T (T-1), C)
    probs = torch.gather(all_probs, -1, input[:, 1:].unsqueeze(-1)).squeeze() # (B, T (T-1))
    log_perplexities = torch.log(probs.masked_fill(mask[:, 1:] == 0, 1)).sum(-1) / mask[:, 1:].sum(-1) * -1

    return log_perplexities, probs

@torch.no_grad()
def get_trajectory_probability(
        model,
        data,
        device,
        ctx,
        eot_token,
        results,
        debug=False
): 
    
    input = data["data"].to(device) 
    mask = data["mask"].to(device)

    if debug:
        log_perplexities_slow = get_perplexity_slow(input, model, eot_token, ctx, device, bedug=debug)

    log_perplexities_fast, raw_probs = get_perplexity_fast(input, model, mask)
    
    if debug:
        close = torch.allclose(log_perplexities_slow, log_perplexities_fast, atol=1e-1)
        print(f"Were the perplexities close? {close}")

        if not close:
            pdb.set_trace()
    
    raw_probs_list = []
    trajectory = []

    for idx in range(raw_probs.size(0)):
        probs = raw_probs[idx, :mask[idx].sum().item() - 1] # get the probability of all the tokens up to the eot token
        raw_probs_list.append(probs.tolist())
        traj = input[idx, 1:mask[idx].sum().item()] # get the trokens without the SOT up to the EOT
        trajectory.append(traj.tolist())

    log_perplexity = log_perplexities_fast.tolist()
    outlier = data["metadata"]
    seq_length = (mask.sum(-1) - 1).tolist() # we don't include the SOT
    # raw_probs = raw_probs.tolist()
    # trajectory = input.tolist()

    try:
        assert len(log_perplexity) == len(outlier) == len(seq_length) == len(raw_probs_list) == len(trajectory)
    except:
        pdb.set_trace()

    results["log_perplexity"].extend(log_perplexity)
    results["outlier"].extend(outlier)
    results["seq_length"].extend(seq_length)
    results["raw_probs"].extend(raw_probs_list)
    results["trajectory"].extend(trajectory)

    # pdb.set_trace()

@torch.no_grad()
def eval_porto(model, device, dataset_config=None, dataloader=None):
    """eval on the porto dataset"""
    # ratio, level, prob
    assert  dataset_config or dataloader

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    if not dataloader:
        dataset = PortoDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)
    
    results = defaultdict(list)
    for data in tqdm(dataloader, desc="normal trajectories", total=len(dataloader)):
        
        get_trajectory_probability(
                    model,
                    data,
                    device,
                    ctx,
                    -1, # EOT token don't need it in this context 
                    results,
                    debug=False
            )

        # ipdb.set_trace()

    return results

def main(eval_args):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print(f"model path: {eval_args.model_file_path}")
    checkpoint = torch.load(eval_args.model_file_path, map_location=device)
    args = checkpoint["args"]
    
    model_conf = checkpoint["model_config"]
    model_conf.logging = False
    model = LMTAD(model_conf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    dataset_config = checkpoint["dataset_config"]
    dataset_config.logging = False

    if dataset_config.include_outliers:
        
        out_dir = f"./eval_results/{'/'.join(args.out_dir.split('/')[2:])}"
        out_dir += "/include_outliers"
        print(f"outdir {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        dataset = PortoDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)

        results = defaultdict(list)

        for data in tqdm(dataloader, desc="normal trajectories", total=len(dataloader)):

            get_trajectory_probability(
                    model,
                    data,
                    device,
                    ctx,
                    dataset.dictionary.eot_token(), # EOT token 
                    results,
                    debug=False
            )

        df_to_save = pd.DataFrame(results)
        
        output_file = f"outliers_config_ratio_{dataset_config.outlier_ratio }_level_{dataset_config.outlier_level}_prob_{dataset_config.outlier_prob}.tsv"
        
        df_to_save.to_csv(
                f"{out_dir}/{output_file}", 
                index=False,
                sep="\t"
            )
    else:
        # config = checkpoint["config"] #PortoConfig()
        dataset_config.include_outliers = True

        # ratio, level, prob
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

            out_dir = f"./eval_results/{'/'.join(args.out_dir.split('/')[2:])}"
            print(f"outdir {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            dataset = PortoDataset(dataset_config)
            dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)

            results = defaultdict(list)

            for data in tqdm(dataloader, desc="normal trajectories", total=len(dataloader)):

                get_trajectory_probability(
                        model,
                        data,
                        device,
                        ctx,
                        dataset.dictionary.eot_token(), # EOT token 
                        results,
                        debug=False
                )

            df_to_save = pd.DataFrame(results)
            
            output_file = f"{Path(eval_args.model_file_path).stem}_outliers_config_ratio_{dataset_config.outlier_ratio }_level_{dataset_config.outlier_level}_prob_{dataset_config.outlier_prob}.tsv"
            
            df_to_save.to_csv(
                    f"{out_dir}/{output_file}", 
                    index=False,
                    sep="\t"
                )
            
        dfs = load_tsvs(out_dir)
        plot_metrics(dfs, "log_perplexity", out_dir)

           
if __name__ == "__main__":
    eval_args = get_parser()
    main(eval_args)


import os
import math
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
import argparse

# import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models import LMTADConfig, LMTAD
from datasets import (VocabDictionary, 
                      POLConfig, 
                      POLDataset)

from utils import log
from plot_utils import (plot_agent_surprisal_rate, 
                        plot_metrics_pattern_of_life, 
                        plot_agent_perlexity_over_date)

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
                    # print("break the loop")
                    pass
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
        dictionary,
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
    
    raw_probs_list = []
    trajectories = []
    user_ids = []
    dates = []
    outliers = []
    surprise_rates = []
    top_k_surprise_rates = []
    min_probablities = []
    k = 3

    for idx in range(raw_probs.size(0)):
        probs = raw_probs[idx, :mask[idx].sum().item() - 1] # get the probability of all the tokens up to the eot token
        min_prob = probs[1:].min().item()

        sorted_probs, _ = probs[1:].sort()
        top_k_probs = sorted_probs[:k]
        top_k_surprise_rate = -torch.log2(top_k_probs).mean()

        top_k_probs = probs[1:k+1]
        suprisal_rate = -np.log2(min_prob)

        raw_probs_list.append(probs.tolist())
        traj = input[idx, 1:mask[idx].sum().item()] # get the trokens without the SOT up to the EOT
        trajectories.append(dictionary.decode(traj.tolist()))
        
        user_ids.append(data["metadata"][idx][0])
        dates.append(data["metadata"][idx][1])
        outliers.append(data["metadata"][idx][2])

        surprise_rates.append(suprisal_rate)
        min_probablities.append(min_prob)
        top_k_surprise_rates.append(top_k_surprise_rate.item())

    log_perplexity = log_perplexities_fast.tolist()
    seq_length = (mask.sum(-1) - 1).tolist() # we don't include the SOT

    try:
        assert len(log_perplexity) == len(outliers) == len(seq_length) == len(raw_probs_list) == len(trajectories) == len(dates) == len(outliers)
    except:
        pass

    

    results["user_id"].extend(user_ids)
    results["date"].extend(dates)
    results["log_perplexity"].extend(log_perplexity)
    results["min_probability"].extend(min_probablities)
    results["max_surpirsal_rate"].extend(surprise_rates)
    results["top_k_surpirsal_rate"].extend(top_k_surprise_rates)
    results["outlier"].extend(outliers)
    results["seq_length"].extend(seq_length)
    results["raw_probs"].extend(raw_probs_list)
    results["trajectory"].extend(trajectories)


@torch.no_grad()
def eval_pattern_of_life(dataset_config, model, device, dictionary=None, dataloader=None):
    assert dataset_config or dataloader

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    if not dataloader:
        dataset_config.logging = False
        dataset = POLDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)

    results = defaultdict(list)

    for data in tqdm(dataloader, desc="normal trajectories", total=len(dataloader)):
        
        get_trajectory_probability(
                    model,
                    data,
                    dictionary,
                    device,
                    ctx,
                    -1, # EOT token 
                    results,
                    debug=False
            )
        
    return results

def main(eval_args):
    """run eval main function"""

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model_name="results/work-outliers/checkin-bjng/ckpt_distance_duration_place.pt"

    print(eval_args.model_file_path)

    checkpoint = torch.load(eval_args.model_file_path, map_location=device)
    args = checkpoint["args"]

    # ipdb.set_trace()
    out_dir = f"./eval_results/{'/'.join(args.out_dir.split('/')[2:])}"

    # ipdb.set_trace()
    os.makedirs(out_dir, exist_ok=True)

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

    results = defaultdict(list)
    # ipdb.set_trace()

    if args.dataset == "pol":
        # results = eval_pattern_of_life(dataset_config, model, device, dictionary=None, dataloader=None)
        dataset_config.logging = False
        dataset = POLDataset(dataset_config)
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate)

        for data in tqdm(dataloader, desc="normal trajectories", total=len(dataloader)):
            
            get_trajectory_probability(
                        model,
                        data,
                        dataset.dictionary,
                        device,
                        ctx,
                        dataset.dictionary.eot_token(), # EOT token 
                        results,
                        debug=False
                )
            
        df_to_save = pd.DataFrame(results)
        output_file = f"results.tsv"
        df_to_save.to_csv(
                f"{out_dir}/{output_file}", 
                index=False,
                sep="\t"
            )

        df_to_save["date"] = pd.to_datetime(df_to_save["date"])
        df_to_save["id"] = df_to_save.user_id.str.split("_").str[1].astype(int)
        df_to_save["outlier_label"] =   np.where(df_to_save["outlier"] == "non outlier", 0, 1)
        plot_agent_surprisal_rate(df_to_save , 5, 6, out_dir)
        results = plot_metrics_pattern_of_life(df_to_save, "log_perplexity", out_dir)
        filtered_data = plot_agent_perlexity_over_date(df_to_save, "log_perplexity", out_dir)
        

if __name__ == "__main__":
    
    eval_args = get_parser()
    main(eval_args)
#bin/sh

root_dir="./code"
cd ${root_dir}

model_file_path="results/LMTAD/pol_old/work-outliers/checkin-atl/outlier_True/features_place/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckptepoch_11_batch_12386.pt"

python eval_lm.py --model_file_path ${model_file_path}

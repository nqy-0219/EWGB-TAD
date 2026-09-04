#!bin/sh

dataset="porto" # pol, porto
batch_size=32

#training configuration
eval_interval=1000
log_interval=50
max_iters=50

#model config
block_size=-1
n_layer=8
n_head=12
n_embd=768
dropout=0.2
lr_decay_iters=60000
beta1=0.9
beta2=0.99
min_lr=3e-5
lr=3e-4
warmup_iters=5000
weight_decay=1e-1
decay_lr=True
grad_clip=1.0

if [[ "${dataset}" == "porto" ]] ; then
    
    data_dir="./data"
    data_file_name="porto_processed"
    out_dir="./results/LMTAD/porto"
    outlier_days=0
    output_file_name=None
    features="no_features"
    grid_leng=25

elif [[ "${dataset}" == "pol" ]]; then

    dataset_name="checkin-atl"
    data_dir="./data/pol/work-outliers/${dataset_name}"
    data_file_name="data"
    features="place"
    outlier_days=14
    out_dir="./results/LMTAD/pol/work-outliers/${dataset_name}"
    output_file_name=None
    grid_leng=25

fi

root_dir="./code"
cd ${root_dir}
python train_LMTAD.py \
    --data_dir ${data_dir} --data_file_name ${data_file_name} --dataset ${dataset} --outlier_days ${outlier_days} \
    --features ${features} --batch_size ${batch_size} --grid_leng ${grid_leng} \
    --out_dir ${out_dir} --output_file_name ${output_file_name} \
    --eval_interval ${eval_interval} --log_interval ${log_interval} --max_iters ${max_iters} \
    --block_size ${block_size} --n_layer ${n_layer} --n_head ${n_head} --n_embd ${n_embd} --dropout ${dropout} \
    --lr_decay_iters ${lr_decay_iters} --beta1 ${beta1} --beta2 ${beta1} --min_lr ${min_lr} --lr ${lr} \
    --warmup_iters ${warmup_iters} --weight_decay ${weight_decay} --decay_lr ${decay_lr} --grad_clip ${grad_clip} \
    --debug
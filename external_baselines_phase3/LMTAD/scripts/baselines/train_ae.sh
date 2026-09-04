#!bin/sh


#data arguments
dataset="porto" # porto, pol


batch_size=64

#model config
model_type="gmvae" #vae, dae, gmvae
block_size=-1
nlayers=1
dim_h=1024
dim_emb=768
dim_z=128
# dim_h=128
# dim_emb=64
# dim_z=32
dropout=0.2
lr=3e-4

output_file_name=None

#training configuration
max_iters=50
log_interval=100

grad_clip=1.0

if [[ "${dataset}" == "porto" ]]; then
    # # ae porto

    grid_leng=25
    outlier_days=14
    features="place" #gps,distance,duration,place
    data_dir="./data" # for porto
    data_file_name="porto_processed" # porto
    out_dir="./results/${model_type}/${dataset}" # for proto
    

elif [[ "${dataset}" == "pol" ]]; then
    # ae pattern of life 
    grid_leng=25
    outlier_days=14
    features="place" #gps,distance,duration,place
    dataset_name="checkin-atl" #checkin-atl, checkin-bjng
    data_file_name="data" #pattern of life
    data_dir="./data/${dataset}/work-outliers/${dataset_name}" #pattern of life
    out_dir="./results/${model_type}/${dataset}/work-outliers/${dataset_name}" # for pattern_of_life

elif [[ "${dataset}" == "trial0" ]]; then
    # ae pattern of life
    data_dir="./data/trial0"
    grid_leng=300
    outlier_days=0
    features="gps" #gps,distance,duration,agent_id
    data_file_name="no_file"
    out_dir="./results/${model_type}/${dataset}" # for proto

fi

root_dir="./code"
cd ${root_dir}

python train_ae.py \
    --data_dir ${data_dir} --data_file_name ${data_file_name} --dataset ${dataset} --outlier_days ${outlier_days} \
    --features ${features} --batch_size ${batch_size} --grid_leng ${grid_leng} \
    --out_dir ${out_dir} --output_file_name ${output_file_name} \
    --max_iters ${max_iters} --log_interval ${log_interval} --model_type ${model_type}\
    --block_size ${block_size} --nlayers ${nlayers} --dim_h ${dim_h} --dim_emb ${dim_emb} --dim_z ${dim_z} --dropout ${dropout} \
    --lr ${lr} --grad_clip ${grad_clip} --save_all_ckpts --debug
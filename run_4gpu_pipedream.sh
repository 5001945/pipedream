#!/usr/bin/env bash
cd runtime/image_classification

module="models.vgg16.gpus=4"
batch_size=64  # 256/4
imagenet="/home/mapae/imagenet"
master_addr="localhost"
config_path="models/vgg16/gpus=4_straight/mp_conf.json"  # config_path="models/vgg16/gpus=4/hybrid_conf.json"
distributed_backend="nccl"

script="main_with_runtime.py"
args="--module $module -b $batch_size --data_dir $imagenet --master_addr $master_addr --config_path $config_path --distributed_backend $distributed_backend"

python $script $args --rank 0 --local_rank 0
python $script $args --rank 1 --local_rank 1
python $script $args --rank 2 --local_rank 2
python $script $args --rank 3 --local_rank 3

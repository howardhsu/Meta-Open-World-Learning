#!/bin/bash

top_n=$1
ncls=$2
run=0

mkdir -p ../runs

baseline="../runs/l2ac_"${top_n}"_"${ncls}
mkdir -p $baseline
run_dir=$baseline/$run/
mkdir -p ${run_dir}

config="{\"seed\":"$run", \"batch_size\": 256, \"model_type\": \"l2ac\", \"top_n\": "${top_n}", \"ncls\": "$ncls",\"set_mode\": \"train1\", \"db\": \"amazon\", \"out_dir\": \""${run_dir}"\", \"set_modes\": [\"test_25\", \"test_50\", \"test_75\"], \"vote_n\": 1}"
config_file=$run_dir/config.json
echo -e $config > ${config_file}
python ../src/train_l2ac.py ${config_file}
python ../src/eval.py ${config_file}


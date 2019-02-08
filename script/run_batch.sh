#!/bin/bash

mkdir -p ../runs
for top_n in 1 3 5 10 15 20
do
    for ncls in 2 4 6 10 15 20
    do
        for run in `seq 0 1 9`
        do
            baseline="../runs/l2ac_"${top_n}"_"${ncls}
            mkdir -p $baseline
            run_dir=$baseline/$run/
            mkdir -p ${run_dir}
            
            config="{\"seed\":"$run", \"batch_size\": 256, \"model_type\": \"l2ac\", \"top_n\": "${top_n}", \"ncls\": "$ncls",\"set_mode\": \"train1\", \"db\": \"amazon\", \"out_dir\": \""${run_dir}"\", \"set_modes\": [\"test_25\", \"test_50\", \"test_75\"], \"vote_n\": 1}"
            config_file=${run_dir}/config.json
            echo -e $config > ${config_file}
            python ../src/train_l2ac.py ${config_file} > ${run_dir}/train.log 2>&1
            python ../src/eval.py ${config_file} > ${run_dir}/eval.log 2>&1
        done
    done
done          
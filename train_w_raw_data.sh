#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Error: Missing arguments."
  echo "Usage: ./train_w_raw_data.sh [DATA_PATH] [SEED_PATH]"
  exit 1
fi

data_path=$1
seed_path=$2

declare -a unreduceable_models=("cpcprot" "seqvec")
declare -a models=("bepler" "cpcprot" "esm" "esm1b" "fasttext" "glove" "onehot" "plusrnn" "prottrans_albert" "prottrans_bert" "prottrans_t5bfd" "prottrans_xlnet_uniref100" "prottrans_t5xlu50" "word2vec")
find $data_path -type f -name "*.csv" | while IFS= read -r filepath; do
    filename=$(basename "$filepath" .csv)
    filepath="${filepath%/*}"
    # This is a patch, remove
    if [[ $filename == *train_dataset ]]; then
        while IFS="" read -r seed || [ -n "$seed" ]; do 
            echo "Using $filepath and $filename with seed $seed"
            
            for model in ${models[@]}; do
                if ! [[ " $unreduceable_models[@] " =~ $model ]]; then
                    python execute_training.py -i ${filepath} -e ${model} -s $seed -y ./results/$(basename "$filepath")_${model}_history_2D.csv -p ./results/metrics.csv -m ./results/$(basename "$filepath")_${model}_model_2D.keras --Dim2D --nosplitted -d $filename
                fi
                    python execute_training.py -i ${filepath} -e ${model}_reduced -s $seed -y ./results/$(basename "$filepath")_${model}_history_1D.csv -p ./results/metrics.csv -m ./results/$(basename "$filepath")_${model}_1D.keras --nosplitted -d $filename
            done

            for i in $(seq 0 7); do
                    python execute_training.py -i ${filepath} -e Group_${i} -s $seed -y ./results/$(basename "$filepath")_Group_${i}_history.csv -p ./results/metrics.csv -m ./results/$(basename "$filepath")_Group_${i}.keras --nosplitted -d $filename
                    python execute_training.py -i ${filepath} -e Group_${i}_fft -s $seed -y ./results/$(basename "$filepath")_Group_${i}_history_fft.csv -p ./results/metrics.csv -m ./results/$(basename "$filepath")_Group_${i}_fft.keras --nosplitted -d $filename
            done
        done < $seed_path
    fi
    
done

#!/bin/zsh

if [ $# -lt 2 ]; then
  echo "Error: Missing arguments."
  echo "Usage: ./train.sh [DATA_PATH] [SEED]"
  exit 1
fi

data_path=$1
seed=$2

declare -a unreduceable_models=("cpcprot" "seqvec")
declare -a models=("bepler" "cpcprot" "esm" "esm1b" "fasttext" "glove" "onehot" "plusrnn" "prottrans_albert" "prottrans_bert" "prottrans_t5bfd" "prottrans_xlnet_uniref100" "prottrans_t5xlu50" "word2vec")
for model in ${models[@]}; do
    if ! [[ " $unreduceable_models[@] " =~ $model ]]; then
        python scripts/execute_training.py -i ${data_path} -e ${model} -s $seed -y ./results/${model}_history_2D.csv -p ./results/metrics.csv -m ./results/${model}_model_2D.keras --Dim2D
    fi
        python scripts/execute_training.py -i ${data_path} -e ${model}_reduced -s $seed -y ./results/${model}_history_1D.csv -p ./results/metrics.csv -m ./results/${model}_1D.keras
done

for i in $(seq 0 7); do
        python scripts/execute_training.py -i ${data_path} -e Group_${i} -s $seed -y ./results/Group_${i}_history.csv -p ./results/metrics.csv -m ./results/Group_${i}.keras
        python scripts/execute_training.py -i ${data_path} -e Group_${i}_fft -s $seed -y ./results/Group_${i}_history_fft.csv -p ./results/metrics.csv -m ./results/Group_${i}_fft.keras
done

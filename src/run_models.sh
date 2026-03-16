#!/bin/bash

datasets=("LPMC" "SwissMetro" "easySHARE")
models=("TasteNet" "DNN")
functional_intercepts=("true" "false")
functional_params=("true" "false")

for fi in "${functional_intercepts[@]}"; do
  for fp in "${functional_params[@]}"; do
    for dataset in "${datasets[@]}"; do
      for model in "${models[@]}"; do
        if [[ "$model" == "DNN" && "$fp" == "true" ]]; then
          continue
        fi

        echo "Running $dataset $model fi=$fi fp=$fp"

        python main.py \
          --functional_intercept $fi \
          --functional_params $fp \
          --model $model \
          --save_model true \
          --optimal_hyperparams true \
          --dataset $dataset

      done
    done
  done
done
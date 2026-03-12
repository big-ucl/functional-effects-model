datasets=("LPMC" "SwissMetro" "easySHARE")
models=("TasteNet" "DNN")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for func_int in True False; do
      for func_params in True False; do

        if [[ "$model" == "DNN" && "$func_params" == "True" ]]; then
          continue
        fi

        python hyperparameter_search.py \
          --dataset $dataset \
          --model $model \
          --func_int $func_int \
          --func_params $func_params

      done
    done
  done
done
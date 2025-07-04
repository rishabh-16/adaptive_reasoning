# Usage: compute_hyperparam_indices <SLURM_ARRAY_TASK_ID> <HYPERPARAM_NAMES[@]> <HYPERPARAM_INDICES[@]>
# Sets the array HYPERPARAM_INDICES with the correct indices for each hyperparameter.
compute_hyperparam_indices() {
    local slurm_id=$1
    shift
    local -n names=$1
    shift
    local -n indices=$1

    local num_hypers=${#names[@]}
    local remainder=$slurm_id

    for ((i=0; i<num_hypers; i++)); do
        local product=1
        for ((j=i+1; j<num_hypers; j++)); do
            arr_name="${names[j]}"
            local -n arr_ref=$arr_name
            local len=${#arr_ref[@]}
            if [[ $len -eq 0 ]]; then
                echo "Error: Hyperparameter array '${arr_name}' is empty." >&2
                return 1
            fi
            product=$((product * len))
        done
        arr_name="${names[i]}"
        local -n arr_ref=$arr_name
        local len=${#arr_ref[@]}
        if [[ $len -eq 0 ]]; then
            echo "Error: Hyperparameter array '${arr_name}' is empty." >&2
            return 1
        fi
        if [[ $product -eq 0 ]]; then
            echo "Error: Product of lengths is zero, cannot divide." >&2
            return 1
        fi
        local index=$(( (remainder / product) % len ))
        indices[i]=$index
    done
}
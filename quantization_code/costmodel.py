# file used to compute inference costs
def get_cost(
        model, 
        num_input_tokens, 
        num_output_tokens, # assume this is a list of num tokens for each output
        weight_precision,
        activation_precision,
        groupsize,
    ):

    # get model params
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # get num_bytes_per_param (assume asymmetric quant)
    if weight_precision == 16:
        bytes_per_weight = 2
    else:
        bytes_per_weight = (groupsize * weight_precision + 2*16) / groupsize
    if activation_precision == 16:  
        bytes_per_activation = 2
    else:
        bytes_per_activation = (groupsize * activation_precision + 2*16) / groupsize

    # get model size in bytes
    layer_size = 4 * hidden_dim * hidden_dim + 3 * hidden_dim * intermediate_dim
    model_size = layer_size * num_layers
    model_size_bytes = model_size * bytes_per_weight

    # get number of weight memory ops
    num_model_loads = max(num_output_tokens) + 1 # add 1 for prefill
    model_weight_mem_ops = num_model_loads * model_size_bytes
    
    # get KV cache size in bytes
    generation_kv_mem_ops = 0
    for i in range(num_output_tokens):
        output_tokens = num_output_tokens[i]
        num_tokens_loaded = output_tokens * (output_tokens+1) / 2
        generation_kv_cache_size_bytes = 2 * num_layers * head_dim * num_kv_heads * num_tokens_loaded * bytes_per_activation
        generation_kv_mem_ops += generation_kv_cache_size_bytes

    # account for prefill - need to load the prefill KV cache at each step
    prefill_kv_cache_size_bytes = 2 * num_layers * head_dim * num_kv_heads * num_input_tokens * bytes_per_activation
    num_prefill_loads = max(num_output_tokens)
    prefill_kv_mem_ops = num_prefill_loads * prefill_kv_cache_size_bytes

    # total kv cache budget
    model_kv_mem_ops = generation_kv_mem_ops + prefill_kv_mem_ops

    # return sum and each separately
    return model_weight_mem_ops + model_kv_mem_ops, model_weight_mem_ops, model_kv_mem_ops
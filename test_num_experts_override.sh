#!/bin/bash

# Test script to verify num_experts_per_tok override works
# Usage: ./test_num_experts_override.sh [num_experts_per_tok]

NUM_EXPERTS_PER_TOK=${1:-4}  # Default to 4 if not provided

echo "Testing num_experts_per_tok override with value: $NUM_EXPERTS_PER_TOK"
echo "=========================================="

cd /home/rishabhtiwari/adaptive_reasoning/LLaMA-Factory/

# Test the argument parsing
echo "Testing argument parsing..."
python -c "
import sys
sys.path.append('src')
from llamafactory.hparams import get_train_args

# Test with command line override
args = {
    'config_file': '../train_configs/OpenThinker3_qwen3.yaml',
    'num_experts_per_tok': $NUM_EXPERTS_PER_TOK
}

try:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    print(f'✓ Successfully parsed num_experts_per_tok: {model_args.num_experts_per_tok}')
    
    # Test config loading and patching
    from llamafactory.model import load_tokenizer, load_config
    from llamafactory.model.patcher import patch_config
    
    tokenizer_module = load_tokenizer(model_args)
    config = load_config(model_args)
    
    print(f'Original config num_experts_per_tok: {getattr(config, \"num_experts_per_tok\", \"Not found\")}')
    
    # Apply patches
    patch_config(config, tokenizer_module['tokenizer'], model_args, {}, is_trainable=True)
    
    print(f'✓ Patched config num_experts_per_tok: {getattr(config, \"num_experts_per_tok\", \"Not found\")}')
    
    if getattr(config, 'num_experts_per_tok', None) == $NUM_EXPERTS_PER_TOK:
        print('✓ SUCCESS: num_experts_per_tok override is working correctly!')
    else:
        print('✗ FAILED: num_experts_per_tok override did not work')
        
except Exception as e:
    print(f'✗ Error: {e}')
"

echo "=========================================="
echo "Test completed!"

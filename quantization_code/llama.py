import time

import torch
import torch.nn as nn

from modelutils import *
from quant import *

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def quantize_kv(model):
    dev = model.device
    layers = model.model.layers

    # iterate over layers
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        # Find value projection layers
        v_layers = find_layers_v(layer)
        
        for name, v_layer in v_layers.items():
            print(f"Adding quantization hook to {name}")
            
            # Create quantizer for this layer
            quantizer = Quantizer()
            quantizer.configure(
                args.abits, perchannel=True, sym=False, mse=False, groupsize=args.groupsize
            )
            
            # Define quantization hook
            def quantize_hook(module, input, output):
                # Store original dtype
                original_dtype = output.dtype
                # Configure quantizer based on output statistics
                module.quantizer.find_params(output, weight=False)
                # Quantize the output and preserve original dtype
                quantized_output = quantize(output, module.quantizer.scale, module.quantizer.zero, module.quantizer.maxq)
                return quantized_output.to(original_dtype)
            
            # Register the hook
            v_layer.quantizer = quantizer
            v_layer.register_forward_hook(quantize_hook)
        
        # Find attention modules for key quantization after RoPE
        attn_modules = find_attention_modules(layer)
        
        for name, attn_module in attn_modules.items():
            # Create quantizer for keys
            quantizer = Quantizer()
            quantizer.configure(
                args.abits, perchannel=True, sym=False, mse=False, groupsize=args.groupsize
            )
            
            # Store original forward method
            original_forward = attn_module.forward
            
            def create_key_quantized_forward(orig_forward, quantizer):
                def key_quantized_forward(self, *args, **kwargs):
                    # Call original forward but intercept to quantize keys after RoPE
                    import types
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    
                    # Store original apply_rotary_pos_emb
                    original_rope = apply_rotary_pos_emb
                    
                    def quantized_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                        # Apply original RoPE
                        q_rope, k_rope = original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
                        
                        # Quantize the key states after RoPE
                        original_dtype = k_rope.dtype
                        quantizer.find_params(k_rope, weight=False)
                        k_rope_quantized = quantize(k_rope, quantizer.scale, quantizer.zero, quantizer.maxq)
                        k_rope = k_rope_quantized.to(original_dtype)
                        
                        return q_rope, k_rope
                    
                    # Temporarily replace the function
                    import transformers.models.llama.modeling_llama
                    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = quantized_rope
                    
                    try:
                        # Call original forward
                        result = orig_forward(*args, **kwargs)
                    finally:
                        # Restore original function
                        transformers.models.llama.modeling_llama.apply_rotary_pos_emb = original_rope
                    
                    return result
                
                return key_quantized_forward
            
            # Replace the forward method
            attn_module.key_quantizer = quantizer
            attn_module.forward = create_key_quantized_forward(original_forward, quantizer).__get__(attn_module, type(attn_module))

@torch.no_grad()
def quantize_model(model):
    dev = model.device
    layers = model.model.layers

    # iterate over layers
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:

            # apply simulated quantization
            quantizer = Quantizer()
            quantizer.configure(
                args.wbits, perchannel=True, sym=False, mse=False, groupsize=args.groupsize
            )
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = quantize(
                W, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(next(iter(layer.parameters())).dtype)

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # compute positional embeddings
    position_embeddings = model.model.rotary_emb(inps[0], position_ids)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for weight quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for activationquantization; use 16 for no KV quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=128,
        help='Groupsize to use for quantization; default uses full row.'
    )

    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    print('Quantizing KV cache...')
    if args.abits < 16:
        tick = time.time()
        quantize_kv(model)

    print('Quantizing Model...')
    if args.wbits < 16:
        tick = time.time()
        quantize_model(model)

    datasets = ['wikitext2']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

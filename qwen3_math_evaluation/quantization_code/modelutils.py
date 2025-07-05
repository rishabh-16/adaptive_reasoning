import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_layers_v(module, name=''):
    """Find layers that contain 'v_proj' in their name."""
    res = {}
    for name1, child in module.named_children():
        full_name = name + '.' + name1 if name != '' else name1
        if 'v_proj' in full_name:
            res[full_name] = child
        res.update(find_layers_v(child, name=full_name))
    return res


def find_attention_modules(module, name=''):
    """Find attention modules (self_attn) in the model."""
    res = {}
    for name1, child in module.named_children():
        full_name = name + '.' + name1 if name != '' else name1
        if 'self_attn' in full_name:
            res[full_name] = child
        res.update(find_attention_modules(child, name=full_name))
    return res



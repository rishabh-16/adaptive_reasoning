import numpy as np
import torch
import torch.nn as nn

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False, groupsize=-1
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.groupsize = groupsize
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        
        # Handle grouped quantization
        if self.groupsize > 0:
            if weight:
                # For weights (2D): group along last dimension (input features)
                assert(len(shape) == 2)
                # [A, B] -> [A, B/group_size, group_size] -> [A*B/group_size, group_size]
                x = x.reshape(x.shape[0], -1, self.groupsize)  # [A, B/group_size, group_size]
                x = x.reshape(-1, self.groupsize)  # [A*B/group_size, group_size]
            else:
                # For activations: group along the last dimension
                if len(shape) == 3:
                    # For 3D KV cache: [A, B, C] -> [A, B, C/group_size, group_size] -> [A*B*C/group_size, group_size]
                    x = x.reshape(*shape[:-1], -1, self.groupsize)  # [A, B, C/group_size, group_size]
                    x = x.reshape(-1, self.groupsize)  # [A*B*C/group_size, group_size]
                else:
                    assert(len(shape) == 4)
                    # For 4D KV cache: [A, B, C, D] -> [A, B, C, D/group_size, group_size] -> [A*B*C*D/group_size, group_size]
                    x = x.reshape(*shape[:-1], -1, self.groupsize)  # [A, B, C, D/group_size, group_size]
                    x = x.reshape(-1, self.groupsize)  # [A*B*C*D/group_size, group_size]
        else:
            assert(False)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        # zero guard
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        # Handle reshaping for different quantization modes
        if self.groupsize > 0:
            # Grouped quantization - need to broadcast group parameters back to original tensor shape
            num_groups_last_dim = (shape[-1] + self.groupsize - 1) // self.groupsize
            
            if weight:
                # For weights [A, B]: we have [A*B/group_size] scale/zero values
                # Reshape to [A, B/group_size] then expand to [A, B]
                num_groups_total = shape[0] * num_groups_last_dim
                self.scale = self.scale.reshape(shape[0], num_groups_last_dim, 1)
                self.zero = self.zero.reshape(shape[0], num_groups_last_dim, 1)
                
                # Expand to original shape
                self.scale = self.scale.repeat(1, 1, self.groupsize)
                self.zero = self.zero.repeat(1, 1, self.groupsize)
                
                # Final reshape
                self.scale = self.scale.reshape(shape[0], shape[1])
                self.zero = self.zero.reshape(shape[0], shape[1])
            else:
                # For activations: need to reshape back to original tensor shape
                if len(shape) == 3:
                    # For 3D [A, B, C]: we have [A*B*C/group_size] scale/zero values
                    # Reshape to [A, B, C/group_size] then expand to [A, B, C]
                    total_elements = shape[0] * shape[1] * num_groups_last_dim
                    self.scale = self.scale.reshape(shape[0], shape[1], num_groups_last_dim, 1)
                    self.zero = self.zero.reshape(shape[0], shape[1], num_groups_last_dim, 1)
                    
                    # Expand to original shape
                    self.scale = self.scale.repeat(1, 1, 1, self.groupsize)
                    self.zero = self.zero.repeat(1, 1, 1, self.groupsize)
                    
                    # Final reshape
                    self.scale = self.scale.reshape(shape)
                    self.zero = self.zero.reshape(shape)
                else:
                    assert(len(shape) == 4)
                    # For 4D [A, B, C, D]: we have [A*B*C*D/group_size] scale/zero values
                    # Reshape to [A, B, C, D/group_size] then expand to [A, B, C, D]
                    total_elements = shape[0] * shape[1] * shape[2] * num_groups_last_dim
                    self.scale = self.scale.reshape(shape[0], shape[1], shape[2], num_groups_last_dim, 1)
                    self.zero = self.zero.reshape(shape[0], shape[1], shape[2], num_groups_last_dim, 1)
                    
                    # Expand to original shape
                    self.scale = self.scale.repeat(1, 1, 1, 1, self.groupsize)
                    self.zero = self.zero.repeat(1, 1, 1, 1, self.groupsize)
                    
                    # Final reshape
                    self.scale = self.scale.reshape(shape)
                    self.zero = self.zero.reshape(shape)
        else:
            assert(False)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

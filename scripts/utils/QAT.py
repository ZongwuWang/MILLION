"""
Includes a differentiable function for PQ and a module to apply it.
This is used for quantization-aware training (QAT) in transformer models.
"""

import torch
from .pq_utils import sa_encode_4d_keops, sa_decode_4d

class PQFakeQuantFunc(torch.autograd.Function):
    """
    Define a differentiable fake quantization function for product quantization.
    It applies PQ quantization and dequantization in the forward pass, and STE in the backward pass.
    """
    
    @staticmethod
    def forward(ctx, x, codebooks):
        """
        x: key_states or value_states, shaped (bs, num_heads, n, d)
        codebooks: precomputed codebooks for quantization, shaped (M, c, d//M)
        """
        return sa_decode_4d(sa_encode_4d_keops(x, codebooks), codebooks)

    @staticmethod
    def backward(ctx, grad_output):
        # STE (Straight-Through Estimator) is used here
        return grad_output, None

class PQFakeQuantModule(torch.nn.Module):
    def __init__(self, codebooks):
        super().__init__()
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=False) # codebooks are static during fine-tuning

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, x):
        return PQFakeQuantFunc.apply(x, self.codebooks)
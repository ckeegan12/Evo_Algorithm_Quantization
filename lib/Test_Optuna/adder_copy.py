import torch
import torch.nn as nn
from torch.autograd import Function
import math
import adder_cuda_copy

class AdderFunction(Function):
    @staticmethod
    def forward(ctx, X, W, stride, padding):
        N, Ci, Hi, Wi = X.size()
        Co, _, K, _ = W.size()
        Ho = (Hi + 2 * padding - K) // stride + 1
        Wo = (Wi + 2 * padding - K) // stride + 1

        # Use our Fused Forward Kernel
        out_flat = torch.empty((Co, N * Ho * Wo), device=X.device)
        adder_cuda_copy.ADDER_CONV_FUSED_copy(X, W, out_flat, stride, padding)
        
        output = out_flat.view(Co, N, Ho, Wo).permute(1, 0, 2, 3).contiguous()

        if X.requires_grad or W.requires_grad:
            ctx.save_for_backward(X, W)
            ctx.stride = stride
            ctx.padding = padding

        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        # Ensure grad_output is contiguous for the CUDA kernel
        # Permute grad_output to match the (Co, N*Ho*Wo) layout used in the kernel
        grad_output_p = grad_output.permute(1, 0, 2, 3).contiguous().view(W.size(0), -1)
        
        grad_W = torch.zeros_like(W)
        grad_X = torch.zeros_like(X)
        
        # Call the new Fused Backward Kernel - NO UNFOLD/FOLD!
        adder_cuda_copy.ADDER_BACKWARD_FUSED_copy(grad_output_p, X, W, grad_W, grad_X, stride, padding)
        
        # AdderNet Gradient Normalization
        # We normalize the weight gradient to balance the learning scale
        norm = grad_W.norm(p=2).clamp(min=1e-12)
        grad_W = grad_W / norm * math.sqrt(W.numel()) / 5
        
        return grad_X, grad_W, None, None

class adder2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size

        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = AdderFunction.apply(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output

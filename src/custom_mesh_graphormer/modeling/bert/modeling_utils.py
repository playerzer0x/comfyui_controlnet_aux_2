from transformers.modeling_utils import *
import torch
import torch.nn as nn

# Try multiple import locations for Conv1D to handle different transformers versions
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    try:
        from transformers.modeling_utils import Conv1D
    except ImportError:
        try:
            from transformers import Conv1D
        except ImportError:
            # Fallback: define Conv1D if it's not available
            import torch.nn as nn
            class Conv1D(nn.Module):
                """
                1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
                Basically works like a linear layer but the weights are transposed.
                """
                def __init__(self, nf, nx):
                    super().__init__()
                    self.nf = nf
                    w = torch.empty(nx, nf)
                    nn.init.normal_(w, std=0.02)
                    self.weight = nn.Parameter(w)
                    self.bias = nn.Parameter(torch.zeros(nf))

                def forward(self, x):
                    size_out = x.size()[:-1] + (self.nf,)
                    x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
                    x = x.view(*size_out)
                    return x


def prune_linear_layer(layer, index, dim=0):
    """
    Prune a linear layer to keep only selected indices.
    
    Args:
        layer: The linear layer to prune (torch.nn.Linear)
        index: The indices to keep (torch.LongTensor)
        dim: The dimension to prune (0 for input, 1 for output)
    
    Returns:
        The pruned linear layer
    """
    index = index.to(layer.weight.device)
    
    W = layer.weight.index_select(dim, index).clone().detach()
    
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.index_select(0, index).clone().detach()
        else:
            b = layer.bias.clone().detach()
    else:
        b = None
    
    if dim == 0:
        new_size = (index.shape[0], layer.weight.shape[1])
    else:
        new_size = (layer.weight.shape[0], index.shape[0])
    
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.data = W.contiguous()
    if b is not None:
        new_layer.bias.data = b.contiguous()
    
    return new_layer

# Backwards compatibility alias: older code imports `prune_layer`
prune_layer = prune_linear_layer
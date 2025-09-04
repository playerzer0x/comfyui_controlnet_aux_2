from transformers.modeling_utils import *
import torch
import torch.nn as nn


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
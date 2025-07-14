import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def module_types_to_param_keys(model, module_types=[nn.Conv2d, nn.Linear]):
    keys = []
    for name, module in model.named_modules():
        for module_type in module_types:
            if isinstance(module, module_type):
                keys.append(name + '.weight')
                if module.bias is not None:
                    keys.append(name + '.bias')
                 
                
    return keys

  
def fuse_all_conv_bn(model: nn.Module, verbose=False) -> nn.Module:
    """
    Recursively fuses all Conv2d and BatchNorm2d layers in a model.

    This function modifies the model in-place and sets it to evaluation mode.
    
    The fusion pattern is specifically for a Conv2d layer followed immediately
    by a BatchNorm2d layer.

    Args:
        model (nn.Module): The model to be fused.

    Returns:
        nn.Module: The fused model (same as the input model, modified in-place).
    """
    # 1. Set the model to evaluation mode. This is required for fusion.
    model.eval()

    # 2. Recursively fuse children modules first (post-order traversal)
    for name, child in model.named_children():
        fuse_all_conv_bn(child)

    # 3. Fuse layers at the current level
    # We iterate through the module names and check for Conv-BN patterns
    # We use a list of module names to handle changes while iterating
    module_names = list(model._modules.keys())
    
    # Iterate backwards to safely modify the module dictionary
    for i in range(len(module_names) - 2, -1, -1):
        current_name = module_names[i]
        next_name = module_names[i+1]
        
        current_module = model._modules[current_name]
        next_module = model._modules[next_name]

        # Check for the Conv2d -> BatchNorm2d pattern
        if isinstance(current_module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
            if verbose:
                print(f"Fusing '{current_name}' (Conv2d) and '{next_name}' (BatchNorm2d)")
            
            # Fuse the two modules
            fused_conv = fuse_conv_bn_eval(current_module, next_module)
            
            # Replace the original Conv2d with the fused module
            model._modules[current_name] = fused_conv
            
            # Replace the BatchNorm2d with an Identity layer. This is crucial
            # to preserve the model's structure and forward pass logic,
            # especially in nn.Sequential containers.
            model._modules[next_name] = nn.Identity()
            
    return model 

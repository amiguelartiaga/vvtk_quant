import torch
import numpy as np
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import pandas as pd

def quant(x, levels):
    distance = (x[..., None] - levels).abs()          # shape: (*sig_shape, n_levels)
    indices = distance.argmin(dim=-1)                 # shape: *sig_shape
    xq = levels[indices]                              # shape: *sig_shape
    return xq, indices
def dquant(x, levels, temp=1.0, eps=1e-3):
    """
    quantization with a loop.
    """
    if len(levels) < 2:
        # If less than 2 levels, return the constant level value broadcasted to x's shape
        return torch.full_like(x, levels[0].item())

    limits = (levels[1:] - levels[:-1]) / 2 + levels[:-1]

    # Initialize xq as a tensor with the same shape as x, filled with the first level's value.
    xq = torch.full_like(x, levels[0].item())

    # Iteratively add the transitions from each level to the next.
    for i in range(len(levels) - 1):
        delta = levels[i+1] - levels[i]
        xq += torch.sigmoid(temp * (x - limits[i] - eps)) * delta

    return xq
def dquant_fast(x, levels, temp=1.0, eps=1e-3):
    """
    Fully vectorized fast soft quantization.
    This version is an approximation that only considers the two nearest
    transition boundaries for each point. It is mathematically equivalent
    to the corrected dquant for high temp values.
    """
    L = len(levels)
    if L < 2:
        return levels.repeat(*x.shape)

    # Calculate the midpoints between levels (the transition boundaries).
    limits = (levels[:-1] + levels[1:]) / 2.0

    # Find the hard quantization index for each input value.
    with torch.no_grad():
        indices = torch.searchsorted(limits, x)

    # Get the central, left, and right quantization levels for each input.
    level_c = levels[indices]
    level_l = levels[torch.clamp(indices - 1, min=0)]
    level_r = levels[torch.clamp(indices + 1, max=L - 1)]

    # Initialize sigmoid outputs.
    s_left = torch.zeros_like(x)
    s_right = torch.zeros_like(x)

    # --- Calculate left sigmoid (transition from the left bin) ---
    # This sigmoid should be near 1 if x is far from the left boundary.
    mask_left = indices > 0
    if mask_left.any():
        limit_vals = limits[indices[mask_left] - 1]
        s_left[mask_left] = torch.sigmoid(
            temp * (x[mask_left] - limit_vals - eps)
        )

    # --- Calculate right sigmoid (transition to the right bin) ---
    # This sigmoid should be near 0 if x is far from the right boundary.
    mask_right = indices < (L - 1)
    if mask_right.any():
        limit_vals = limits[indices[mask_right]]
        s_right[mask_right] = torch.sigmoid(
            temp * (x[mask_right] - limit_vals - eps)
        )

    # Calculate the change in value contributed by each neighbor.
    delta_left = level_c - level_l
    delta_right = level_r - level_c

    # The final value is an interpolation based on the sigmoids.
    # Start at the hard level `level_c`.
    # Subtract the influence from the left: `delta_left * (1 - s_left)`
    # Add the influence from the right: `delta_right * s_right`
    xq = level_c - delta_left * (1 - s_left) + delta_right * s_right

    return xq

def equal_probability_bins(pmf, x=None, k=8):
    """
    Compute edges for equal probability quantization.
    Args:
        pmf (array-like): Probability mass function.
        x (array-like, optional): Values corresponding to the pmf. If None, defaults to range(len(pmf)).
        k (int): Number of quantization levels.
    Returns:
        edges (array): Quantization edges.
    """
    
    pmf = np.asarray(pmf, dtype=float)
    pmf /= pmf.sum()

    if x is None:
        x = np.arange(len(pmf), dtype=float)

    cdf = np.cumsum(pmf)
    quantiles = np.linspace(0, 1, k + 1)
    edges = np.interp(quantiles,
                      np.concatenate(([0.0], cdf)),
                      np.concatenate(([x[0]], x)))
    return edges
def calibrate_quant_v0(model, qkeys=None, b=8):
    """
    Static quantization function for a model. Single quantization edge for all parameters.

    Args:
        model (nn.Module): The model to be quantized.
        b (int): Bit-width for quantization (default is 8).
        
    Returns:
        edge_dict (dict): Dictionary containing quantization edges for each layer.
    """    
    state_dict = model.state_dict()    
    if qkeys is None:
        qkeys = [k for k in state_dict.keys() if 
            any(param_name in k for param_name in ['weight', 'bias'])]  
     
    p_ = []    
    for key, param in state_dict.items():
        if key in qkeys:
            p = param.data.cpu().numpy().flatten()             
            p_ += p.tolist()
                
    h, x = np.histogram(p_, bins=512)
    prob =  h/(np.sum(h)+1e-3)
    e = equal_probability_bins(prob,x=x[1:], k=2**b)
      
    edge_dict = {}
    for key in state_dict.keys():    
        if not key in qkeys:
            continue        
        edge_dict[key] = e[1:]
    
    return edge_dict
def calibrate_quant_v1(model, qkeys=None, b=8):
    """
    Static quantization function for a model. Quantization edge for each parameter tensor.

    Args:
        model (nn.Module): The model to be quantized.
        b (int): Bit-width for quantization (default is 8).
        
    Returns:
        edge_dict (dict): Dictionary containing quantization edges for each layer.
    """    
    state_dict = model.state_dict()    
    if qkeys is None:
        qkeys = [k for k in state_dict.keys() if 
            any(param_name in k for param_name in ['weight', 'bias'])]  
     
    edge_dict = {}
    for key, param in state_dict.items():
        if key in qkeys:
            p = param.data.cpu().numpy().flatten()
            h, x = np.histogram(p, bins=512)
            prob =  h/(np.sum(h)+1e-3)
            e = equal_probability_bins(prob,x=x[1:], k=2**b)
               
            edge_dict[key] = e[1:]
    
    return edge_dict
def calibrate_quant_v2(model, qkeys=None, b=8, K=8,
                       bins=64, hrange=(-2, 2)):
    """
    Static quantization function for a model. Clusters parameters using k-means
    and computes quantization edges for each cluster.

    Args:
        model (nn.Module): The model to be quantized.
        b (int): Bit-width for quantization (default is 8).
        K (int): Number of clusters for k-means (default is 4).
        param_names (list): List of parameter names to consider for quantization (default is ['weight', 'bias']).
        bins (int): Number of bins for histogram (default is 64).
        hrange (tuple): Range for histogram (default is (-2, 2)).
        
    Returns:
        edge_dict (dict): Dictionary containing quantization edges for each layer.
    """

    state_dict = model.state_dict()    
    if qkeys is None:
        qkeys = [k for k in state_dict.keys() if 
            any(param_name in k for param_name in ['weight', 'bias'])]  
    h_ = []
    p_ = []
    id = {}
    for key, param in state_dict.items():
        if key in qkeys:
            p = param.data.cpu().numpy().flatten()
            h, x = np.histogram(p, bins=bins, range=hrange)     
            h = h / (np.sum(h)+1e-3)  # Normalize histogram       
            id[key] = len(h_)
            h_.append(h)
            p_.append(p.tolist())
            
    H = np.stack(h_)    
    kmeans = KMeans(n_clusters=K, random_state=0).fit(H)
    indices = kmeans.labels_
    pk = defaultdict(list)
    for i, p in enumerate(p_):
        pk[indices[i]] += p
        
    e_ = []
    for i in range(K):
        h, x = np.histogram(pk[i], bins=512)
        prob =  h/(np.sum(h)+1e-3)
        e = equal_probability_bins(prob,x=x[1:], k=2**b)
        e_.append(e.tolist())
    
    edge_dict = {}
    for key in state_dict.keys():    
        if not key in qkeys:
            continue
        e = e_[indices[id[key]]]  # Get the quantization edges for this parameter
        edge_dict[key] = e[1:]
    
    return edge_dict

def calibrate_quant_v3(model, qkeys=None, b=8, csv=None):
    """
    Static quantization function for a model. Single quantization edge for all parameters.

    Args:
        model (nn.Module): The model to be quantized.
        b (int): Bit-width for quantization (default is 8).
        csv: Path to a CSV file for saving quantization edges (default is None).
    Returns:
        edge_dict (dict): Dictionary containing quantization edges for each layer.
    """    
    state_dict = model.state_dict()    
    if qkeys is None:
        qkeys = [k for k in state_dict.keys() if 
            any(param_name in k for param_name in ['weight', 'bias'])]  
     
    #read csv
    df = pd.read_csv(csv)
    p_ = sorted(df['value'].tolist())
    print(p_) 
    edge_dict = {}
    for key in state_dict.keys():    
        if not key in qkeys:
            continue        
        edge_dict[key] = p_
    
    return edge_dict
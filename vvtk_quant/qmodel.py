import torch
from .quant import calibrate_quant_v0, calibrate_quant_v1, calibrate_quant_v2, calibrate_quant_v3, quant
from .utils import fuse_all_conv_bn, module_types_to_param_keys
from .io import pack, unpack
import gzip, pickle
import numpy as np

from .qmodules import QConv2d, QLinear

class VVTK_QModel(torch.nn.Module):
    def __init__(self, model, b=None, temp=10, fuse_conv_bn=False, module_types=[torch.nn.Conv2d, torch.nn.Linear]  ):
        super(VVTK_QModel, self).__init__()
        self.model = model
        self.b = b
        self.edge_diqct = {}
        self.ind_dict = {}
        self.temp = temp
        self.module_types = module_types
        self.qkeys = module_types_to_param_keys(model, module_types)
        print(f"Quantization keys: {self.qkeys}")
        
        if fuse_conv_bn:
            self.fuse_conv_bn(verbose=True)
        
    def forward(self, x):
        return self.model(x)
        
    def fuse_conv_bn(self, verbose=False):
        self.model = fuse_all_conv_bn(self.model, verbose=verbose)
        
    def calibrate_quant(self, v=1, b=None, K=None, csv=None):
        self.b = b if b is not None else self.b
        assert self.b is not None, "Bit width 'b' must be specified for quantization."   
        if v==0:            
            self.edge_dict = calibrate_quant_v0(self.model, qkeys=self.qkeys, b=self.b)
        elif v==1:
            self.edge_dict = calibrate_quant_v1(self.model, qkeys=self.qkeys, b=self.b)
        elif v==2:
            self.edge_dict = calibrate_quant_v2(self.model, qkeys=self.qkeys, b=self.b, K=K)
        elif v==3:
            self.edge_dict = calibrate_quant_v3(self.model, qkeys=self.qkeys, b=self.b, csv=csv)
        
        
        print(f"Quantization edges calibrated with version {v}, bit width {self.b}, edge dictionary size: {len(self.edge_dict)}.")

    def apply_quant(self): 
        device = next(self.model.parameters()).device
        if len(self.edge_dict) == 0:
            raise ValueError("Edge dictionary is empty. Please calibrate quantization first.")
        qkeys = list(self.edge_dict.keys())             
        state = self.model.state_dict()
        qstate = {}               
        for k, v in state.items():    
            if not k in qkeys:
                # print(f"Skipping {k} with dtype {v.dtype}")
                qstate[k] = v
                continue    
            
            e = self.edge_dict[k]
            e = torch.tensor(e, dtype=torch.float32).to(device)
            vq, ind = quant(v, e)
            v = e[ind]
            ind = ind.flatten().cpu().numpy().tolist()
            self.ind_dict[k] = ind
            state[k] = v 
        self.model.load_state_dict(state, strict=False)
        
    def save(self, path):
        if len(self.ind_dict) == 0:
            raise ValueError("Edge dictionary is empty. Please calibrate ans apply quantization first.")
        state = self.model.state_dict()
        qkeys = list(self.edge_dict.keys())     
        quantized_state = {}          
        for k, v in state.items():    
            if not k in qkeys:
                # print(f"Skipping {k} with dtype {v.dtype}")
                quantized_state[k] = v
                continue    
            s = v.shape 
            ind = self.ind_dict[k]
            
            packed = pack(ind, self.b)
            quantized_state[k] = [s, packed]
        
        with gzip.open(path, "wb") as f:
            pickle.dump((self.b, quantized_state, self.edge_dict), f)

    def load(self, path):
        device = next(self.model.parameters()).device
        with gzip.open(path, "rb") as f:
            self.b, qstate_packed, self.edge_dict = pickle.load(f)
        
        qstate_unpacked = {}
        for k, v in qstate_packed.items():
            if isinstance(v, list):
                s, v_packed = v
                ind = unpack(v_packed, self.b, np.prod(s))
                ind = torch.tensor(ind).reshape(s)
                e = self.edge_dict[k] 
                e = torch.tensor(e, dtype=torch.float32).to(device)
                v = e[ind]
                qstate_unpacked[k] = v.to(device)
            else:
                qstate_unpacked[k] = v.to(device)
        
        self.model.load_state_dict(qstate_unpacked, strict=True)
        print(f"Quantized model loaded from {path} with bit width {self.b} and edge dictionary size {len(self.edge_dict)}.")
        
    def update_temp(self, temp, verbose=False):        
        """
        Recursively update the temperature for all quantization layers in the model if temp attribute exists.
        Args:
            model (nn.Module): The model containing quantization layers.
            temp (float): The new temperature value to set.
        """
        for name, module in self.model.named_modules():            
            if hasattr(module, 'temp'):
                if verbose:
                    print(f"Updating temperature for {name} to {temp}")
                module.temp = temp
    
    def replace_qmodules(self):
        """
        Replace all quantization modules in the model with their quantized versions.
        This is useful for inference after calibration and quantization.
        """
        device = next(self.model.parameters()).device
        replace_conv2d = torch.nn.Conv2d in self.module_types
        replace_linear = torch.nn.Linear in self.module_types
         
        for name, module in self.model.named_modules():
            if replace_conv2d and isinstance(module, torch.nn.Conv2d):
                print(f"Replacing {name} with QConv2d")
                e1 = self.edge_dict[name + '.weight']
                e2 = self.edge_dict.get(name + '.bias', None)
                e1 = torch.tensor(e1, dtype=torch.float32).to(device)
                e2 = torch.tensor(e2, dtype=torch.float32).to(device) if e2 is not None else None
                
                # Split the name to locate the parent
                path = name.split(".")
                if len(path) == 1:
                    # top-level module
                    setattr(self.model, name, QConv2d(module, 
                                                      levels_weight=e1, 
                                                      levels_bias=e2, 
                                                      temp=self.temp))
                else:
                    parent = self.model
                    for p in path[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, path[-1], QConv2d(module, 
                                                      levels_weight=e1, 
                                                      levels_bias=e2, 
                                                      temp=self.temp))

            elif replace_linear and isinstance(module, torch.nn.Linear):
                print(f"Replacing {name} with QLinear")
                e1 = self.edge_dict[name + '.weight']
                e2 = self.edge_dict.get(name + '.bias', None)
                e1 = torch.tensor(e1, dtype=torch.float32).to(device)
                e2 = torch.tensor(e2, dtype=torch.float32).to(device) if e2 is not None else None
                
                
                
                path = name.split(".")
                if len(path) == 1:
                    setattr(self.model, name, QLinear(module, 
                                                      levels_weight=e1, 
                                                      levels_bias=e2, 
                                                      temp=self.temp))
                else:
                    parent = self.model
                    for p in path[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, path[-1], QLinear(module, 
                                                      levels_weight=e1, 
                                                      levels_bias=e2, 
                                                      temp=self.temp))
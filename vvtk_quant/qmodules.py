
import torch

from .quant import dquant_fast, quant

class QConv2d(torch.nn.Conv2d):
    def __init__(self, module: torch.nn.Conv2d,
                levels_weight=None,
                levels_bias=None,
                temp=10.0):
        # Call super with same parameters as the original module
        super().__init__(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype
        )

        # Copy weights and bias data
        with torch.no_grad():
            self.weight.copy_(module.weight)
            if module.bias is not None:
                self.bias.copy_(module.bias)

        self.levels_weight = levels_weight
        self.levels_bias = levels_bias
        self.temp = temp
        
    def forward(self, x):
        weight = self.weight
        # print('weight before:', weight)
        weight = dquant_fast(weight, self.levels_weight, temp=self.temp)
        # print('weight after:', weight)

        bias = self.bias
        if bias is not None:
            print('bias before:', bias)
            bias = dquant_fast(bias, self.levels_bias, temp=self.temp)
            print('bias after:', bias)
            
        # explicit functional call
        return torch.nn.functional.conv2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    def apply_quant(self):
        self.weight.data = quant(self.weight, self.levels_weight)[0]
        if self.bias is not None:
            self.bias.data = quant(self.bias, self.levels_bias)[0]

class QLinear(torch.nn.Linear):
    def __init__(self, module: torch.nn.Linear,
                levels_weight=None,
                levels_bias=None,
                temp=10.0):
        super().__init__(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        
        # Copy weights and bias
        with torch.no_grad():
            self.weight.copy_(module.weight)
            if module.bias is not None:
                self.bias.copy_(module.bias)

        self.levels_weight = levels_weight
        self.levels_bias = levels_bias
        self.temp = temp
        
    def forward(self, x):
        weight = self.weight
        # print('weight before:', weight)
        weight = dquant_fast(weight, self.levels_weight, temp=self.temp)
        # print('weight after:', weight)
        
        bias = self.bias
        if bias is not None:
            print('bias before:', bias)
            bias = dquant_fast(bias, self.levels_bias, temp=self.temp)
            print('bias after:', bias)
            
        return torch.nn.functional.linear(
            x,
            weight,
            bias
        )
        
    def apply_quant(self):
        self.weight.data = quant(self.weight, self.levels_weight)[0]
        if self.bias is not None:
            self.bias.data = quant(self.bias, self.levels_bias)[0]
            
#--------------------------------------------------

if __name__ == "__main__":
    levels_weight = torch.tensor([0.25, 0.5, 0.75, 1.0])
    levels_bias = torch.tensor([0.25, 0.5, 0.75, 1.0])
    temp=100.0
    
    # Example usage of QConv2d
    model = torch.nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=True)
    model.weight.data.fill_(0.6)  # Set weight to a constant for testing
    model.bias.data.fill_(0.8)  # Set bias to a constant for testing
    qmodel = QConv2d(model, 
                    levels_weight=levels_weight,
                    levels_bias=levels_bias, 
                    temp=temp)
    print("QConv2d weight:", qmodel.weight   )
    x = torch.linspace(0, 1, 4).view(1, 1, 1, 4)
    
    print("-"*50)
    print("Input:", x)
    output = model(x)
    print("Output of Conv2d:", output)
    print("-"*50)
    print("Input:", x)
    output = qmodel(x)
    print("Output of QConv2d:", output)
    print("-"*50)
    qmodel.apply_quant()
    print("After applying quantization:")
    print("Input:", x)
    output = qmodel(x)
    print("Output of QConv2d*:", output)
    print("-"*50)
    
    
    # Example usage of QLinear
    model = torch.nn.Linear(1, 1, bias=True)
    
    model.weight.data.fill_(0.6)  # Set weight to a constant for
    model.bias.data.fill_(0.8)  # Set bias to a constant for testing
    
    qmodel = QLinear(model,
                    levels_weight=levels_weight,
                    levels_bias=levels_bias, 
                    temp=temp)
    print("QLinear weight:", qmodel.weight)
    x = torch.linspace(0, 1, 4).view(4, 1)
    print("-"*50)
    print("Input:", x)
    output = model(x)
    print("Output of Linear:", output)
    print("-"*50)
    print("Input:", x)
    output = qmodel(x)
    print("Output of QLinear:", output)
    print("-"*50)
    qmodel.apply_quant()
    print("After applying quantization:")
    print("Input:", x)
    output = qmodel(x)
    print("Output of QConv2d*:", output)
    print("-"*50)
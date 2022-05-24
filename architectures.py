import torch
from functools import partial
import importlib
if importlib.util.find_spec('torch.cuda'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"
def initialize_weights(m, init_type = "xavier_uniform_", gain_type='relu', init_seed=False):
    if type(m) == torch.nn.Linear:
        if type(init_seed) == type(0) and device != "cpu":
            torch.cuda.manual_seed_all(init_seed)
        getattr(torch.nn.init, init_type)(m.weight, gain=torch.nn.init.calculate_gain(gain_type))
        
class DenseLayers(torch.nn.Module):#Basic Feed Forward Dense MLP
    def __init__(self, 
                 n_features, 
                 width = 1024, 
                 depth = 2, 
                 output = 0,#Number of neurons on output layer
                 activation = "ReLU", 
                 dropout = 0.,
                 batch_norm = False,
                 initializer_params = {}, 
                 device = device):
        super(DenseLayers, self).__init__()
        
        layers = [torch.nn.Linear(n_features, width, device = device),  getattr(torch.nn, activation)()]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        for layer in range(1,depth):
            layers += [torch.nn.Linear(width, width, device = device),  getattr(torch.nn, activation)()]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        if output:
            layers += [torch.nn.Linear(width, output, device = device)]
        self.model = torch.nn.Sequential(*layers)
        self.apply(partial(initialize_weights,**initializer_params))
    
    def forward(self, activation):
        return self.model.forward(activation) 
    
class ResidualLayers(torch.nn.Module):#Resblock network inspired by https://github.com/yandex-research/rtdl/blob/main/bin/resnet.py
    def __init__(self, 
                 n_features, 
                 width = 1024, 
                 depth = 2, 
                 output = 0,#Number of neurons on output layer
                 block_depth = 1,
                 activation = "ReLU", 
                 dropout = 0.,
                 batch_norm = False,
                 initializer_params = {}, 
                 device = device):
        super(ResidualLayers, self).__init__()
        
        layers = [torch.nn.Linear(n_features, width, device = device),  getattr(torch.nn, activation)()]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        for layer in range(1,depth):
            layers += [ResBlock(width = width, 
                 block_depth = block_depth,
                 activation = "ReLU", 
                 dropout = dropout,
                 batch_norm = batch_norm,
                 initializer_params = initializer_params, 
                 device = device)]
        if output:
            layers += [torch.nn.Linear(width, output, device = device)]
        self.model = torch.nn.Sequential(*layers)
        self.apply(partial(initialize_weights,**initializer_params))
    
    def forward(self, activation):
        return self.model.forward(activation)
    
class ResBlock(torch.nn.Module):
    def __init__(self,
                 width = 1024, 
                 block_depth = 2,
                 activation = "ReLU", 
                 dropout = 0.,
                 batch_norm = False,
                 initializer_params = {}, 
                 device = device):
        super(ResBlock, self).__init__()
        layers = []
        for layer in range(block_depth):
            layers += [torch.nn.Linear(width, width, device = device),  getattr(torch.nn, activation)()]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        self.model = torch.nn.Sequential(*layers)
        self.apply(partial(initialize_weights,**initializer_params))

    def forward(self, activation):
        return self.model.forward(activation)+activation
    
class GLULayers(torch.nn.Module):#Gated Linear Units
    def __init__(self, 
                 n_features, 
                 width = 1024, 
                 depth = 2, 
                 output = 0,#Number of neurons on output layer
                 gate_activation = "Sigmoid",
                 output_activation = "ReLU",
                 dropout = 0.,
                 batch_norm = False,
                 gate_initializer_params = {"gain_type":'sigmoid'},
                 linear_initializer_params = {},
                 output_initializer_params = {}, 
                 device = device):
        super(GLULayers, self).__init__()
        
        layers = [GatedLinearUnit(n_features, width, 
                                  gate_activation = gate_activation, 
                                  gate_initializer_params = gate_initializer_params,
                                  linear_initializer_params = linear_initializer_params,
                                  device = device)]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        for layer in range(1,depth-1):
            layers += [GatedLinearUnit(width, width, 
                                  gate_activation = gate_activation, 
                                  gate_initializer_params = gate_initializer_params,
                                  linear_initializer_params = linear_initializer_params,
                                  device = device)]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        layers += [torch.nn.Linear(width, width, device = device),  getattr(torch.nn, output_activation)()]
        initialize_weights(layers[-2],**output_initializer_params)
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        if output:
            layers += [torch.nn.Linear(width, output, device = device)]
            initialize_weights(layers[-1],**output_initializer_params)
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, activation):
        return self.model.forward(activation)
    
class ResidualGLULayers(torch.nn.Module):#ResNet architecture with GLU on each layer
    def __init__(self, 
                 n_features, 
                 width = 1024, 
                 depth = 2, 
                 output = 0,#Number of neurons on output layer
                 block_depth = 1,
                 gate_activation = "Sigmoid",
                 dropout = 0.,
                 batch_norm = False,
                 gate_initializer_params = {"gain_type":'sigmoid'},
                 linear_initializer_params = {},
                 device = device):
        super(ResidualGLULayers, self).__init__()
        
        layers = [GatedLinearUnit(n_features, width, 
                                  gate_activation = gate_activation, 
                                  gate_initializer_params = gate_initializer_params,
                                  linear_initializer_params = linear_initializer_params,
                                  device = device)]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        for layer in range(1,depth):
            layers += [ResGLUBlock(width = width, 
                 block_depth = block_depth,
                gate_activation = gate_activation, 
                gate_initializer_params = gate_initializer_params,
                linear_initializer_params = linear_initializer_params,
                 dropout = dropout,
                 batch_norm = batch_norm,
                 device = device)]
        if output:
            layers += [torch.nn.Linear(width, output, device = device)]
            initialize_weights(layers[-1],**output_initializer_params)
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, activation):
        return self.model.forward(activation)
    
class ResGLUBlock(torch.nn.Module):
    def __init__(self,
                 width = 1024, 
                 block_depth = 2,
                 dropout = 0.,
                 batch_norm = False,
                 gate_activation = "Sigmoid",
                 gate_initializer_params = {"gain_type":'sigmoid'},
                 linear_initializer_params = {},
                 device = device):
        super(ResGLUBlock, self).__init__()
        layers = []
        for layer in range(block_depth):
            layers += [GatedLinearUnit(width, width, 
                                  gate_activation = gate_activation, 
                                  gate_initializer_params = gate_initializer_params,
                                  linear_initializer_params = linear_initializer_params,
                                  device = device)]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, activation):
        return self.model.forward(activation)+activation
    
class GatedLinearUnit(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 gate_activation = "Sigmoid",
                 linear_initializer_params = {},
                 gate_initializer_params = {"gain_type":'sigmoid'},
                 device = device
                ):
        super(GatedLinearUnit, self).__init__()
        self.linear_layer = torch.nn.Linear(input_dim, output_dim, device = device)
        self.gate_layer = torch.nn.Linear(input_dim, output_dim, device = device)

        initialize_weights(self.linear_layer, **linear_initializer_params)
        initialize_weights(self.gate_layer, **gate_initializer_params)
        self.gate_activation = getattr(torch.nn, gate_activation)()
        
    def forward(self, activation):
        return self.linear_layer(activation) * self.gate_activation(self.gate_layer(activation))
    
class CustomDenseLayers(torch.nn.Module):
    def __init__(self,
                 n_features, 
                 hidden_layers = (1024,1024),
                 output = 0,#Number of neurons on output layer
                 activation = "ReLU", 
                 dropout = 0.,
                 batch_norm = False,
                 initializer_params = {}, 
                 device = device):
        super(CustomDenseLayers, self).__init__()
        
        layers = [torch.nn.Linear(n_features, hidden_layers[0], device = device),  getattr(torch.nn, activation)()]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(hidden_layers[layer], device = device)]
        for layer in range(1,len(hidden_layers)):
            layers += [torch.nn.Linear(hidden_layers[layer-1], hidden_layers[layer], device = device),  getattr(torch.nn, activation)()]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(hidden_layers[layer], device = device)]
        if output:
            layers += [torch.nn.Linear(hidden_layers[-1], output, device = device)]
        self.model = torch.nn.Sequential(*layers)
        self.apply(partial(initialize_weights,**initializer_params))
    
    def forward(self, activation):
        return self.model.forward(activation)
    
class BasicConvNet(torch.nn.Module): # from github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, 
                 output = 0,#Number of neurons on output layer
                 dropout = False,
                 batch_norm = False,
                 device = device):
        super(BasicConvNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, device = device)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, device = device)
        self.dropout1 = torch.nn.Dropout(0.25) if dropout else None
        self.dropout2 = torch.nn.Dropout(0.5) if dropout and output else None
        self.fc1 = torch.nn.Linear(9216, 128, device = device)
        self.fc2 = torch.nn.Linear(128, output, device = device) if output else None
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        if self.dropout1: x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        if self.dropout2: x = self.dropout2(x)
        if self.fc2: x = self.fc2(x)
        return x
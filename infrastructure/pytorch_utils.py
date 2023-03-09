import torch
import torch.nn as nn
from typing import Union
from collections import OrderedDict



Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
    'softmax' : nn.Softmax(dim = -1),
}

def build_mlp(input_size: int, output_size: int, n_layers: int, size: int, activation: Activation = "tanh", output_activation: Activation = "identity", init_method = None) -> nn.Module:

    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for _ in range(n_layers):
        current_layer = nn.Sequential(OrderedDict([
            ("fully_connected", nn.Linear(input_size, size)),
            ("nonlienarity", activation)]))
        
        if init_method is not None:
            current_layer.apply(init_method)

        layers.append(current_layer)
        input_size = size

    final_layer = nn.Sequential(
        OrderedDict([
            ("output_layer", nn.Linear(size, output_size)),
            ("output_activation", output_activation)
        ])
    )

    if init_method is not None:
        final_layer.apply(init_method)

    layers.append(final_layer)

    return nn.Sequential(*layers)






import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print("Warning: No GPU deteced. Running on CPU.")

class OceanStateEncoder(nn.Module):

    def __init__(self, input_shape, num_filters, kernel_size=(3,3)):
        super(OceanStateEncoder, self).__init__()

        self.input_shape = input_shape
        self.in_channels = input_shape[1]
        self.state_size = input_shape[2] * input_shape[3] * num_filters[-1]
        # self.in_channels = input_shape[0]
        # self.state_size = input_shape[1] * input_shape[2] * filters[-1]
        self.num_filters = num_filters
        self.conv_params = {
            'kernel_size': kernel_size,
            'padding': 'same'
        }
        self.conv1 = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.num_filters[0],
            **self.conv_params
        )
        
        self.conv2 = nn.Conv2d(
            in_channels = self.num_filters[0],
            out_channels = self.num_filters[1],
            **self.conv_params
        )
        
        self.conv3 = nn.Conv2d(
            in_channels = self.num_filters[1],
            out_channels = self.num_filters[2],
            **self.conv_params
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x

class OceanStateDecoder(nn.Module):

    def __init__(self, state_size, route_input_size, hidden_layers, output_size, output_activation='relu'):
        super(OceanStateDecoder, self).__init__()
        self.state_size = state_size
        self.route_input_size = route_input_size
        self.in_size = state_size + route_input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.output_activation = output_activation
        self.dense1 = nn.Linear(self.in_size, self.hidden_layers[0])
        self.dense_out = nn.Linear(self.hidden_layers[0], self.output_size)
    
    def forward(self, ocean_state, route_input):
        route_batch_size = route_input.shape[0]
        ocean_state_batch = ocean_state.repeat(route_batch_size, 1)
        x = torch.cat([route_input, ocean_state_batch], dim=1)
        x = torch.arctan(self.dense1(x))
        x = self.dense_out(x)
        if self.output_activation == 'relu':
            x = F.relu(x)
        return x
    
class ForseaAutoEncoder(nn.Module):

    def __init__(self, ocean_input_shape, route_input_size, output_size, filters, kernel_size, hidden_layers, output_activation='relu'):
        super(ForseaAutoEncoder, self).__init__()
        self.input_shape = ocean_input_shape
        self.encoder_module = OceanStateEncoder(ocean_input_shape, filters, kernel_size)
        
        self.state_size = self.encoder_module.state_size
        self.route_input_size = route_input_size
        self.output_size = output_size
        self.decoder_module = OceanStateDecoder(self.state_size, route_input_size, hidden_layers, output_size, output_activation=output_activation)
    
    def forward(self, ocean_input, route_input):
        ocean_state = self.encoder_module(ocean_input)
        self.current_state = ocean_state
        output = self.decoder_module(ocean_state, route_input)
        return output
    
    def encoder(self, ocean_input):
        ocean_state = self.encoder_module(ocean_input)
        self.current_state = ocean_state
        return ocean_state
    
    def decoder(self, route_input):
        ocean_state = self.current_state
        output = self.decoder_module(ocean_state, route_input)
        return output
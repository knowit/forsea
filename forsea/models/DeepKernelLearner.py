import torch
import torch.nn as nn 
import torch.nn.functional as F
import gpytorch

class FeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(FeatureExtractor, self).__init__()
        self.input_shape = input_shape
        # self.in_channels = input_shape[1]
        self.conv_params = {
            'kernel_size': 3,
            'padding': 'same'
        }        
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32,
            **self.conv_params
        )
        # self.conv2 = nn.Conv2d(
        #     in_channels=self.input_shape[1], 
        #     out_channels=64,
        #     **self.conv_params
        # )
        # self.conv3 = nn.Conv2d(
        #     in_channels=self.input_shape[1], 
        #     out_channels=128,
        #     **self.conv_params
        # )
        self.lstm = nn.LSTM(
            input_size=input_shape[0] * input_shape[1] * 32,
            # input_size=input_shape[2]//4 * input_shape[3]//4 * 128,
            hidden_size=2
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 2)
        print(x.shape)
        lstm_out, _ = self.lstm(x)
        x = F.relu(lstm_out)
        return x


class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, input_shape, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = FeatureExtractor(input_shape)

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            print(x.shape)
            projected_x = self.feature_extractor(x)

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        print('NN', data_dim)
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

# class LargeFeatureExtractor(nn.Module):
#     def __init__(self, data_dim, hidden_layer=50, encoding_size=2):
#         super(LargeFeatureExtractor, self).__init__()
#         self.data_dim = data_dim
#         self.hidden_layer = hidden_layer
#         self.dense1 = nn.Linear(self.data_dim, hidden_layer)
#         self.dense2 = nn.Linear(hidden_layer, encoding_size)
    
#     def forward(self, x):
#         print(x.shape)
#         x = F.relu(self.dense1(x))
#         print(x.shape)
#         x = F.tanh(self.dense1(x))
#         return x

class GPRegressionModel_tmp(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel_tmp, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = LargeFeatureExtractor(train_x.shape[-1])

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
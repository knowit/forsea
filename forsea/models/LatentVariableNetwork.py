import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print("Warning: No GPU deteced. Running on CPU.")


def conv_layer(in_channels, out_channels, kernel_size=3, padding='same'):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class LatentVariableEncoder(nn.Module):

    def __init__(self, input_shape, filters, hidden_layers, latent_points=10, latent_dims=2):
        super(LatentVariableEncoder, self).__init__()

        self.input_shape = input_shape
        self.in_channels = input_shape[1]
        self.filters = filters
        
        self.dense_in = self.filters[-1] * input_shape[2] * input_shape[3]
        self.hidden_layers = hidden_layers
        self.latent_points = latent_points
        self.latent_dims = latent_dims

        self.conv1 = conv_layer(self.in_channels, self.filters[0])
        self.conv2 = conv_layer(self.filters[0], self.filters[1])
        self.conv3 = conv_layer(self.filters[1], self.filters[2])

        self.dense1 = nn.Linear(self.dense_in, self.hidden_layers[0])
        self.dense2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
        self.dense_latent_x = nn.Linear(self.hidden_layers[1], self.latent_points * self.latent_dims)
        self.dense_latent_y = nn.Linear(self.hidden_layers[1], self.latent_points)
    
    def forward(self, ocean_in, route_in=None, return_gp_model=False):
        x = F.relu(self.conv1(ocean_in))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        latent_x = F.relu(self.dense_latent_x(x)).reshape((self.latent_points, self.latent_dims))
        latent_y = F.relu(self.dense_latent_y(x)).reshape((self.latent_points,))
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        gp_model = GPModel(latent_x, latent_y, likelihood).cuda()
        if return_gp_model:
             return gp_model, likelihood
        else:
            likelihood.eval()
            gp_model.eval()
            return likelihood(gp_model(route_in)).mean
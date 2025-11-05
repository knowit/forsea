import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from math import prod



class OceanStateEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        num_filters: tuple[int, int, int, int],
        output_size: int,
        output_activation: bool,
    ):
        super(OceanStateEncoder, self).__init__()

        self.input_shape = input_shape
        self.in_channels = input_shape[1]
        self.state_shape: tuple[int, int, int] = (
            output_size,
            input_shape[2],
            input_shape[3],
        )
        self.num_filters = num_filters
        self.output_size = output_size
        self.output_activation = output_activation
        self.conv_params = {"kernel_size": (3, 3), "padding": "same"}
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_filters[0],
            **self.conv_params
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.num_filters[0],
            out_channels=self.num_filters[1],
            **self.conv_params
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[2],
            **self.conv_params
        )

        self.conv4 = nn.Conv2d(
            in_channels=self.num_filters[2],
            out_channels=self.num_filters[3],
            **self.conv_params
        )

        self.conv_out = nn.Conv2d(
            in_channels=self.num_filters[-1],
            out_channels=self.output_size,
            **self.conv_params
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_out(x)
        if self.output_activation is None:
            pass
        elif self.output_activation == "relu":
            x = F.relu(x)
        elif self.output_activation == "sigmoid":
            x = F.sigmoid(x)
        return x


class GaussianFilterDecoder(nn.Module):
    def __init__(self, state_dims: tuple[int, int, int], land_mask: Tensor) -> None:
        super(GaussianFilterDecoder, self).__init__()
        self.state_channels = state_dims[0]
        self.state_shape = state_dims[1:]
        self.land_mask = land_mask.cuda()
        self.xy_mesh = self.get_mesh(self.state_shape).cuda()
        self.sigma = nn.Parameter(torch.Tensor(1), requires_grad=True)
        # self.sigma = nn.Parameter(torch.Tensor(2), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.sigma.data.fill_(0.01)

    def get_mesh(self, shape: tuple[int, int]) -> Tensor:
        ny, nx = shape
        y = torch.linspace(0, 1, ny)
        x = torch.linspace(0, 1, nx)
        X, Y = torch.meshgrid((y, x), indexing="ij")
        return torch.cat([X[None], Y[None]], dim=0)

    def forward(self, ocean_state: Tensor, route_input: Tensor) -> Tensor:
        route_batch_size = route_input.shape[0]
        ocean_state_batch = ocean_state.expand(route_batch_size, -1, -1, -1)
        m_batch = route_input[:, :, None, None].expand(-1, -1, *self.state_shape)
        xy_batch = self.xy_mesh.expand(route_batch_size, -1, -1, -1)
        sigma2 = self.sigma**2
        z = torch.exp(-0.5 * torch.sum((xy_batch - m_batch) ** 2, dim=1) / sigma2) / (
            2 * torch.pi * sigma2
        ) * self.land_mask[None]
        # sigma2_batch = (self.sigma**2)[None,:,None,None].expand(route_batch_size, 2, *self.state_shape)
        # z = torch.exp(-0.5 * torch.sum((xy_batch - m_batch)**2 / sigma2_batch, dim=1)) / (2*torch.pi*self.sigma[0]*self.sigma[1])
        filter_batch = z[:, None].expand(-1, self.state_channels, -1, -1)
        out = torch.sum(ocean_state_batch * filter_batch, dim=(2, 3))
        return out


class ForseaAutoEncoder(nn.Module):
    def __init__(
        self,
        ocean_input_shape: tuple[int, int, int, int],
        filters: tuple[int, int, int],
        output_size: int,
        land_mask: Tensor,
        output_activation: str,
    ):
        super(ForseaAutoEncoder, self).__init__()
        self.input_shape = ocean_input_shape
        self.encoder_module = OceanStateEncoder(
            ocean_input_shape, filters, output_size, output_activation
        )

        self.state_shape = self.encoder_module.state_shape
        self.output_size = output_size
        self.decoder_module = GaussianFilterDecoder(self.state_shape, land_mask)

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

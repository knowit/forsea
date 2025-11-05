import torch.nn as nn
import torch
from torch import Tensor


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, activation="tanh"):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = "same"#kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = self.activation(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * self.activation(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          activation="relu"))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

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

class  LSTMAutoEncoder(nn.Module):
    def __init__(self, state_dims, land_mask, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False) -> None:
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first, bias, return_all_layers)
        self.decoder = GaussianFilterDecoder(state_dims, land_mask)
    
    def forward(self, ocean_input, route_input):
        lstm_output, lstm_state = self.encoder(ocean_input)
        return self.decoder(lstm_output[0][0,-1], route_input)
        # if len(route_input) != lstm_output[0].shape[1]:
        #     raise IndexError(f"Inconsistent sequence lengths: route: {len(route_input)}, ocean_state: {lstm_output[0].shape[1]}")
        # out = []
        # for t, route in enumerate(route_input):
        #     out.append(self.decoder(lstm_output[0][0,t], route))
        # return out

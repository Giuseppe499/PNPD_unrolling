from torch import nn
from math import floor

def conv_out_dim(in_dim, kernel_size, stride=1, padding=0, dilation=1):
    return floor((in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

class ConvNet(nn.Module):
  def __init__(self,
              num_outputs: int = 1,
              num_filters: list[int] = [3*2, 32],
              fc_hidden_layers: list[int] = [128],
              activation = nn.ReLU(),
              output_activation = nn.Sigmoid(),
              pool = nn.MaxPool2d(2, 2),
              kernel_sizes: int | list[int] = 5,
              in_channels: int = 2,
              input_shape: tuple[int, int] = (256, 256),
              dropout_prob = .0
              ):
    super(ConvNet, self).__init__()

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * (len(num_filters))

    self.activation = activation
    self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
    self.dropout2D = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()
    self.pool = pool
    num_filters = [in_channels] + num_filters
    self.conv_layers = [nn.Conv2d(num_filters[i], num_filters[i+1], kernel_size=kernel_sizes[i]) for i in range(len(num_filters)-1)]
    self.conv_layers = nn.ModuleList(self.conv_layers)

    final_shape = input_shape
    for layer in self.conv_layers:
      # Conv2d
      new_width = conv_out_dim(final_shape[0], layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0])
      new_height = conv_out_dim(final_shape[1], layer.kernel_size[1], layer.stride[1], layer.padding[1], layer.dilation[1])
      # Pool
      new_width = conv_out_dim(new_width, self.pool.kernel_size, self.pool.stride, self.pool.padding, self.pool.dilation)
      new_height = conv_out_dim(new_height, self.pool.kernel_size, self.pool.stride, self.pool.padding, self.pool.dilation)
      # Update final_shape
      final_shape = (new_width, new_height)

    print(f"Final shape: {final_shape}")

    fc_hidden_layers = [final_shape[0] * final_shape[1] * num_filters[-1]] + fc_hidden_layers + [num_outputs]
    self.fc_layers = [nn.Linear(fc_hidden_layers[i], fc_hidden_layers[i+1]) for i in range(len(fc_hidden_layers)-1)]
    self.fc_layers = nn.ModuleList(self.fc_layers)

    self.output_activation = output_activation

  def forward(self, x):
    for layer in self.conv_layers:
      x = self.pool(self.activation(layer(x)))
      # x = self.dropout2D(x)
    x = x.view(-1, self.fc_layers[0].in_features)
    for layer in self.fc_layers[:-1]:
      x = self.activation(layer(x))
      x = self.dropout(x)
    x = self.fc_layers[-1](x)
    x = self.output_activation(x)
    return x
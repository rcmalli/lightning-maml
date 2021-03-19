import torch
import torch.nn as nn

"""
It is taken from
https://github.com/flennerhag/warpgrad/blob/master/src/omniglot/model.py
"""

class UnSqueeze(nn.Module):
    """Create channel dim if necessary."""

    def __init__(self):
        super(UnSqueeze, self).__init__()

    def forward(self, input):
        """Creates channel dimension on a 3-d tensor.

        Null-op if input is a 4-d tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.dim() == 4:
            return input
        return input.unsqueeze(1)


class Squeeze(nn.Module):
    """Undo excess dimensions"""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(Squeeze, self).__init__()

    def forward(self, input):
        """Squeeze singular dimensions of an input tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.size(0) != 0:
            return input.squeeze()
        input = input.squeeze()
        return input.view(1, *input.size())


class Linear(nn.Module):
    """Wrapper around torch.nn.Linear to deal with single/multi-headed mode.

    Arguments:
        multi_head (bool): multi-headed mode.
        num_features_in (int): number of features in input.
        num_features_out (int): number of features in output.
        **kwargs: optional arguments to pass to torch.nn.Linear.
    """

    def __init__(self, multi_head, num_features_in,
                 num_features_out, **kwargs):
        super(Linear, self).__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out

        self.multi_head = multi_head

        def _linear_factory():
            return nn.Linear(num_features_in, num_features_out, **kwargs)

        if self.multi_head:
            self.linear = nn.ModuleList([_linear_factory()] * kwargs[
                'num_classes'])
        else:
            self.linear = _linear_factory()

    def forward(self, x, idx=None):
        if self.multi_head:
            assert idx is not None, "Pass head idx in multi-headed mode."
            return self.linear[idx](x)
        return self.linear(x)

    def reset_parameters(self):
        """Reset parameters if in multi-headed mode."""
        if self.multi_head:
            for lin in self.linear:
                lin.reset_parameters()
        else:
            self.linear.reset_parameters()

class OmniConv(nn.Module):
    """OmniConv classifier.

    Arguments:
        num_classes (int): number of classes to predict in each alphabet
        num_layers (int): number of convolutional layers (default=4).
        kernel_size (int): kernel size in each convolution (default=3).
        num_filters (int): number of output filters in each convolution
            (default=64)
        imsize (tuple): tuple of image height and width dimension.
        padding (bool, int, tuple): padding argument to convolution layers
            (default=True).
        batch_norm (bool): use batch normalization in each convolution layer
            (default=True).
        multi_head (bool): multi-headed training (default=False).
    """

    def __init__(self, num_classes, num_layers=4, kernel_size=3,
                 num_filters=64, imsize=(28, 28), padding=True,
                 batch_norm=True, multi_head=False):
        super(OmniConv, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.imsize = imsize
        self.batch_norm = batch_norm
        self.multi_head = multi_head

        def conv_block(nin):
            block = [nn.Conv2d(nin, num_filters, kernel_size, padding=padding),
                     nn.MaxPool2d(2)]
            if batch_norm:
                block.append(nn.BatchNorm2d(num_filters))
            block.append(nn.ReLU())
            return block

        layers = [UnSqueeze()]
        for i in range(num_layers):
            layers.extend(conv_block(1 if i == 0 else num_filters))

        layers.append(Squeeze())

        self.conv = nn.Sequential(*layers)
        self.head = Linear(self.multi_head, num_filters, num_classes)

    def forward(self, input, idx=None):
        input = self.conv(input)
        return self.head(input, idx)

    def init_adaptation(self):
        """Reset stats for new task"""
        # Reset if multi-head, otherwise null-op
        self.head.reset_parameters()

        # Reset BN running stats
        for m in self.modules():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()
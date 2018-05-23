import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function, Variable


class ConstantPadNd(Function):

    @staticmethod
    def forward(ctx, input, pad, value=0):
        ctx.pad = pad
        ctx.value = value
        ctx.input_size = input.size()
        ctx.l_inp = len(input.size())
        ctx.pad_tup = tuple([(a, b) for a, b in zip(pad[:-1:2], pad[1::2])][::-1])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i]) for i, d in enumerate(input.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # crop input if necessary
        output = input.new(input.size()[:(ctx.l_diff)] + new_dim).fill_(ctx.value)
        c_input = input

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] < 0:
                c_input = c_input.narrow(i, -p[0], c_input.size(i) + p[0])
            if p[1] < 0:
                c_input = c_input.narrow(i, 0, c_input.size(i) + p[1])

        # crop output if necessary
        c_output = output
        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                c_output = c_output.narrow(i, p[0], c_output.size(i) - p[0])
            if p[1] > 0:
                c_output = c_output.narrow(i, 0, c_output.size(i) - p[1])
        c_output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x,) for x in ctx.input_size]

        def narrow_slice(dim, start, length):
            grad_input_slices[dim] = (slice(grad_input_slices[dim].start + start,
                                            grad_input_slices[dim].start + start + length))

        def slice_length(dim):
            return grad_input_slices[dim].stop - grad_input_slices[dim].start

        #  crop grad_input if necessary
        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] < 0:
                narrow_slice(i, -p[0], slice_length(i) + p[0])
            if p[1] < 0:
                narrow_slice(i, 0, slice_length(i) + p[1])

        # crop grad_output if necessary
        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0], cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0, cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None


def pad(input, pad, mode='constant', value=0):
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= input.dim(), 'Padding length too large'
    if mode == 'constant':
        return ConstantPadNd.apply(input, pad, value)


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1):
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.
    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = F.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = F.avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div


class Inception(nn.Module):
    def __init__(self, in_channel, br_1x1, br_3x3_reduce, br_3x3,
                 br_5x5_reduce, br_5x5, pool_proj):
        super(Inception, self).__init__()
        self.add_module('1x1', nn.Conv2d(in_channel, br_1x1, kernel_size=1))
        self.add_module('relu_1x1', nn.ReLU())

        self.add_module('3x3_reduce', nn.Conv2d(in_channel, br_3x3_reduce, kernel_size=1))
        self.add_module('relu_3x3_reduce', nn.ReLU())
        self.add_module('3x3', nn.Conv2d(br_3x3_reduce, br_3x3, kernel_size=3, padding=1))
        self.add_module('relu_3x3', nn.ReLU())

        self.add_module('5x5_reduce', nn.Conv2d(in_channel, br_5x5_reduce, kernel_size=1))
        self.add_module('relu_5x5_reduce', nn.ReLU())
        self.add_module('5x5', nn.Conv2d(br_5x5_reduce, br_5x5, kernel_size=5, padding=2))
        self.add_module('relu_5x5', nn.ReLU())

        self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.add_module('pool_proj', nn.Conv2d(in_channel, pool_proj, kernel_size=1))
        self.add_module('relu_pool_proj', nn.ReLU())

    def forward(self, x):
        x1 = getattr(self, '1x1')(x)
        x1 = getattr(self, 'relu_1x1')(x1)

        x2 = getattr(self, '3x3_reduce')(x)
        x2 = getattr(self, 'relu_3x3_reduce')(x2)
        x2 = getattr(self, '3x3')(x2)
        x2 = getattr(self, 'relu_3x3')(x2)

        x3 = getattr(self, '5x5_reduce')(x)
        x3 = getattr(self, 'relu_5x5_reduce')(x3)
        x3 = getattr(self, '5x5')(x3)
        x3 = getattr(self, 'relu_5x5')(x3)

        x4 = getattr(self, 'pool')(x)
        x4 = getattr(self, 'pool_proj')(x4)
        x4 = getattr(self, 'relu_pool_proj')(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.add_module('conv1.7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        self.add_module('conv1.relu_7x7', nn.ReLU())

        self.add_module('pool1', nn.MaxPool2d(3, stride=2, ceil_mode=True))

        self.add_module('conv2.3x3_reduce', nn.Conv2d(64, 64, kernel_size=1))
        self.add_module('conv2.relu_3x3_reduce', nn.ReLU())
        self.add_module('conv2.3x3', nn.Conv2d(64, 192, kernel_size=3, padding=1))
        self.add_module('conv2.relu_3x3', nn.ReLU())

        self.add_module('pool2', nn.MaxPool2d(3, stride=2, ceil_mode=True))

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.add_module('pool3', nn.MaxPool2d(3, stride=2, ceil_mode=True))

        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.add_module('pool4', nn.MaxPool2d(3, stride=2, ceil_mode=True))

        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.add_module('pool5', nn.AvgPool2d(7, stride=1))

        self.add_module('drop', nn.Dropout2d(p=0.4))

        self.add_module('loss3.classifier', nn.Linear(1024, 1000))

    def forward(self, x):
        x = getattr(self, 'conv1.7x7_s2')(x)
        x = getattr(self, 'conv1.relu_7x7')(x)
        x = self.pool1(x)
        x = local_response_norm(x, 5)

        x = getattr(self, 'conv2.3x3_reduce')(x)
        x = getattr(self, 'conv2.relu_3x3_reduce')(x)
        x = getattr(self, 'conv2.3x3')(x)
        x = getattr(self, 'conv2.relu_3x3')(x)
        x = local_response_norm(x, 5)
        x = self.pool2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)

        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.pool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.pool5(x)
        x = self.drop(x)

        x = x.view(-1, 1024)
        x = getattr(self, 'loss3.classifier')(x)

        return x


def get_googlenet(pretrain=False, pth_path='./weights/googlenet.pth'):
    model = GoogleNet()
    if pretrain:
        model.load_state_dict(torch.load(pth_path))

    return model

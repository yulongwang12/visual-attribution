from torch.autograd import Function, Variable
from torch._thnn import type2backend
from torch.nn.modules.utils import _pair


class EBLinear(Function):

    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp, weight, bias)
        output = inp.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_variables

        wplus = weight.clone().clamp(min=0)

        output = inp.matmul(wplus.t())
        normalized_grad_output = grad_output / (output + 1e-10)
        normalized_grad_output = normalized_grad_output * (output > 0).float()

        grad_inp = normalized_grad_output.matmul(wplus)
        grad_inp = grad_inp * inp

        return grad_inp, None, None


def _output_size(inp, weight, pad, dilation, stride):
    pad = pad[0]
    dilation = dilation[0]
    stride = stride[0]

    channels = weight.size(0)
    output_size = (inp.size(0), channels)
    for d in range(inp.dim() - 2):
        in_size = inp.size(d + 2)
        kernel = dilation * (weight.size(d + 2) - 1) + 1
        output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError("convolution inp is too small (output would be {})".format(
            'x'.join(map(str, output_size))))
    return output_size


class EBConv2d(Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(inp, weight, bias)

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        kH, kW = weight.size(2), weight.size(3)

        output_size = _output_size(inp, weight, padding, dilation, stride)
        output = inp.new(*output_size)
        columns = inp.new(*output_size)
        ones = inp.new(*output_size)

        backend = type2backend[inp.type()]
        f = getattr(backend, 'SpatialConvolutionMM_updateOutput')
        f(backend.library_state, inp, output, weight, bias, columns, ones,
          kH, kW, ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        output_size = _output_size(inp, weight, padding, dilation, stride)
        kH, kW = weight.size(2), weight.size(3)

        wplus = weight.clone().clamp(min=0)
        new_output = inp.new(*output_size)
        columns = inp.new(*output_size)
        ones = inp.new(*output_size)

        backend = type2backend[inp.type()]
        f = getattr(backend, 'SpatialConvolutionMM_updateOutput')
        f(backend.library_state, inp, new_output, wplus, None, columns, ones,
          kH, kW, ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1])

        normalized_grad_output = grad_output.data / (new_output + 1e-10)
        normalized_grad_output = normalized_grad_output * (new_output > 0).float()

        grad_inp = inp.new()
        grad_inp.resize_as_(inp)

        g = getattr(backend, 'SpatialConvolutionMM_updateGradInput')
        g(backend.library_state, inp, normalized_grad_output, grad_inp, wplus, columns, ones,
          kH, kW, ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1])

        grad_inp = grad_inp * inp

        return Variable(grad_inp), None, None, None, None, None, None


class EBAvgPool2d(Function):

    @staticmethod
    def forward(ctx, inp, kernel_size, stride=None, padding=0,
                ceil_mode=False, count_include_pad=True):
        ctx.kernel_size = (kernel_size, kernel_size)
        stride = stride if stride is not None else kernel_size
        ctx.stride = (stride, stride)
        ctx.padding = (padding, padding)
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        backend = type2backend[type(inp)]
        output = inp.new()

        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            inp, output,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)

        ctx.save_for_backward(inp, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        backend = type2backend[type(grad_output.data)]
        inp, output = ctx.saved_tensors

        normalized_grad_output = grad_output.data / (output + 1e-10)
        normalized_grad_output = normalized_grad_output * (output > 0).float()

        grad_inp = inp.new()

        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            inp, normalized_grad_output, grad_inp,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)

        grad_inp = grad_inp * inp

        return Variable(grad_inp), None, None, None, None, None

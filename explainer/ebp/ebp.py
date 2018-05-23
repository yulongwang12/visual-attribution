import types
import torch
from explainer.ebp.functions import EBConv2d, EBLinear, EBAvgPool2d


def get_layer(model, key_list):
    a = model
    for key in key_list:
        a = a._modules[key]
    return a

class ExcitationBackpropExplainer(object):
    def __init__(self, model, output_layer_keys=None):
        self.output_layer = get_layer(model, output_layer_keys)
        self.model = model
        self._override_backward()
        self._register_hooks()

    def _override_backward(self):
        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)
        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        def replace(m):
            name = m.__class__.__name__
            if name == 'Linear':
                m.forward = types.MethodType(new_linear, m)
            elif name == 'Conv2d':
                m.forward = types.MethodType(new_conv2d, m)
            elif name == 'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)

        self.model.apply(replace)

    def _register_hooks(self):
        self.intermediate_vars = []
        def forward_hook(m, i, o):
            self.intermediate_vars.append(o)

        self.output_layer.register_forward_hook(forward_hook)

    def explain(self, inp, ind=None):
        self.intermediate_vars = []

        output = self.model(inp)
        output_var = self.intermediate_vars[0]

        if ind is None:
            ind = output.data.max(1)[1]
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)

        attmap_var = torch.autograd.grad(output, output_var, grad_out, retain_graph=True)
        attmap = attmap_var[0].data.clone()
        attmap = torch.clamp(attmap.sum(1).unsqueeze(1), min=0.0)

        return attmap


class ContrastiveExcitationBackpropExplainer(object):
    def __init__(self, model, intermediate_layer_keys=None, output_layer_keys=None, final_linear_keys=None):
        self.intermediate_layer = get_layer(model, intermediate_layer_keys)
        self.output_layer = get_layer(model, output_layer_keys)
        self.final_linear = get_layer(model, final_linear_keys)
        self.model = model
        self._override_backward()
        self._register_hooks()

    def _override_backward(self):
        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)
        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        def replace(m):
            name = m.__class__.__name__
            if name == 'Linear':
                m.forward = types.MethodType(new_linear, m)
            elif name == 'Conv2d':
                m.forward = types.MethodType(new_conv2d, m)
            elif name == 'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)

        self.model.apply(replace)

    def _register_hooks(self):
        self.intermediate_vars = []
        def forward_hook(m, i, o):
            self.intermediate_vars.append(o)

        self.intermediate_layer.register_forward_hook(forward_hook)
        self.output_layer.register_forward_hook(forward_hook)

    def explain(self, inp, ind=None):
        self.intermediate_vars = []

        output = self.model(inp)
        output_var, intermediate_var = self.intermediate_vars

        if ind is None:
            ind = output.data.max(1)[1]
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)

        self.final_linear.weight.data *= -1.0
        neg_map_var = torch.autograd.grad(output, intermediate_var, grad_out, retain_graph=True)
        neg_map = neg_map_var[0].data.clone()

        self.final_linear.weight.data *= -1.0
        pos_map_var = torch.autograd.grad(output, intermediate_var, grad_out, retain_graph=True)
        pos_map = pos_map_var[0].data.clone()

        diff = pos_map - neg_map
        attmap_var = torch.autograd.grad(intermediate_var, output_var, diff, retain_graph=True)

        attmap = attmap_var[0].data.clone()
        attmap = torch.clamp(attmap.sum(1).unsqueeze(1), min=0.0)

        return attmap

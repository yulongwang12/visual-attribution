import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch import optim
import torch
import types
from explainer.backprop import SaliencyExplainer


def first_forward(self, x):
    out = F.relu(x)
    self.control_gate = Parameter(out.data.clone())
    self.control_gate.data.fill_(1.0)
    return out


def mask_forward(self, x):
    out = F.relu(x)
    out = self.control_gate * out
    return out


def replace_first(m):
    name = m.__class__.__name__
    if name.find('ReLU') != -1:
        m.forward = types.MethodType(first_forward, m)


def replace_mask(m):
    name = m.__class__.__name__
    if name.find('ReLU') != -1:
        m.forward = types.MethodType(mask_forward, m)


class FeedbackExplainer(SaliencyExplainer):
    def __init__(self, model, input_size, class_num=1000, lr=0.1, lambd=0.01, max_iters=30):
        super(FeedbackExplainer, self).__init__(model)
        self.model = model
        self.lr = lr
        self.lambd = lambd
        self.max_iters = max_iters
        self.input_size = input_size
        self.class_num = class_num
        self.control_gates = []

        self._init_control_gates()
        self.model.apply(replace_mask)

    def _init_control_gates(self):
        self.model.apply(replace_first)
        input_placeholder = Variable(torch.randn(*self.input_size).cuda())
        _ = self.model(input_placeholder)
        for m in self.model.modules():
            if m.__class__.__name__ == 'ReLU':
                self.control_gates.append(m.control_gate)

    def _reset_control_gates(self):
        for i in range(len(self.control_gates)):
            self.control_gates[i].data.fill_(1.0)
            if self.control_gates[i].grad is not None:
                self.control_gates[i].grad.data.fill_(0.0)

    def explain(self, inp, ind=None):
        self._reset_control_gates()
        optimizer = optim.SGD(self.control_gates, lr=self.lr, momentum=0.9, weight_decay=0.0)

        mask = torch.zeros(self.input_size[0], self.class_num).cuda()
        mask.scatter_(1, ind.unsqueeze(1), 1)
        mask_var = Variable(mask)

        for j in range(self.max_iters):
            output = self.model(inp)
            loss = -(output * mask_var).sum()

            for v in self.control_gates:
                loss += self.lambd * v.abs().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for v in self.control_gates:
                v.data.clamp_(0, 1)

        saliency = super(FeedbackExplainer, self).explain(inp, ind)
        return saliency

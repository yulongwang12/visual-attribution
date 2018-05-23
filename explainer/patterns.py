import numpy as np
import torch
import torch.nn.functional as F


def load_patterns(filename):
    f = np.load(filename)
    ret = {}
    for prefix in ["A", "r", "mu"]:
        l = sum([x.startswith(prefix) for x in f.keys()])
        ret.update({prefix: [f["%s_%i" % (prefix, i)] for i in range(l)]})
    return ret


def load_params(filename):
    f = np.load(filename)
    weights = []
    for i in range(32):
        if i in [26, 28, 30]:
            weights.append(f['arr_%d' % i].T)
        else:
            weights.append(f['arr_%d' % i])

    return weights


class PatternNetExplainer(object):
    def __init__(self, model, params_file=None, pattern_file=None):
        self.model = model
        self.weights = list(self.model.parameters())
        self.np_weights = load_params(params_file)
        self.np_patterns = load_patterns(pattern_file)['A']
        self._to_cuda()

    def _to_cuda(self):
        for i in range(len(self.np_weights)):
            self.np_weights[i] = torch.from_numpy(self.np_weights[i]).float().cuda()

        for i in range(len(self.np_patterns)):
            self.np_patterns[i] = torch.from_numpy(self.np_patterns[i]).float().cuda()

    def _fill_in_params(self):
        for i in range(32):
            self.weights[i].data.copy_(self.np_weights[i])

    def _fill_in_patterns(self):
        for i in range(0, 26, 2):
            self.weights[i].data.copy_(self.np_patterns[int(i / 2)])
        for i in range(26, 32, 2):
            self.weights[i].data.copy_(self.np_patterns[int(i / 2)].t())

    def explain(self, inp, ind=None):
        self._fill_in_params()

        output = self.model(inp)
        prob = F.softmax(output)
        if ind is None:
            ind = output.data.max(1)[1]

        probvalue = prob.data.gather(1, ind.unsqueeze(0).t())

        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), probvalue)

        self._fill_in_patterns()
        output.backward(grad_out)

        return inp.grad.data


class PatternLRPExplainer(PatternNetExplainer):
    def __init__(self, model, params_file=None, pattern_file=None):
        super(PatternLRPExplainer, self).__init__(model, params_file, pattern_file)

    def _fill_in_patterns(self):
        for i in range(0, 26, 2):
            self.weights[i].data.copy_(
                self.weights[i].data * self.np_patterns[int(i / 2)]
            )
        for i in range(26, 32, 2):
            self.weights[i].data.copy_(
                self.weights[i].data * self.np_patterns[int(i / 2)].t()
            )

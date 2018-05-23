import torch.nn as nn
from torch.nn import functional as F
import torch
import os


class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def SimpleCNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):
        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):
            _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=kernel_size//2, bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return nn.Sequential(*_modules)



def SimpleUpsamplerSubpixel(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        SimpleCNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)


class UNetUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, passthrough_channels, follow_up_residual_blocks=1, upsampler_block=SimpleUpsamplerSubpixel,
                 upsampler_kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
        super(UNetUpsampler, self).__init__()
        assert follow_up_residual_blocks >= 1, 'You must follow up with residuals when using unet!'
        assert passthrough_channels >= 1, 'You must use passthrough with unet'
        self.upsampler = upsampler_block(in_channels=in_channels,
                                         out_channels=out_channels, kernel_size=upsampler_kernel_size, activation_fn=activation_fn)
        self.follow_up = BottleneckBlock(out_channels+passthrough_channels, out_channels, layers=follow_up_residual_blocks, activation_fn=activation_fn)

    def forward(self, inp, passthrough):
        upsampled = self.upsampler(inp)
        upsampled = torch.cat((upsampled, passthrough), 1)
        return self.follow_up(upsampled)



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck_ratio=4, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
        super(Bottleneck, self).__init__()
        bottleneck_channels = out_channels//bottleneck_ratio
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn()

        if stride != 1 or in_channels != out_channels :
            self.residual_transformer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.residual_transformer = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual_transformer is not None:
            residual = self.residual_transformer(residual)
        out += residual

        out = self.activation_fn(out)
        return out

def BottleneckBlock(in_channels, out_channels, stride=1, layers=1, activation_fn=lambda: torch.nn.ReLU(inplace=False)):
    assert layers > 0 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(Bottleneck(current_channels, out_channels, stride=stride if layer==0 else 1, activation_fn=activation_fn))
        current_channels = out_channels
    return nn.Sequential(*_modules) if len(_modules)>1 else _modules[0]


class SaliencyModel(nn.Module):
    def __init__(self, encoder, encoder_scales, encoder_base, upsampler_scales, upsampler_base, fix_encoder=True,
                 use_simple_activation=False, allow_selector=False, num_classes=1000):
        super(SaliencyModel, self).__init__()
        assert upsampler_scales <= encoder_scales

        self.encoder = encoder  # decoder must return at least scale0 to scaleN where N is num_scales
        self.upsampler_scales = upsampler_scales
        self.encoder_scales = encoder_scales
        self.fix_encoder = fix_encoder
        self.use_simple_activation = use_simple_activation

        # now build the decoder for the specified number of scales
        # start with the top scale
        down = self.encoder_scales
        modulator_sizes = []
        for up in reversed(range(self.upsampler_scales)):
            upsampler_chans = upsampler_base * 2**(up+1)
            encoder_chans = encoder_base * 2**down
            inc = upsampler_chans if down!=encoder_scales else encoder_chans
            modulator_sizes.append(inc)
            self.add_module('up%d'%up,
                            UNetUpsampler(
                                in_channels=inc,
                                passthrough_channels=encoder_chans//2,
                                out_channels=upsampler_chans//2,
                                follow_up_residual_blocks=1,
                                activation_fn=lambda: nn.ReLU(),
                            ))
            down -= 1

        self.to_saliency_chans = nn.Conv2d(upsampler_base, 2, 1)

        self.allow_selector = allow_selector

        if self.allow_selector:
            s = encoder_base*2**encoder_scales
            self.selector_module = nn.Embedding(num_classes, s)
            self.selector_module.weight.data.normal_(0, 1./s**0.5)


    def minimialistic_restore(self, save_dir):
        assert self.fix_encoder, 'You should not use this function if you are not using a pre-trained encoder like resnet'

        p = os.path.join(save_dir)
        if not os.path.exists(p):
            print('Could not find any checkpoint at %s, skipping restore' % p)
            return
        for name, data in torch.load(p, map_location=lambda storage, loc: storage).items():
            self._modules[name].load_state_dict(data)


    def forward(self, _images, _selectors=None, pt_store=None, model_confidence=0.):
        # forward pass through the encoder
        out = self.encoder(_images)
        if self.fix_encoder:
            out = [e.detach() for e in out]

        down = self.encoder_scales
        main_flow = out[down]

        if self.allow_selector:
            assert _selectors is not None
            em = torch.squeeze(self.selector_module(_selectors.view(-1, 1)), 1)
            act = torch.sum(main_flow * em.view(-1, 2048, 1, 1), 1, keepdim=True)
            th = torch.sigmoid(act - model_confidence)
            main_flow = main_flow*th

            ex = torch.mean(torch.mean(act, 3), 2)
            exists_logits = torch.cat((-ex / 2., ex / 2.), 1)
        else:
            exists_logits = None

        for up in reversed(range(self.upsampler_scales)):
            assert down > 0
            main_flow = self._modules['up%d'%up](main_flow, out[down-1])
            down -= 1
        # now get the final saliency map (the reslution of the map = resolution_of_the_image / (2**(encoder_scales-upsampler_scales)))
        saliency_chans = self.to_saliency_chans(main_flow)

        if self.use_simple_activation:
            return torch.unsqueeze(torch.sigmoid(saliency_chans[:,0,:,:]/2), dim=1), exists_logits, out[-1]


        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:,1,:,:])
        return torch.unsqueeze(a/(a+b), dim=1), exists_logits, out[-1]
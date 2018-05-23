from PIL import ImageFilter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


class Blur(object):
    def __init__(self, radius=10):
        self.radius = radius

    def __call__(self, img):
        blurred_img = img.filter(ImageFilter.GaussianBlur(self.radius))
        return blurred_img

def tv_norm(input, tv_beta):
    row_grad = torch.mean(torch.abs((input[:, :, :-1, :] - input[:, :, 1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((input[:, :, :, :-1] - input[:, :, :, 1:])).pow(tv_beta))
    return row_grad + col_grad

def get_transforms(if_inception=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    blur_transf = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        Blur(),
        transforms.ToTensor(),
        normalize
    ])

    if if_inception:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        transf = transforms.Compose([
            transforms.Scale(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ])

        blur_transf = transforms.Compose([
            transforms.Scale(299),
            transforms.CenterCrop(299),
            Blur(),
            transforms.ToTensor(),
            normalize
        ])

    return transf, blur_transf

class PerturbExplainer(object):
    def __init__(self, model, num_iters=500, lr=0.1,
                 l1_lambda=0.01, tv_lambda=0.2, tv_beta=3,
                 mask_scale=8, if_upsample=True):
        # this is for vgg19
        # for vgg16, l1_lambda = 0.005, num_iters=300, tv_lambda = 0.1
        self.model = model
        self.num_iters = num_iters
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.tv_lambda = tv_lambda
        self.tv_beta = tv_beta
        self.mask_scale = mask_scale
        self.if_upsample = if_upsample

    def explain(self, inp, null_inp, ind=None):
        if ind is None:
            output = self.model(inp)
            ind = output.max(1)[1][0]

        mask_init = torch.ones(
            1, 1,
            int(inp.data.size(2) / self.mask_scale),
            int(inp.data.size(3) / self.mask_scale)
        )

        mask_var = Variable(mask_init.cuda(), requires_grad=True)

        optimizer = optim.Adam([mask_var], lr=self.lr)

        for t in range(self.num_iters):
            real_mask = F.upsample(mask_var, scale_factor=self.mask_scale, mode='bilinear')

            x = inp * real_mask + null_inp * (1 - real_mask)

            output = self.model(x)
            prob = F.softmax(output)

            loss = self.l1_lambda * torch.mean(torch.abs(1 - mask_var)) + \
                   self.tv_lambda * tv_norm(mask_var, self.tv_beta) + \
                   prob[0, ind]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask_var.data.clamp_(0, 1)

        if self.if_upsample:
            mask_var = F.upsample(mask_var, scale_factor=self.mask_scale, mode='bilinear')

        return 1 - mask_var.data

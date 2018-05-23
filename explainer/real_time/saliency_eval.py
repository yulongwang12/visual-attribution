from explainer.real_time.pytorch_fixes import SaliencyModel
from explainer.real_time.resnet_encoder import resnet50encoder
import torch.nn.functional as F
from torch.autograd import Variable


def get_pretrained_saliency_fn(model_dir, cuda=True, return_classification_logits=False):
    ''' returns a saliency function that takes images and class selectors as inputs. If cuda=True then places the model on a GPU.
    You can also specify model_confidence - smaller values (~0) will show any object in the image that even slightly resembles the specified class
    while higher values (~5) will show only the most salient parts.
    Params of the saliency function:
    images - input images of shape (C, H, W) or (N, C, H, W) if in batch. Can be either a numpy array, a Tensor or a Variable
    selectors - class ids to be masked. Can be either an int or an array with N integers. Again can be either a numpy array, a Tensor or a Variable
    model_confidence - a float, 6 by default, you may want to decrease this value to obtain more complete saliency maps.

    returns a Variable of shape (N, 1, H, W) with one saliency maps for each input image.
    '''
    saliency = SaliencyModel(resnet50encoder(pretrained=True), 5, 64, 3, 64, fix_encoder=True, use_simple_activation=False, allow_selector=True)
    saliency.minimialistic_restore(model_dir)
    saliency.train(False)
    if cuda:
        saliency = saliency.cuda()
    def fn(images, selectors, model_confidence=6):
        selectors = Variable(selectors)
        masks, _, cls_logits = saliency(images * 2, selectors, model_confidence=model_confidence)
        sal_map = F.upsample(masks, (images.size(2), images.size(3)), mode='bilinear')
        if not return_classification_logits:
            return sal_map
        return sal_map, cls_logits
    return fn


class RealTimeSaliencyExplainer(object):
    def __init__(self, model_dir, cuda=True, return_classification_logits=False):
        self.saliency_fn = get_pretrained_saliency_fn(model_dir, cuda, return_classification_logits)

    def explain(self, inp, ind):
        mask_var = self.saliency_fn(inp, ind)
        return mask_var.data

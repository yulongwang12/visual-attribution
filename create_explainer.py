from explainer import backprop as bp
from explainer import deeplift as df
from explainer import gradcam as gc
from explainer import patterns as pt
from explainer import ebp
from explainer import real_time as rt


def get_explainer(model, name):
    methods = {
        'vanilla_grad': bp.VanillaGradExplainer,
        'grad_x_input': bp.GradxInputExplainer,
        'saliency': bp.SaliencyExplainer,
        'integrate_grad': bp.IntegrateGradExplainer,
        'deconv': bp.DeconvExplainer,
        'guided_backprop': bp.GuidedBackpropExplainer,
        'deeplift_rescale': df.DeepLIFTRescaleExplainer,
        'gradcam': gc.GradCAMExplainer,
        'pattern_net': pt.PatternNetExplainer,
        'pattern_lrp': pt.PatternLRPExplainer,
        'excitation_backprop': ebp.ExcitationBackpropExplainer,
        'contrastive_excitation_backprop': ebp.ContrastiveExcitationBackpropExplainer,
        'real_time_saliency': rt.RealTimeSaliencyExplainer
    }

    if name == 'smooth_grad':
        base_explainer = methods['vanilla_grad'](model)
        explainer = bp.SmoothGradExplainer(base_explainer)

    elif name.find('pattern') != -1:
        explainer = methods[name](
            model,
            params_file='./weights/imagenet_224_vgg_16.npz',
            pattern_file='./weights/imagenet_224_vgg_16.patterns.A_only.npz'
        )

    elif name == 'gradcam':
        if model.__class__.__name__ == 'VGG':
            explainer = methods[name](
                model, target_layer_name_keys=['features', '30'] # pool5
            )
        elif model.__class__.__name__ == 'GoogleNet':
            explainer = methods[name](
                model, target_layer_name_keys=['pool5'], use_inp=True,
            )
        elif model.__class__.__name__ == 'ResNet':
            explainer = methods[name](
                model, target_layer_name_keys=['avgpool'], use_inp=True,
            )

    elif name == 'excitation_backprop':
        if model.__class__.__name__ == 'VGG': # vgg16
            explainer = methods[name](
                model,
                output_layer_keys=['features', '23']  # pool4
            )
        elif model.__class__.__name__ == 'ResNet': # resnet50
            explainer = methods[name](
                model,
                output_layer_keys=['layer4', '1', 'conv1']  # res4a
            )
        elif model.__class__.__name__ == 'GoogleNet': # googlent
            explainer = methods[name](
                model,
                output_layer_keys=['pool2']
            )

    elif name == 'contrastive_excitation_backprop':
        if model.__class__.__name__ == 'VGG': # vgg16
            explainer = methods[name](
                model,
                intermediate_layer_keys=['features', '30'], # pool5
                output_layer_keys=['features', '23'],  # pool4
                final_linear_keys=['classifier', '6']  # fc8
            )
        elif model.__class__.__name__ == 'ResNet': # resnet50
            explainer = methods[name](
                model,
                intermediate_layer_keys=['avgpool'],
                output_layer_keys=['layer4', '1', 'conv1'],  # res4a
                final_linear_keys=['fc']
            )
        elif model.__class__.__name__ == 'GoogleNet':
            explainer = methods[name](
                model,
                intermediate_layer_keys=['pool5'],
                output_layer_keys=['pool2'],
                final_linear_keys=['loss3.classifier']
            )
    elif name == 'real_time_saliency':
        explainer = methods[name]('./weights/model-1.ckpt')

    else:
        explainer = methods[name](model)

    return explainer


def get_heatmap(saliency):
    saliency = saliency.squeeze()

    if len(saliency.size()) == 2:
        return saliency.abs().cpu().numpy()
    else:
        return saliency.abs().max(0)[0].cpu().numpy()

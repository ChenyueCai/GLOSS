# Code taken from: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Solution.ipynb
# MIT License
#
# Copyright (c) 2018 Udacity
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from PIL import Image
from io import BytesIO
import numpy as np

from accelerate import Accelerator
import torch
import torchvision
from torchvision import transforms, models

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation

# style_weights = {'conv1_1': 1.,
#                  'conv2_1': 0.75,
#                  'conv3_1': 0.2,
#                  'conv4_1': 0.2,
#                  'conv5_1': 0.2}

style_weights = {'relu1_2': 1.0 / 2.6,
                 'relu2_2': 1.0 / 4.8,
                 'relu3_3': 1.0 / 3.7,
                 'relu4_3': 1.0 / 5.6,
                 'relu5_3': 10.0 / 1.5}


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        # layers = {'0': 'conv1_1',
        #           '5': 'conv2_1',
        #           '10': 'conv3_1',
        #           '19': 'conv4_1',
        #           '21': 'conv4_2',  ## content representation
        #           '28': 'conv5_1'}
        layers = {'3': 'relu1_2',
                  '8': 'relu2_2',
                  '15': 'relu3_3',
                  '22': 'relu4_3',
                  '29': 'relu5_3'}

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.reshape(b * d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


def style_loss(style, target, vgg):
    style_features = get_features(style, vgg)
    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # get the features from your target image
    target_features = get_features(target, vgg)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    return style_loss


def style_loss_precomputed(style_grams, target_grams, gram_norms):
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_gram = target_grams[layer]
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / gram_norms[layer]

    return style_loss


class StyleLoss:
    __singleton__ = None

    @staticmethod
    def Singleton():
        if StyleLoss.__singleton__ is None:
            StyleLoss.__singleton__ = StyleLoss()
        return StyleLoss.__singleton__

    def __init__(self):
        self.accelerator = Accelerator(
            mixed_precision="fp16")
        self.net_vgg = torchvision.models.vgg16(pretrained=True).features
        for param in self.net_vgg.parameters():
            param.requires_grad_(False)
        self.net_vgg = self.accelerator.prepare(self.net_vgg)
        self.vgg_renorm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def compute_features(self, input):
        # TODO: equip to do faster computation by precomputing features
        pass

    @staticmethod
    def compute(style, target):
        vgg = StyleLoss().Singleton().net_vgg
        renorm = StyleLoss().Singleton().vgg_renorm
        return style_loss(renorm(style), renorm(target), vgg)

#!/usr/bin/python
# -*- coding: utf-8 -*-
import torchvision
import argparse
import os
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Wide_ResNet101_2_Weights
from tqdm import tqdm
from common3d import (get_pdn_small, get_pdn_medium, get_pdn_xsmall,
                      ImageFolderWithoutTarget, InfiniteDataloader)

from src.i3res import I3ResNet
import unfoldNd


def get_argparse():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-o', '--output_folder',
                        default='output/pretraining/small')
    return parser.parse_args()


# variables
model_size = 'small'
imagenet_train_path = '/trinity/home/mzijta/repositories/phd_project/anomalib/datasets/imagenette/imagenette2/train'
seed = 42
on_gpu = torch.cuda.is_available()
device = 'cuda' if on_gpu else 'cpu'
frame_nb = 16
image_size = 64
half_image_size = int(image_size / 2)
eight_image_size = int(image_size / 8)
# constants
out_channels = 384
grayscale_transform = transforms.RandomGrayscale(0.1)  # apply same to both
extractor_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pdn_transform = transforms.Compose([
    transforms.Resize((half_image_size, half_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_transform(image):
    image = grayscale_transform(image)
    return extractor_transform(image), pdn_transform(image)


def expand_2d_to_3d_image(image, image_size):
    stack_direction = random.randint(0, 2)
    if stack_direction == 0:
        image = image.unsqueeze(2).repeat(1, 1, image_size, 1, 1)
    elif stack_direction == 1:
        image = image.unsqueeze(3).repeat(1, 1, 1, image_size, 1)
    elif stack_direction == 2:
        image = image.unsqueeze(4).repeat(1, 1, 1, 1, image_size)
    return image


def main():
    resnet = torchvision.models.resnet101(pretrained=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    if not os.path.isdir(config.output_folder):
        os.makedirs(config.output_folder)
    backbone = I3ResNet(copy.deepcopy(resnet))
    backbone.load_state_dict(torch.load("models/model_3d_state_dict_n32_101.pth"), strict=False)
    # backbone = torch.load(r"models/model_3d_state_dict.pth")
    backbone.eval()

    extractor = FeatureExtractor(backbone=backbone,
                                 layers_to_extract_from=['layer2', 'layer3'],
                                 device=device,
                                 input_shape=(3, image_size, image_size, image_size))

    if model_size == 'small':
        pdn = get_pdn_small(out_channels, padding=True)
    elif model_size == 'medium':
        pdn = get_pdn_medium(out_channels, padding=True)
    elif model_size == 'xsmall':
        pdn = get_pdn_xsmall(out_channels, padding=True)
    else:
        raise Exception()
    print('pdn loaded')
    train_set = ImageFolderWithoutTarget(imagenet_train_path, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=7, pin_memory=True)
    train_loader = InfiniteDataloader(train_loader)

    channel_mean, channel_std = feature_normalization(extractor=extractor,
                                                      train_loader=train_loader, steps=10000)

    pdn.train()
    if on_gpu:
        pdn = pdn.cuda()

    optimizer = torch.optim.Adam(pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    tqdm_obj = tqdm(range(60000))
    for iteration, (image_fe, image_pdn) in zip(tqdm_obj, train_loader):
        image_fe = expand_2d_to_3d_image(image_fe, image_size)
        image_pdn = expand_2d_to_3d_image(image_pdn, half_image_size)
        if on_gpu:
            image_fe = image_fe.cuda()
            image_pdn = image_pdn.cuda()
        target = extractor.embed(image_fe)
        target = (target - channel_mean) / channel_std
        prediction = pdn(image_pdn)
        loss = torch.mean((target - prediction) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_obj.set_description(f'{(loss.item())}')

        if iteration % 10000 == 0:
            torch.save(pdn,
                       os.path.join(config.output_folder,
                                    f'teacher_{model_size}_tmp.pth'))
            torch.save(pdn.state_dict(),
                       os.path.join(config.output_folder,
                                    f'teacher_{model_size}_tmp_state.pth'))
    torch.save(pdn,
               os.path.join(config.output_folder,
                            f'teacher_{model_size}_final.pth'))
    torch.save(pdn.state_dict(),
               os.path.join(config.output_folder,
                            f'teacher_{model_size}_final_state.pth'))


@torch.no_grad()
def feature_normalization(extractor, train_loader, steps=10000):
    mean_outputs = []
    normalization_count = 0
    with tqdm(desc='Computing mean of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            image_fe = expand_2d_to_3d_image(image_fe, image_size)
            if on_gpu:
                image_fe = image_fe.cuda()
            output = extractor.embed(image_fe)
            mean_output = torch.mean(output, dim=[0, 2, 3, 4])
            mean_outputs.append(mean_output)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None, None]

    mean_distances = []
    normalization_count = 0
    with tqdm(desc='Computing variance of features', total=steps) as pbar:
        for image_fe, _ in train_loader:
            image_fe = expand_2d_to_3d_image(image_fe, image_size)
            if on_gpu:
                image_fe = image_fe.cuda()
            output = extractor.embed(image_fe)
            print(output.size())
            print(channel_mean.size())
            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3, 4])
            mean_distances.append(mean_distance)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, input_shape):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.input_shape = input_shape
        print('input feature', input_shape)
        self.patch_maker = PatchMaker(3, stride=1)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )

        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        print('feature dimensions', feature_dimensions)
        self.forward_modules["feature_aggregator"] = feature_aggregator
        preprocessing = Preprocessing(feature_dimensions, 1024)
        self.forward_modules["preprocessing"] = preprocessing

        preadapt_aggregator = Aggregator(target_dim=out_channels)

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.forward_modules.eval()

    @torch.no_grad()
    def embed(self, images):
        """Returns feature embeddings for images."""

        _ = self.forward_modules["feature_aggregator"].eval()
        features = self.forward_modules["feature_aggregator"](images)

        print('images', images.size())
        features = [features[layer] for layer in self.layers_to_extract_from]
        print('features of images layer2', features[0].size())
        print('features of images layer3', features[1].size())
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in
            features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]
        print('ref', ref_num_patches)

        for i in range(1, len(features)):
            _features = features[i]
            print(_features.size())
            patch_dims = patch_shapes[i]
            print('patch dims', patch_dims)
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], patch_dims[2],
                *_features.shape[2:]
            )
            print('features', _features.size())
            _features = _features.permute(0, -4, -3, -2, -1, 1, 2, 3)
            print('features', _features.size())
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-3:])
            print('features', _features.size())
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1], ref_num_patches[2]),
                mode="trilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-3], ref_num_patches[0], ref_num_patches[1], ref_num_patches[2]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2, 3, 4)
            _features = _features.reshape(len(_features), -1,
                                          *_features.shape[-4:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-4:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        print('preembed features', features.size())
        features = torch.reshape(features, (-1, eight_image_size, eight_image_size, eight_image_size, out_channels))
        features = torch.permute(features, (0, 4, 1, 2, 3))
        features = features[0:1]
        print('embed features', features.size())

        return features


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = unfoldNd.UnfoldNd(
            kernel_size=self.patchsize, stride=self.stride, padding=padding,
            dilation=1
        )
        unfolded_features = unfolder(features)
        print('features', features.size())
        print('unf', unfolded_features.size())
        number_of_total_patches = []
        for s in features.shape[-3:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, self.patchsize, -1
        )
        print('unf patch', unfolded_features.size())
        unfolded_features = unfolded_features.permute(0, 5, 1, 2, 3, 4)
        print('unf patch perm', unfolded_features.size())
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features,
                                     self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][
                        extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in
                self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


if __name__ == '__main__':
    main()

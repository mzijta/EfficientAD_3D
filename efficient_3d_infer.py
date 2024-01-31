#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import nibabel as nib
import random
from tqdm import tqdm
from common3d import get_autoencoder256, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
import skimage.transform as skTrans
import itk


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='embryos_checked',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output_3d_model')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 5000
image_size = 128
which_weights = 'tmp' # 'tmp' or 'final'
topad = True # True or False
experiment = 'embryos_checked_small_256_pad_out5000'

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    print(config)

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    
    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset, experiment)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, experiment)
    print(test_output_dir)
    if not os.path.isdir(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.isdir(test_output_dir):
        os.makedirs(test_output_dir)

    # load data
    full_train_set = list_full_paths(os.path.join(dataset_path, config.subdataset, 'train', 'good'))
    test_set_path = os.path.join(dataset_path, config.subdataset, 'test')
    test_set_bad = list_full_paths(os.path.join(test_set_path,'bad'))
    test_set_good = list_full_paths(os.path.join(test_set_path, 'good'))
    validation_set = list_full_paths(os.path.join(dataset_path, config.subdataset, 'val', 'good'))
    train_set = full_train_set

    #train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
    #                          num_workers=4, pin_memory=True)
    #train_loader_infinite = InfiniteDataloader(train_loader)
    #validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels, padding=topad)
        student = get_pdn_small(2 * out_channels, padding=topad)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels, padding=topad)
        student = get_pdn_medium(2 * out_channels, padding=topad)
    else:
        raise Exception()

    if topad:
        autoencoder = get_autoencoder256(out_channels)
    else:
        autoencoder = get_autoencoder256_nopad(out_channels)
 
    teacher = torch.load(os.path.join(train_output_dir, 'teacher_{}.pth'.format(which_weights)), map_location='cpu')
    student = torch.load(os.path.join(train_output_dir, 'student_{}.pth'.format(which_weights)), map_location='cpu')
    autoencoder = torch.load(os.path.join(train_output_dir,
                                         'autoencoder_{}.pth'.format(which_weights)), map_location='cpu')
    #teacher.load_state_dict(state_dict_teacher)
    #student.load_state_dict(state_dict_student)
    #autoencoder.load_state_dict(state_dict_ae)
    teacher.eval()
    student.eval()
    autoencoder.eval()


    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
    
    teacher_mean, teacher_std = teacher_normalization(teacher, train_set)

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set_good=test_set_good, test_set_defect=test_set_bad, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))

  






def test(test_set_good, test_set_defect, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    list_good_defect = len(test_set_good)*['good'] + len(test_set_defect)*['defect']
    test_set = test_set_good + test_set_defect
    i=0
    for image_in_list in test_set:
        image = load_nifti_image_to_5d_tensor(image_in_list, image_size)
        og_image = nib.load(image_in_list)
        og_image_array = og_image.get_fdata()
        shape_og_image = np.shape(og_image_array)
        orig_width = shape_og_image[0]
        orig_height = shape_og_image[1]
        orig_length = shape_og_image[2]
        #image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        print(map_combined.size())
        if topad:
            map_combined = map_combined
        else:
            map_combined = torch.nn.functional.pad(map_combined, (4,4,4,4,4,4))
        print(map_combined.size())
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width, orig_length), mode='trilinear')
        print(map_combined.size())
        map_combined = map_combined[0, 0].cpu().numpy()
        
        map_st = torch.nn.functional.interpolate(
            map_st, (orig_height, orig_width, orig_length), mode='trilinear')
        map_st = map_st[0, 0].cpu().numpy()
        
        map_ae = torch.nn.functional.interpolate(
            map_ae, (orig_height, orig_width, orig_length), mode='trilinear')
        map_ae = map_ae[0, 0].cpu().numpy()
        ##teacher_output_save = torch.nn.functional.pad(teacher(image), (4, 4, 4, 4, 4, 4))
        #teacher_output_save =  teacher(image)
        #teacher_output_save = torch.nn.functional.interpolate(teacher_output_save, (orig_height, orig_width, orig_length), mode='trilinear')
        #teacher_output_save = teacher_output_save[0,0].cpu().detach().numpy()
        #ae_output_save =  autoencoder(image)
        #ae_output_save = torch.nn.functional.interpolate(ae_output_save, (orig_height, orig_width, orig_length), mode='trilinear')
        #ae_output_save = ae_output_save[0, 0].cpu().detach().numpy()
        #st_output_save =  student(image)
        #st_output_save = torch.nn.functional.interpolate(st_output_save, (orig_height, orig_width, orig_length), mode='trilinear')
        #st_output_save = st_output_save[0, 0].cpu().detach().numpy()
        #inter_impose = np.stack((teacher_output_save, st_output_save, ae_output_save), axis=-1)

        defect_class = list_good_defect[i]
        if test_output_dir is not None:
            img_nm = os.path.basename(image_in_list)
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm)
            final_img = nib.Nifti1Image(map_combined, og_image.affine)
            nib.save(final_img, file)
            
            #if not os.path.exists(os.path.join(test_output_dir, 'inter')):
            #    os.makedirs(os.path.join(test_output_dir, 'inter'))
            #file = os.path.join(test_output_dir, 'inter', img_nm)
            #final_img = nib.Nifti1Image(inter_impose, og_image.affine)
            #nib.save(final_img, file)
            
            #class_name = defect_class + '_st'
            #if not os.path.exists(os.path.join(test_output_dir, class_name)):
            #    os.makedirs(os.path.join(test_output_dir, class_name))
            #file = os.path.join(test_output_dir, class_name, img_nm)
            #final_img = nib.Nifti1Image(map_st, og_image.affine)
            #nib.save(final_img, file)
            
            #class_name = defect_class + '_ae'
            #if not os.path.exists(os.path.join(test_output_dir, class_name)):
            #    os.makedirs(os.path.join(test_output_dir, class_name))
            #file = os.path.join(test_output_dir, class_name, img_nm)
            #final_img = nib.Nifti1Image(map_ae, og_image.affine)
            #nib.save(final_img, file)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        print(y_true_image)
        print(y_score_image)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        i = i + 1
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image_in_list in validation_loader:
        image = load_nifti_image_to_5d_tensor(image_in_list,image_size)
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image_in_list in train_loader:
        train_image = load_nifti_image_to_5d_tensor(train_image_in_list, image_size)
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3,4])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None, None]

    mean_distances = []
    for train_image_in_list in train_loader:
        train_image = load_nifti_image_to_5d_tensor(train_image_in_list, image_size)
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3,4])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

def load_nifti_image_to_5d_tensor(image_in_list, image_size_len):
    image = nib.load(image_in_list).get_fdata()
    result = skTrans.resize(image, (image_size_len, image_size_len, image_size_len), order=1, preserve_range=True)
    image = (result - np.min(result)) / (np.max(result) - np.min(result))
    image = np.resize(image, (1, 3, image_size_len, image_size_len, image_size_len)).astype(np.float32)
    image = torch.from_numpy(image)
    return image

if __name__ == '__main__':
    main()

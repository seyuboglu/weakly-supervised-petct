"""
Functional implementation of several basic data-augmentation transforms
for pytorch video inputs.

Sources:
- https://github.com/hassony2/torch_videovision
- https://github.com/YU-Zhiyang/opencv_transforms_torchvision
"""

import numbers
import random

import cv2
# from matplotlib import pyplot as plt
import numpy as np
import PIL
import scipy
import torch
import torchvision


INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}


def rotate(clip, degrees=0.0, resample='BILINEAR', expand=False, center=None):
    """
    Randomly rotates the image by +/- angle degrees.
    """
    degrees = (-degrees, degrees)
    degrees = random.uniform(degrees[0], degrees[1])
    t, h, w, c = clip.shape
    point = center or (w/2, h/2)

    M = cv2.getRotationMatrix2D(point, angle=-degrees, scale=1)
    rotated = np.array([cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample]) for img in clip])
    return rotated


def random_horizontal_flip(clip):
    """
    Args:
    img (PIL.Image or numpy.ndarray): List of images to be cropped
    in format (h, w, c) in numpy.ndarray
    Returns:
    PIL.Image or numpy.ndarray: Randomly flipped clip
    """
    if random.random() < 0.5:
        if isinstance(clip[0], np.ndarray):
            return [np.fliplr(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [
                img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
            ]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))
    return clip


def random_temporal_flip(clip):
    """Randomly flips the exam from head to toe.
    """
    clip = np.flip(clip, 0) if random.random() < 0.5 else clip
    return np.ascontiguousarray(clip)


def random_crop(clip, size):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """
    h, w = size
    if isinstance(clip[0], np.ndarray):
        im_h, im_w, im_c = clip[0].shape
    elif isinstance(clip[0], PIL.Image.Image):
        im_w, im_h = clip[0].size
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    if w > im_w or h > im_h:
        error_msg = (
            'Initial image size should be larger then '
            'cropped size but got cropped sizes : ({w}, {h}) while '
            'initial image is ({im_w}, {im_h})'.format(
                im_w=im_w, im_h=im_h, w=w, h=h))
        raise ValueError(error_msg)

    x1 = random.randint(0, im_w - w)
    y1 = random.randint(0, im_h - h)
    cropped = crop_clip(clip, y1, x1, h, w)
    resized = resize_clip(clip, (im_w, im_h))
    return cropped


def center_crop(clip, size):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """
    h, w = size
    if isinstance(clip[0], np.ndarray):
        im_h, im_w, im_c = clip[0].shape
    elif isinstance(clip[0], PIL.Image.Image):
        im_w, im_h = clip[0].size
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    if w > im_w or h > im_h:
        error_msg = (
            'Initial image size should be larger then '
            'cropped size but got cropped sizes : ({w}, {h}) while '
            'initial image is ({im_w}, {im_h})'.format(
                im_w=im_w, im_h=im_h, w=w, h=h))
        raise ValueError(error_msg)

    x1 = int(round((im_w - w) / 2.))
    y1 = int(round((im_h - h) / 2.))
    cropped = crop_clip(clip, y1, x1, h, w)
    resized = resize_clip(clip, (im_w, im_h))
    return cropped


def color_jitter(clip, brightness=[], contrast=[], saturation=[], hue=[]):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    clip (np.ndarray) (T, H, W, C) matrix
    dims (list) the indices of the channels to change
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """
    if isinstance(clip[0], np.ndarray):
        num_channels = clip.shape[3] # iterate through channels
        if len(brightness) == 0:
            brightness = [0] * num_channels
        if len(contrast) == 0:
            contrast = [0] * num_channels
        if len(saturation) == 0:
            saturation = [0] * num_channels
        if len(hue) == 0:
            hue = [0] * num_channels

        for i in range(num_channels):
            brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                get_jitter_params(
                    brightness[i], contrast[i], saturation[i], hue[i])
            if brightness_factor != None:
                clip[:,:,:,i] = clip[:,:,:,i] * brightness_factor
            if contrast_factor != None:
                mean = round(clip[:,:,:,i].mean())
                clip[:,:,:,i] = (1 - contrast_factor) * mean + contrast_factor * clip[:,:,:,i]
            if saturation_factor != None:
                raise NotImplementedError('Saturation augmentation ' +
                                          'not yet implemented')
            if hue_factor != None:
                raise NotImplementedError('Hue augmentation not ' +
                                          'yet implemented.')

    else:
        raise TypeError('Expected numpy.ndarray ' +
                        'but got list of {0}'.format(type(clip[0])))
    return clip


def get_jitter_params(brightness, contrast, saturation, hue):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(
            max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = np.array([img[min_h:min_h + h, min_w:min_w + w, :] for img in clip])

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = np.array([
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ])
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

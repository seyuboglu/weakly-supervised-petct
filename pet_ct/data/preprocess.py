"""

"""
import numpy as np
import cv2


def trim_exams(image_channels):
    """
    Trims the larger exam to match length.

    Receives arbitrary list of (num_frames, height, width) as input. The number
    of frames between the exams will be trimmed to match the dimensionality
    of the smallest exams.
    """
    min_frames = min([channel.shape[0] for channel in image_channels])
    return [channel[:min_frames] for channel in image_channels]


def join_channels(image_channels):
    """
    """
    # if len(image_channels) > n_channels:
    #     raise(Exception("More image channels"))
    image_channels = trim_exams(image_channels)
    # image_channels.extend(image_channels[-1:] *
    #                       (n_channels - len(image_channels)))
    exam = np.stack(image_channels, axis=-1)
    return exam


def resize(images, size):
    """
    transform
    Resizes a stack of images. Note: cv2.resize expects the n_frames axis last,
    we use np.moveaxis to address this.

    args:
        images  (np.ndarray) (n_frames, height, width)
        size    (tuple) (new_height, new_width)
    return
        images  (np.ndarray) (n_frames, size[0], size[1])
    """
    resized_images = []
    for i in range(images.shape[0]):
        image = images[i, :, :]
        resized_images.append(cv2.resize(image, size))
    return np.stack(resized_images, axis=0)


def normalize(images):
    """
    """
    return (images - np.mean(images)) / np.std(images)


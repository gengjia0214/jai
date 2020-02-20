"""
Image processing methods.
Mainly for preprocessing.
"""

import cv2 as cv
import numpy as np
import torch


def reverse_color(img):
    """
    Inverse the color
    :param img: img
    :return: color reversed img
    """

    if img.dtype != np.uint8:
        raise Exception("Input image should be in np.uint8 but was {}".format(img.dtype))

    return 255 - img


def grayscale(img, order='BGR'):
    """
    Gray scale the image to single channel
    :param img: image
    :param order: input image order: BGR for cv.imread amd RBG for PIL
    :return: gray-scaled image
    """

    if order == 'BGR':
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif order == 'RBG':
        return cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    else:
        raise Exception("Channel order must be either BGR (opencv) or RBG (PIL) but was {}.".format(order))


def denoise(img, ksize=(3, 3)):
    """
    Denoise the image
    :param img: image
    :param ksize: kernel size for smoothing
    :return: denoised image
    """

    img = cv.blur(img, ksize=ksize)
    return img


def threshold(img, low=0, adaptive_ksize=None, C=None, binary=False):
    """
    Adaptive threshold only work for single channel image
    Threshold the image. Hard threshold or/and adaptive threshold.
    Adaptive threshold will use the mean of the kernal - C
    :param img: image
    :param low: if not None pixel below the lower bound will be set to 0
    :param adaptive_ksize: if not none, apply adaptive threshold
    :param C: value that will be subtracted from the adaptive kernel mean
    :param binary: whether return binary output
    :return: thresholded image
    """

    if low > 0:
        img[img < low] = 0
    mask = None
    if adaptive_ksize:
        mask = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                    thresholdType=cv.THRESH_BINARY, blockSize=13, C=C)
    if mask is not None:
        if binary:
            return mask
        else:
            return img * (mask == 255)
    return img


def centralize_object(img):
    """
    Translate the image to the pixel center
    Works for 0 background images after the threshold. If not thresholded, this method does not have effect.
    :param img: image
    :return: centralized image
    """

    height, width = img.shape[0], img.shape[1]
    col_map = np.repeat(np.arange(0, width)[np.newaxis, :], height, axis=0)
    row_map = np.repeat(np.arange(0, height)[:, np.newaxis], width, axis=1)
    row_centroid = row_map[img != 0].mean()  # get the nonzero pixels across all rows and find the mean
    col_centroid = col_map[img != 0].mean()  # get the nonzero pixels across all cols and find the mean
    row_t = height / 2 - row_centroid
    col_t = width / 2 - col_centroid
    T = np.array([[1, 0, col_t], [0, 1, row_t]])  # translate to the pixel center
    img = cv.warpAffine(img, T, (width, height))
    return img


def resize_object(img, size, pad, margin=5):
    """
    Crop the image using the the non-zero boundary (object), pad and resize
    Code borrowed from https://www.kaggle.com/iafoss/image-preprocessing-128x128
    :param img: image
    :param pad: padding around the bounding box
    :param size: new size
    :param margin: margin of the bounding box around the object
    :return:
    """

    # get the valid range (bbox for the writing) along two axes
    mask = img != 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # crop the bbox with some additional margins
    cmin = cmin - margin if (cmin > margin) else 0
    rmin = rmin - margin if (rmin > margin) else 0
    cmax = cmax + margin if (cmax < img.shape[1] - margin) else img.shape[1]
    rmax = rmax + margin if (rmax < img.shape[0] - margin) else img.shape[0]
    img = img[rmin:rmax, cmin:cmax]

    # cropped size
    lc, lr = cmax - cmin, rmax - rmin
    # find longer edge
    ledge = max(lc, lr) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((ledge - lr) // 2,), ((ledge - lc) // 2,)], mode='constant')
    return cv.resize(img, (size, size))


def rescale(img, dtype=np.float32):
    """
    rescale image to 0-1
    :param img: image
    :param dtype: data type for the ouput
    :return: normalized image
    """

    img = (img / (img.max() + 1e-8)).astype(dtype)
    return img


def standardize(img, dtype=np.float32):
    """
    standardize the data to 0 mean and unit var
    :param img: image
    :param dtype: output datatype
    :return: standardized data
    """

    img -= img.mean()
    img = img / (img.std() + 1e-8)
    img.astype(dtype)

    return img


def add_bgnoise(mu, sigma, img):
    """
    Add some background noise to the binary image
    This will convert image to float32
    :param mu: mean
    :param sigma: std
    :param img: image
    :return:
    """

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    gauss_noise = np.random.normal(mu, sigma, img.shape[0]*img.shape[1]).reshape(img.shape)
    img += gauss_noise
    img[img > 1.0] = 1.0
    img[img < 0] = 0
    return img


def to_tensor(img, dtype=torch.float32):
    """
    To tensor in float
    :param img: image
    :param dtype: output data type
    :return: image in tensor and the
    """

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    if img.ndim == 2:  # if the input image is grayscale, add the pseudo dimension
        img = torch.from_numpy(img)[None, :, :].to(dtype)  # HxW -> 1xHxW
    elif img.ndim == 3:
        img = torch.from_numpy(img).permute((2, 0, 1)).to(dtype)  # HxWx3 -> 3xHxW

    return img


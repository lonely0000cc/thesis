import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
from PIL import Image
import numbers
import torch
import math
import random

class Center_crop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        w1 = int(round((w - tw) / 2.))
        h1 = int(round((h - th) / 2.))
        return img[h1:h1+th, w1:w1+tw, :]

class Normalize(object):
    def __init__(self, mean=0., std=255.):
        self.mean=mean
        self.std=std

    def __call__(self, image):
        image = image - self.mean
        image = image / self.std
        return image

class Merge(object):
    """Merge a group of images
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")


class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            slices_.append(s)
        self.slices = slices_

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                #sl = [slice(None)] * image.ndim
                #sl[self.axis] = s
                ret.append(image[:, :, s[0]:s[1]])
            return ret
        else:
            raise Exception("obj is not an numpy array")

class Random_rotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) or torch.tensor (B x H x W x C) randomly
    """
    def __init__(self, mode, angle_range=(0.0, 360.0), axes=(0, 1)):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        elif isinstance(image, torch.Tensor):
            B, C, H, W = image.size()
            x = torch.arange(0, W, 1).float().cuda() 
            y = torch.arange(0, H, 1).float().cuda()

            x = x - (W-1)/2.
            y = y - (H-1)/2.
            xx = x.repeat(H, 1)
            yy = y.view(H, 1).repeat(1, W)
            grid = torch.stack([xx, yy], dim=0) 

            matrix = torch.tensor([[math.cos(angle), math.sin(angle)],
                                   [-math.sin(angle), math.cos(angle)]]).float().cuda()
            
            grid = torch.mm(matrix, grid.view(2, -1)).view(2, H, W)
            grid[0, :, :] = grid[0, :, :] / ((W - 1)/2.)
            grid[1, :, :] = grid[1, :, :] / ((H - 1)/2.)
            grid = torch.stack([grid[0, :, :], grid[1, :, :]], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
            image = torch.nn.functional.grid_sample(image, grid, padding_mode=self.mode)
            return image
        else:
            raise Exception('unsupported type')

class Crop(object):
    def __init__(self, size, tl):
        self.size = size
        self.tl = tl
        random.seed()

    def __call__(self, img):
        ch, cw = self.size
        tl_h, tl_w = self.tl
        return img[tl_h : ch + tl_h, tl_w : cw + tl_w, :]

class Random_crop(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        random.seed()

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        elif isinstance(img, torch.Tensor):
            _, _, h, w = img.size()

        th, tw = self.size
        if w == tw and h == th:
            return img

        if w==tw:
            w1=0
        else:
            w1 = random.randint(0, w - tw)

        if h==th:
            h1=0
        else:
            h1 = random.randint(0, h - th)

        if isinstance(img, np.ndarray):
            return img[h1: h1 + th, w1:w1 + tw, :]
        elif isinstance(img, torch.Tensor):
            return img[:, :, h1: h1 + th, w1:w1 + tw]

class Random_mask(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w, c = img.shape[:3]
        elif isinstance(img, torch.Tensor):
            b, c, h, w = img.size()

        th, tw = self.size
        if w == tw and h == th:
            return img

        if w==tw:
            w1=0
        else:
            w1 = random.randint(0, w - tw)

        if h==th:
            h1=0
        else:
            h1 = random.randint(0, h - th)

        if isinstance(img, np.ndarray):
            ret = np.zeros((h, w, c))
            ret[h1: h1 + th, w1:w1 + tw, :] = img[h1: h1 + th, w1:w1 + tw, :] 
        elif isinstance(img, torch.Tensor):
            ret = torch.zeros(b, c, h, w).float()
            ret[:, :, h1: h1 + th, w1:w1 + tw] = img[:, :, h1: h1 + th, w1:w1 + tw]
        return ret


class Rotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """
    def __init__(self, angle_degrees, axes=(0, 1), mode='reflect'):
        self.angle = angle_degrees
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = self.angle
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return scipy.misc.imresize(image, self.size)
        elif isinstance(image, Image.Image):
            return image.resize(self.size, Image.BILINEAR)
        else:
            raise Exception('unsupported type')

class Random_gamma_shift(object):
    def __init__(self, gamma_low=0.8, gamma_high=1.2, P=0.5):
        self.gamma_low=gamma_low
        self.gamma_high=gamma_high
        self.P = P

    def __call__(self, image):
        P = np.random.uniform(0, 1, 1)
        if P > self.P:
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            image = image ** random_gamma
        return image

class Random_brightness_shift(object):
    def __init__(self, brightness_low=0.5, brightness_high=2.0, P=0.5):
        self.brightness_low=brightness_low
        self.brightness_high=brightness_high
        self.P = P

    def __call__(self, image):
        P = np.random.uniform(0, 1, 1)
        if P > self.P:
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            image = image * random_brightness
        return image      

class Clip(object):
    def __init__(self, low=0., high=1.):
        self.low = low
        self.high = high

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return np.clip(image, self.low, self.high)
        elif isinstance(image, torch.Tensor):
            return image.clamp(self.low, self.high)

class To_tensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img
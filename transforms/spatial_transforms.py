from __future__ import absolute_import

import random
import math
import numbers
import collections
from torchvision.transforms import *
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
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
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        if tensor.size(0) == 1:
            return tensor

        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class NormalizeSub(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        if tensor.size(0) == 1:
            mean = [0.5]
            std = [0.25]
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
            return tensor

        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)

    def randomize_parameters(self):
        pass


class RandomCrop(object):
    """Crops the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size

        x1 = int(round(self.tl_x * (w - tw)))
        y1 = int(round(self.tl_y * (h - th)))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        self.tl_x = random.random()
        self.tl_y = random.random()



class pad(Pad):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        super(pad, self).__init__(padding, fill, padding_mode)

    def __call__(self, imgs):
        pad_imgs = []
        for img in imgs:
            pad_imgs.append(F.pad(img, self.padding, self.fill, self.padding_mode))

        return pad_imgs
    def randomize_parameters(self):
        pass

class random_crop(RandomCrop):

    def __init__(self, size, padding=0, pad_if_needed=False):
        super(random_crop, self).__init__(size, padding, pad_if_needed)

    def __call__(self, imgs):

        # print(len(imgs))
        i, j, h, w = self.get_params(imgs[0], self.size)

        crop_imgs = []
        for img in imgs:
            crop_imgs.append(F.crop(img, i, j, h, w))

        return crop_imgs
        
class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.scales) - 1)]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, size, p=0.5, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.height, self.width = self.size
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if not self.cropping:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(self.tl_x * x_maxrange))
        y1 = int(round(self.tl_y * y_maxrange))
        return resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))

    def randomize_parameters(self):
        self.cropping = random.uniform(0, 1) < self.p
        self.tl_x = random.random()
        self.tl_y = random.random()



class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, height=256, width=128, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.height = height
        self.width = width
       
    def __call__(self, img):

        if self.p > self.probability or not self.erasing:
            return img

        x1 = self.x1
        y1 = self.y1
        h, w = self.h, self.w
        img[0, x1:x1+h, y1:y1+w] = self.mean[0]
        img[1, x1:x1+h, y1:y1+w] = self.mean[1]
        img[2, x1:x1+h, y1:y1+w] = self.mean[2]
        return img

    def randomize_parameters(self):
        self.p = random.uniform(0, 1)
        if self.p > self.probability:
            return
        self.erasing = False

        for attempt in range(100):
            area = self.height * self.width
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < self.width and h < self.height:
                self.x1 = random.randint(0, self.height - h)
                self.y1 = random.randint(0, self.width - w)
                self.h = h
                self.w = w
                self.erasing = True
                break

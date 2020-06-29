import torch
import torch.nn as nn
import numpy as np
import skimage
from skimage.transform import downscale_local_mean
import PIL.Image as Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_low_contrast(image_array, h1, w1, crop_size_H, crop_size_W, scale):
	image_array = image_array[h1: h1 + crop_size_H, w1:w1 + crop_size_W, :]
	image_array = skimage.img_as_ubyte(image_array)
	return skimage.exposure.is_low_contrast(
		downscale_local_mean(image_array, (scale, scale, 1)).astype(np.uint8),
		fraction_threshold=0.2)

def load_as_PIL_image(filename):
	im = Image.open(filename)
	im = im.convert('RGB')
	return im

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()
    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0) 
    return grid

def zero_out_pixels(list_shape, prop=0.5):
    mask = torch.rand([1]+[1]+list_shape[2:]).cuda()
    mask[mask<prop] = 0
    mask[mask!=0] = 1
    mask = mask.repeat(list_shape[0],list_shape[1],1,1)
    return mask
def generator_noise(list_shape):
    noise = torch.randn([1]+[1]+list_shape[2:]).cuda()
    noise = noise.repeat(list_shape[0],list_shape[1],1,1)
    return noise

class grid_gradient_central_diff():
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
    
        self.padding=None
        if padding:
            self.padding = nn.ReplicationPad2d(1)

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()
        
        fx_ = torch.tensor([[1,-1],[0,0]]).cuda()
        fy_ = torch.tensor([[1,0],[-1,0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1,0],[0,-1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i,i,:,:] = fxy_
            
        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy
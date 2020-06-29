import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
from utils.functions import grid_gradient_central_diff
from math import ceil
#import cv2
#import skimage
#import os
###############################################################################
# Functions
###############################################################################

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target, weight=None, mean=False):
        error = torch.abs(output - target)
        if weight is not None:
            error = error * weight.float()
            if mean!=False:
                return error.sum() / weight.float().sum()
        if mean!=False:
            return error.mean()
        return error.sum()

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target, weight=None, mean=False):
        error = output - target
        error2 = error**2
        if weight is not None:
            error2 = error2 * weight.float()
            if mean!=False:
                return error2.sum() / weight.float().sum()
        if mean!=False:
            return error2.mean()
        return error2.sum()

class sqrtL2(nn.Module):
    def __init__(self):
        super(sqrtL2, self).__init__()
    def forward(self, output, target, weight=None, mean=False):
        error = torch.norm(output-target, p=2, dim=1).unsqueeze(1)
        if weight is not None:
            error = error * weight.float()
            if mean!=False:
                return error.sum() / weight.sum()
        if mean!=False:
            return error.mean()
        return error.sum()

class VariationLoss(nn.Module):
    def __init__(self, nc, grad_fn=grid_gradient_central_diff):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)

    def forward(self, image, weight=None, mean=False):
        dx, dy = self.grad_fn(image)
        variation = torch.abs(dx) + torch.abs(dy)

        if weight is not None:
            variation = variation * weight.float()
            if mean!=False:
                return variation.sum() / weight.sum()
        if mean!=False:
            return variation.mean()
        return variation.sum()

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class PerceptualLoss():
    
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model
        
    def __init__(self):

        self.criterion = nn.MSELoss()
        self.contentFunc = self.contentFunc()
            
    def get_loss(self, fakeIm, realIm):
        
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
'''
class PerceptualLoss(nn.Module):

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.contentFunc = self.contentFunc()
    
    def forward(self, fakeIm, realIm):
        
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
'''
'''
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        self.loss = nn.MSELoss()
        
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
'''
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        #self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()
        
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def get_g_loss(self, input, target_is_real):

        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class DiscLossWGANGP(GANLoss):

    def __init__(self, LAMBDA=0.1):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = LAMBDA
        
    #def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
    #    self.D_fake = net.forward(fakeB)
    #    return -self.D_fake.mean()
        
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        
        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty
        
    def __call__(self, net, real_data, fake_data, use_gp=True):
        self.D_fake = net.forward(fake_data.detach())
        #self.D_fake = self.D_fake.mean()
        #self.loss_fake = self.get_g_loss(self.D_fake, target_is_real=False)
        self.loss_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(real_data)
        #self.D_real = self.D_real.mean()
        #self.loss_real = self.get_g_loss(self.D_real, target_is_real=True)
        self.loss_real = -self.D_real.mean()
        # Combined loss
        self.loss_D = self.loss_fake + self.loss_real
        if use_gp:
            self.loss_D = self.loss_D + self.calc_gradient_penalty(net, real_data, fake_data)
        return self.loss_D

class FreqLoss(nn.Module):
    def __init__(self):
        super(FreqLoss, self).__init__()
        self.loss = nn.L1Loss()
    '''
    def fft(self, img):
        f = cv2.dft(np.float32(img), flags=cv2.DFT_SCALE | cv2.DFT_COMPLEX_OUTPUT)
        f_shift = np.fft.fftshift(f)
        return f_shift

    def ift(self, f_shift):
        f_shifted = np.fft.ifftshift(f_shift)
        #inv_img = cv2.idft(f_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        inv_img = cv2.idft(f_shifted, flags=cv2.DFT_REAL_OUTPUT)
        inv_img[inv_img>1]=1
        inv_img[inv_img<-1]=-1
        return inv_img

    def norm(self, f_shift):
        f_abs = cv2.magnitude(f_shift[:,:,0],f_shift[:,:,1]) + 1
        f_bounded = 20 * np.log(f_abs)
        #print(f_bounded.min(), f_bounded.max())
        f_img = 255 * f_bounded / np.max(f_bounded)
        f_img = f_img.astype(np.uint8)
        return f_img
    '''
    def forward(self, input, target):
        fake_fft = torch.rfft(input=input, signal_ndim=2, normalized=True, onesided=False)
        real_fft = torch.rfft(input=target, signal_ndim=2, normalized=True, onesided=False)
        #return self.loss(fake_fft.detach(), real_fft)
        return self.loss(fake_fft, real_fft)

class TextureLoss(nn.Module):
    def __init__(self, patch_size=16, use_patch=False):
        super(TextureLoss, self).__init__()
        self.patch_size = patch_size
        self.use_patch = use_patch
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def gram_matrix(self, x):
        # b, c, h, w
        b, c, h, w = x.shape
        #result = torch.tensor(0.0).cuda().expand(b, c, c)
        x = x.view(-1, c, h * w)
        #for i in range(b):
        #    result[i] = torch.matmul(x[i:i+1, :, :], x[i:i+1, :, :].permute(0, 2, 1))[0]
        try:
            return torch.matmul(x, x.permute(0, 2, 1))
        except:
            return torch.matmul(x, x.permute(0, 2, 1))
        #return result
        #return torch.bmm(x, y)
        #torch.bmm(x[0:1, :, :], y[0:1, :, :])

    def square_diff(self, x, y):
        return (x - y) * (x - y)

    def patch_resize(self, x):
        b, c, _, _ = x.shape
        # b, c*k*k, H/k*W/k
        x = self.unfold(x)
        _, _, l = x.shape
        # b, c, k, k, H/k*W/k
        x = x.view(b, c, self.patch_size, self.patch_size, -1)
        # b, H/k*W/k, c, k, k
        x = x.permute(0, 4, 1, 2, 3)
        try:
            x = x.view(-1, c, self.patch_size, self.patch_size)
        except:
            x = x.contiguous().view(-1, c, self.patch_size, self.patch_size)
        return x


    def forward(self, true_texture, fake_texture):
        #print(true_texture[0].shape, fake_texture[0].shape)
        if self.use_patch:
            loss0 = self.square_diff(self.gram_matrix(self.patch_resize(true_texture[0])), \
                self.gram_matrix(self.patch_resize(fake_texture[0]))).mean()
            loss1 = self.square_diff(self.gram_matrix(self.patch_resize(true_texture[1])), \
                self.gram_matrix(self.patch_resize(fake_texture[1]))).mean()
            loss2 = self.square_diff(self.gram_matrix(self.patch_resize(true_texture[2])), \
                self.gram_matrix(self.patch_resize(fake_texture[2]))).mean()
        else:
            loss0 = self.square_diff(self.gram_matrix(true_texture[0]), self.gram_matrix(fake_texture[0])).mean()
            loss1 = self.square_diff(self.gram_matrix(true_texture[1]), self.gram_matrix(fake_texture[1])).mean()
            loss2 = self.square_diff(self.gram_matrix(true_texture[2]), self.gram_matrix(fake_texture[2])).mean()

        return loss0 + loss1 + loss2
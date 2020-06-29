import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
'''
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
'''
class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

'''
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):

        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out

class BasicBlockSig(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.body(x)
        return out


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Downsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m=[]
        if (scale & (scale -1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, n_feats//4, 3, bias))
                m.append(PixelUnshuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act =='relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, n_feats//9, 3, bias))
            m.append(PixelUnshuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*m)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def pixel_unshuffle(self, input, downscale_factor):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        c = input.shape[1]
        kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
            1, downscale_factor, downscale_factor],
            device=input.device)
        for y in range(downscale_factor):
            for x in range(downscale_factor):
                kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
        return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k

        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return self.pixel_unshuffle(input, self.downscale_factor)




def deconv_activation(in_ch, out_ch ,activation = 'relu' ):

    if activation == 'relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True))

def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'relu'):


    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding))

class Encoder(nn.Module):

    def __init__(self, input_nc=3, output_nc=64, activation='selu'):
        super(Encoder, self).__init__()

        self.layer_f = conv_activation(input_nc,  64, kernel_size = 5, stride = 1, padding = 2, activation = activation)
        self.conv1   = conv_activation(64,        64, kernel_size = 5, stride = 1, padding = 2, activation = activation)
        self.conv2   = conv_activation(64,        64, kernel_size = 5, stride = 2, padding = 2, activation = activation)
        self.conv3   = conv_activation(64,        64, kernel_size = 5, stride = 2, padding = 2, activation = activation)
        self.conv4   = conv_activation(64, output_nc, kernel_size = 5, stride = 2, padding = 2, activation = activation)

    def forward(self,x):

        layer_f = self.layer_f(x)
        conv1 = self.conv1(layer_f)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        return conv1,conv2,conv3,conv4


class Decoder(nn.Module):

    def __init__(self, input_nc=128, output_nc=3, activation='selu'):
        super(Decoder, self).__init__()

        self.warp_deconv4 = deconv_activation(input_nc, 64, activation=activation)
        self.warp_deconv3 = deconv_activation(192,      64, activation=activation)
        self.warp_deconv2 = deconv_activation(192,      64, activation=activation)
        self.post_fusion1 = conv_activation(192,   64, kernel_size=5, stride=1, padding=2, activation=activation)
        self.post_fusion2 = conv_activation(64,    64, kernel_size=5, stride=1, padding=2, activation=activation)
        self.final        = conv_activation(64, output_nc, kernel_size=5, stride=1, padding=2, activation='linear')

    def forward(self,LR_conv1, LR_conv2, LR_conv3, LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4):

        concat0 = torch.cat((LR_conv4,warp_conv4),1)
        warp_deconv4 = self.warp_deconv4(concat0)

        concat1 = torch.cat((warp_deconv4,LR_conv3,warp_conv3),1)
        warp_deconv3 = self.warp_deconv3(concat1)

        concat2 = torch.cat((warp_deconv3,LR_conv2,warp_conv2),1)
        warp_deconv2 = self.warp_deconv2(concat2)

        concat3 = torch.cat((warp_deconv2,LR_conv1,warp_conv1),1)
        post_fusion1 = self.post_fusion1(concat3)

        post_fusion2 = self.post_fusion2(post_fusion1)
        final = self.final(post_fusion1)

        return final
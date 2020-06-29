import torch
import torch.nn as nn
import numpy as np
import networks.basenet as basenet

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.SELU(inplace = True))
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.SELU(inplace = True))

def generate_noise(opts):
    noise = torch.randn(opts.batch_size, opts.n_colors, opts.im_crop_H, opts.im_crop_W).cuda()
    #noise = upsampling(noise,size[1], size[2])
    return noise

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=9, learn_residual=False, padding_type='reflect'):

        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        use_bias = True

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):            
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        padAndConv = {
            'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [
                norm_layer(dim),
                nn.ReLU(True)
            ] + [
                nn.Dropout(0.5)
            ] if use_dropout else [] + padAndConv[padding_type] + [
                norm_layer(dim)
            ]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block = nn.Sequential(*blocks)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class UnetFlowGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(UnetFlowGenerator, self).__init__()

        self.conv1a  = conv(input_nc,   16, kernel_size=3, stride=2, padding=1)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1, padding=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1, padding=1)

        self.conv2a  = conv(16,  32, kernel_size=3, stride=2, padding=1)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1, padding=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1, padding=1)

        self.conv3a  = conv(32,  64, kernel_size=3, stride=2, padding=1)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1, padding=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1, padding=1)

        self.conv4a  = conv(64,  96, kernel_size=3, stride=2, padding=1)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1, padding=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1, padding=1)

        self.deconv4   = deconv(96,  64, kernel_size = 4, stride = 2, padding = 1)
        self.deconv40  = conv(64,64, kernel_size=3, stride=1, padding=1)
        self.deconv400 = conv(64,64, kernel_size=3, stride=1, padding=1)

        self.deconv3   = deconv(128, 32, kernel_size = 4, stride = 2, padding = 1)
        self.deconv30  = conv(32,32, kernel_size=3, stride=1, padding=1)
        self.deconv300 = conv(32,32, kernel_size=3, stride=1, padding=1)

        self.tail1  = conv(64,   64, kernel_size=3, stride=1, padding=1)
        self.tail2  = conv(64,   64, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, input):

        c1 = self.conv1b(self.conv1aa(self.conv1a(input)))
        c2 = self.conv2b(self.conv2aa(self.conv2a(c1)))
        c3 = self.conv3b(self.conv3aa(self.conv3a(c2)))
        c4 = self.conv4b(self.conv4aa(self.conv4a(c3)))

        deconv4 = self.deconv400(self.deconv40(self.deconv4(c4)))
        x3 = torch.cat((deconv4, c3),1)
        deconv3 = self.deconv300(self.deconv30(self.deconv3(x3)))

        x2 = torch.cat((deconv3, c2),1)
        tail1 = self.tail1(x2)
        tail2 = self.tail2(tail1)
        output = self.output(tail2)

        return output


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32):
        super(UnetGenerator, self).__init__()
        assert (input_nc == output_nc)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        return output


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        #dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=3, stride=2, padding=1)
        dConv = nn.Sequential(
            nn.Conv2d(outer_nc, inner_nc, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1))
        dRelu = nn.LeakyReLU(0.2, True)
        #dNorm = norm_layer(inner_nc)
        #uRelu = nn.ReLU(True)
        uRelu = nn.SELU(inplace = True)
        #uNorm = norm_layer(outer_nc)
        model = []

        if outermost:
            #uConv  = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            uConv  = nn.ConvTranspose2d(inner_nc * 2, inner_nc, kernel_size=4, stride=2, padding=1)
            tail   = nn.Sequential(
                        conv(inner_nc,   inner_nc, kernel_size=3, stride=1, padding=1),
                        conv(inner_nc,   inner_nc, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1))
            dModel = [dConv]
            #uModel = [uRelu, uConv, nn.Tanh()]
            uModel = [uRelu, uConv, tail]
            #model  = [dModel, uModel]
            model  += dModel + [submodule] + uModel
        elif innermost:
            uConv  = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv]
            #model  = [dModel, uModel]
            model  += dModel + uModel
        else:
            uConv  = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv]
            #model  = [dModel, uModel]
            model += dModel + [submodule] + uModel

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            #print(1, x.size())
            #print(2, self.model(x).size())
            return torch.cat([self.model(x), x], 1)





class FeatureFusionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FeatureFusionGenerator, self).__init__()

        #construct unet structure
        self.head = conv(input_nc, 64, kernel_size = 5, stride = 1, padding = 2)
        self.conv0 = conv(64,       64, kernel_size = 5, stride = 1, padding = 2)

        self.conv1 = conv(64,      128, kernel_size = 5, stride = 2, padding = 2)
        self.conv2 = conv(128,     256, kernel_size = 5, stride = 2, padding = 2)
        self.conv3 = conv(256,     512, kernel_size = 5, stride = 2, padding = 2)
        self.conv4 = conv(512,    1024, kernel_size = 5, stride = 2, padding = 2)
        #self.conv5 = conv(1024,   2048, kernel_size = 5, stride = 2, padding = 2)
        #self.conv6 = conv(2048,   4096, kernel_size = 5, stride = 2, padding = 2)

        #self.deconv6 = deconv(8192, 2048, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv5 = deconv(6144, 1024, kernel_size = 5, stride = 2, padding = 2)
        self.deconv4 = deconv(2048,  512, kernel_size = 4, stride = 2, padding = 1)
        self.deconv3 = deconv(1536,  256, kernel_size = 4, stride = 2, padding = 1)
        self.deconv2 = deconv(768,   128, kernel_size = 4, stride = 2, padding = 1)
        self.deconv1 = deconv(384,    64, kernel_size = 4, stride = 2, padding = 1)

        self.tail1  = conv(192,   64, kernel_size=5, stride=1, padding=2)
        self.tail2  = conv(64,    64, kernel_size=5, stride=1, padding=2)
        #self.output = conv(64, output_nc, kernel_size=5, stride=1, padding=2, activation='linear')
        self.output = nn.Conv2d(64, output_nc, kernel_size=5, stride=1, padding=2)

    def forward(self, input, flows, Backward_warper, reference):

        c1h = self.head(input)
        c10 = self.conv0(c1h)
        c11 = self.conv1(c10)
        c12 = self.conv2(c11)
        c13 = self.conv3(c12)
        c14 = self.conv4(c13)

        c2h = self.head(reference)
        c20 = self.conv0(c2h)
        c21 = self.conv1(c20)
        c22 = self.conv2(c21)
        c23 = self.conv3(c22)
        c24 = self.conv4(c23)

        warp_conv4   = Backward_warper(c24, flows[4]*1.25)
        x4 = torch.cat((c14, warp_conv4),1)
        warp_deconv4 = self.deconv4(x4)

        warp_conv3   = Backward_warper(c23, flows[3]*2.5)
        x3 = torch.cat((warp_deconv4, c13, warp_conv3),1)
        warp_deconv3 = self.deconv3(x3)

        warp_conv2   = Backward_warper(c22, flows[2]*5.0)
        x2 = torch.cat((warp_deconv3, c12, warp_conv2),1)
        warp_deconv2 = self.deconv2(x2)

        warp_conv1   = Backward_warper(c21, flows[1]*10.0)
        x1 = torch.cat((warp_deconv2, c11, warp_conv1),1)
        warp_deconv1 = self.deconv1(x1)

        warp_conv0   = Backward_warper(c20, flows[0]*20.0)
        x0 = torch.cat((warp_deconv1, c10, warp_conv0),1)
        tail1 = self.tail1(x0)
        tail2 = self.tail2(tail1)
        output = self.output(tail2)

        return output

class CrossFusionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(CrossFusionGenerator, self).__init__()

        #construct unet structure
        self.head = conv(input_nc, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv0 = conv(64,      64, kernel_size = 3, stride = 1, padding = 1)

        self.conv1 = conv(64,      64, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = conv(64,      64, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = conv(64,      64, kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = conv(64,      64, kernel_size = 3, stride = 2, padding = 1)
        #self.conv5 = conv(1024,   2048, kernel_size = 5, stride = 2, padding = 2)
        #self.conv6 = conv(2048,   4096, kernel_size = 5, stride = 2, padding = 2)

        #self.deconv6 = deconv(8192, 2048, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv5 = deconv(6144, 1024, kernel_size = 5, stride = 2, padding = 2)
        self.deconv4 = deconv(128,  64, kernel_size = 4, stride = 2, padding = 1)
        self.deconv3 = deconv(192,  64, kernel_size = 4, stride = 2, padding = 1)
        self.deconv2 = deconv(192,  64, kernel_size = 4, stride = 2, padding = 1)
        self.deconv1 = deconv(192,  64, kernel_size = 4, stride = 2, padding = 1)

        self.tail1  = conv(192,   64, kernel_size=3, stride=1, padding=1)
        self.tail2  = conv(64,    64, kernel_size=3, stride=1, padding=1)
        #self.output = conv(64, output_nc, kernel_size=5, stride=1, padding=2, activation='linear')
        self.output = nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, input, flows, Backward_warper, reference):

        c1h = self.head(input)
        c10 = self.conv0(c1h)
        c11 = self.conv1(c10)
        c12 = self.conv2(c11)
        c13 = self.conv3(c12)
        c14 = self.conv4(c13)

        c2h = self.head(reference)
        c20 = self.conv0(c2h)
        c21 = self.conv1(c20)
        c22 = self.conv2(c21)
        c23 = self.conv3(c22)
        c24 = self.conv4(c23)

        warp_conv4   = Backward_warper(c24, flows[4]*1.25)
        x4 = torch.cat((c14, warp_conv4),1)
        warp_deconv4 = self.deconv4(x4)

        warp_conv3   = Backward_warper(c23, flows[3]*2.5)
        x3 = torch.cat((warp_deconv4, c13, warp_conv3),1)
        warp_deconv3 = self.deconv3(x3)

        warp_conv2   = Backward_warper(c22, flows[2]*5.0)
        x2 = torch.cat((warp_deconv3, c12, warp_conv2),1)
        warp_deconv2 = self.deconv2(x2)

        warp_conv1   = Backward_warper(c21, flows[1]*10.0)
        x1 = torch.cat((warp_deconv2, c11, warp_conv1),1)
        warp_deconv1 = self.deconv1(x1)

        warp_conv0   = Backward_warper(c20, flows[0]*20.0)
        x0 = torch.cat((warp_deconv1, c10, warp_conv0),1)
        tail1 = self.tail1(x0)
        tail2 = self.tail2(tail1)
        output = self.output(tail2)

        return output

class FusionFlowGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FusionFlowGenerator, self).__init__()

        #construct unet structure
        #self.head = conv(input_nc, 64, kernel_size = 5, stride = 1, padding = 2)
        #self.conv0 = conv(64,       64, kernel_size = 5, stride = 1, padding = 2)
        #self.conv1 = conv(64,      128, kernel_size = 5, stride = 2, padding = 2)
        #self.conv2 = conv(128,     256, kernel_size = 5, stride = 2, padding = 2)
        #self.conv3 = conv(256,     512, kernel_size = 5, stride = 2, padding = 2)
        #self.conv4 = conv(512,    1024, kernel_size = 5, stride = 2, padding = 2)
        #self.conv5 = conv(1024,   2048, kernel_size = 5, stride = 2, padding = 2)
        #self.conv6 = conv(2048,   4096, kernel_size = 5, stride = 2, padding = 2)

        self.conv1a  = conv(input_nc,   16, kernel_size=3, stride=2, padding=1)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1, padding=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1, padding=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2, padding=1)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1, padding=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1, padding=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2, padding=1)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1, padding=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1, padding=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2, padding=1)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1, padding=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1, padding=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1, padding=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1, padding=1)
        self.conv6a  = conv(128,196, kernel_size=3, stride=2, padding=1)
        self.conv6aa = conv(196,196, kernel_size=3, stride=1, padding=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1, padding=1)

        self.deconv6   = deconv(392, 128, kernel_size = 4, stride = 2, padding = 1)
        self.deconv60  = conv(128,128, kernel_size=3, stride=1, padding=1)
        self.deconv600 = conv(128,128, kernel_size=3, stride=1, padding=1)

        self.deconv5   = deconv(384,  96, kernel_size = 4, stride = 2, padding = 1)
        self.deconv50  = conv(96,96, kernel_size=3, stride=1, padding=1)
        self.deconv500 = conv(96,96, kernel_size=3, stride=1, padding=1)

        self.deconv4   = deconv(288,  64, kernel_size = 4, stride = 2, padding = 1)
        self.deconv40  = conv(64,64, kernel_size=3, stride=1, padding=1)
        self.deconv400 = conv(64,64, kernel_size=3, stride=1, padding=1)

        self.deconv3   = deconv(192,  32, kernel_size = 4, stride = 2, padding = 1)
        self.deconv30  = conv(32,32, kernel_size=3, stride=1, padding=1)
        self.deconv300 = conv(32,32, kernel_size=3, stride=1, padding=1)

        self.deconv2   = deconv(96,   16, kernel_size = 4, stride = 2, padding = 1)
        self.deconv20  = conv(16,16, kernel_size=3, stride=1, padding=1)
        self.deconv200 = conv(16,16, kernel_size=3, stride=1, padding=1)

        self.deconv1   = deconv(48,    3, kernel_size = 4, stride = 2, padding = 1)
        self.deconv10  = conv(3,3, kernel_size=3, stride=1, padding=1)
        self.deconv100 = conv(3,3, kernel_size=3, stride=1, padding=1)

        #self.tail  = conv(9,   9, kernel_size=5, stride=1, padding=2)
        self.output = nn.Conv2d(9, output_nc, kernel_size=5, stride=1, padding=2)

        #self.deconv6 = deconv(8192, 2048, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv5 = deconv(6144, 1024, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv4 = deconv(3072,  512, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv3 = deconv(1536,  256, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv2 = deconv(768,   128, kernel_size = 5, stride = 2, padding = 2)
        #self.deconv1 = deconv(384,    64, kernel_size = 5, stride = 2, padding = 2)
        #self.tail1  = conv(192,   64, kernel_size=5, stride=1, padding=2)
        #self.tail2  = conv(64,    64, kernel_size=5, stride=1, padding=2)
        #self.output = conv(64, output_nc, kernel_size=5, stride=1, padding=2, activation='linear')
        #self.output = nn.Conv2d(64, output_nc, kernel_size=5, stride=1, padding=2)

    def forward(self, input, flows, Backward_warper, reference):

        c11 = self.conv1b(self.conv1aa(self.conv1a(input)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c16 = self.conv6b(self.conv6aa(self.conv6a(c15)))

        c21 = self.conv1b(self.conv1aa(self.conv1a(reference)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c26 = self.conv6b(self.conv6aa(self.conv6a(c15)))

        warp_conv6   = Backward_warper(c26, flows[6]*0.3125)
        x6 = torch.cat((c16, warp_conv6),1)
        warp_deconv6 = self.deconv600(self.deconv60(self.deconv6(x6)))

        warp_conv5   = Backward_warper(c25, flows[5]*0.625)
        x5 = torch.cat((warp_deconv6, c15, warp_conv5),1)
        warp_deconv5 = self.deconv500(self.deconv50(self.deconv5(x5)))

        warp_conv4   = Backward_warper(c24, flows[4]*1.25)
        x4 = torch.cat((warp_deconv5, c14, warp_conv4),1)
        warp_deconv4 = self.deconv400(self.deconv40(self.deconv4(x4)))

        warp_conv3   = Backward_warper(c23, flows[3]*2.5)
        x3 = torch.cat((warp_deconv4, c13, warp_conv3),1)
        warp_deconv3 = self.deconv300(self.deconv30(self.deconv3(x3)))

        warp_conv2   = Backward_warper(c22, flows[2]*5)
        x2 = torch.cat((warp_deconv3, c12, warp_conv2),1)
        warp_deconv2 = self.deconv200(self.deconv20(self.deconv2(x2)))

        warp_conv1   = Backward_warper(c21, flows[1]*10)
        x1 = torch.cat((warp_deconv2, c11, warp_conv1),1)
        warp_deconv1 = self.deconv100(self.deconv10(self.deconv1(x1)))

        warp_conv0   = Backward_warper(reference, flows[0]*20)
        x0 = torch.cat((warp_deconv1, input, warp_conv0),1)
        output  = self.output(x0)

        return output







class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt, num_layer=3):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.num_layer = num_layer
        
        self.head = ConvBlock(opt.n_colors, 64, 3, 1, 1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(self.num_layer):
            block = ConvBlock(64, 64, 3, 1, 1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(64, opt.n_colors, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x





class GradualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer=1):
        super(GradualBlock, self).__init__()
        self.head = basenet.BasicBlock(in_channels=in_channels, out_channels=64)
        self.body = nn.Sequential()
        for i in range(num_layer):
            block = basenet.BasicBlock(64, 64, kernel_size=5, stride=1, padding=3, dilation=1)#+2
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class GradualGenerator(nn.Module):
    def __init__(self, n_blocks):
        super(GradualGenerator, self).__init__()
        self.n_blocks = n_blocks

        #layers = [basenet.BasicBlock(in_channels=3, out_channels=3)]
        layers = []
        for _ in range(self.n_blocks):
            layers.append(GradualBlock(6, 3))
        #layers.append(nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1),nn.Tanh()))
        self.upscale_layers = nn.ModuleList(layers)

    #def forward(self, image_list):
    #    x = self.upscale_layers[0](image_list[0])
    #    for i in range(1, self.n_blocks+1):
    #        x = self.upscale_layers[i]
    #    return x
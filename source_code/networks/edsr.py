import networks.basenet as basenet
import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            basenet.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        
        return x 





class EDLR(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(EDLR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            basenet.Downsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        
        return x




class Feature_extractor(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(Feature_extractor, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        # define body module
        m_body = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        res = self.body(x)
        return res

class Feature_Head(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(Feature_Head, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3 
        
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*m_head)

    def forward(self, x):
        x = self.head(x)
        return x

class Upscalar(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(Upscalar, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)

        m_upscalar = [
            basenet.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.upscalar = nn.Sequential(*m_upscalar)

    def forward(self, x):
        y = self.upscalar(x)
        return y

class Downscalar(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(Downscalar, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)

        m_downscalar = [
            basenet.Downsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.downsclar = nn.Sequential(*m_downscalar)

    def forward(self, x):
        y = self.downsclar(x)
        return y
import networks.basenet as basenet
import torch
import torch.nn as nn

class EEDSR(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(EEDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body_1 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_1.append(conv(n_feats, n_feats, kernel_size))

        m_body_2 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_2.append(conv(n_feats, n_feats, kernel_size))

        m_body_3 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_3.append(conv(n_feats, n_feats, kernel_size))

        m_body_4 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_4.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            basenet.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)

        self.body1 = nn.Sequential(*m_body_1)
        self.gate1 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        self.conv_1 = basenet.BasicBlock(in_channels=n_feats*2, out_channels=n_feats)
        
        self.body2 = nn.Sequential(*m_body_2)
        self.gate2 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        self.conv_2 = basenet.BasicBlock(in_channels=n_feats*2, out_channels=n_feats)
        
        self.body3 = nn.Sequential(*m_body_3)
        self.gate3 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        self.conv_3 = basenet.BasicBlock(in_channels=n_feats*2, out_channels=n_feats)
        
        self.body4 = nn.Sequential(*m_body_4)
        self.gate4 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        self.conv_4 = basenet.BasicBlock(in_channels=n_feats*2, out_channels=n_feats)

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x1_out = self.body1(x) + x
        x2_out = self.body2(x1_out) + x1_out
        x3_out = self.body3(x2_out) + x2_out
        x4_out = self.body4(x3_out) + x3_out

        x4_gated_out = self.gate4(x3_out, x4_out)
        x4_concat = torch.cat((x4_gated_out, x3_out), 1)
        x4_concat_out = self.conv_4(x4_concat)

        x3_gated_out = self.gate3(x2_out, x4_concat_out)
        x3_concat = torch.cat((x3_gated_out, x2_out), 1)
        x3_concat_out = self.conv_3(x3_concat)

        x2_gated_out = self.gate2(x1_out, x3_concat_out)
        x2_concat = torch.cat((x2_gated_out, x1_out), 1)
        x2_concat_out = self.conv_2(x2_concat)

        x1_gated_out = self.gate1(x, x2_concat_out)
        x1_concat = torch.cat((x1_gated_out, x), 1)
        x_feature = self.conv_1(x1_concat)

        #res = self.body(x)
        #res += x
        #x = self.tail(res)
        x_out = self.tail(x_feature)
        
        return x_out

class EEDSR2(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(EEDSR2, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body_1 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_1.append(conv(n_feats, n_feats, kernel_size))

        m_body_2 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_2.append(conv(n_feats, n_feats, kernel_size))

        m_body_3 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_3.append(conv(n_feats, n_feats, kernel_size))

        m_body_4 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_4.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            basenet.Upsampler(conv, scale, n_feats*5, act=False),
            conv(n_feats*5, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)

        self.body1 = nn.Sequential(*m_body_1)
        self.gate1 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        
        self.body2 = nn.Sequential(*m_body_2)
        self.gate2 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        
        self.body3 = nn.Sequential(*m_body_3)
        self.gate3 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)
        
        self.body4 = nn.Sequential(*m_body_4)
        self.gate4 = Gate(F_g=n_feats, F_l=n_feats, F_int=n_feats)

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x1_out = self.body1(x) + x
        x2_out = self.body2(x1_out) + x1_out
        x3_out = self.body3(x2_out) + x2_out
        x4_out = self.body4(x3_out) + x3_out

        x4_gated_out = self.gate4(x3_out, x4_out)
        x3_gated_out = self.gate3(x2_out, x3_out)
        x2_gated_out = self.gate2(x1_out, x2_out)
        x1_gated_out = self.gate1(x, x1_out)

        x_concat = torch.cat((x, x1_gated_out, x2_gated_out, x3_gated_out, x4_gated_out), 1)
        x_out = self.tail(x_concat)
        
        return x_out

class EEDSR3(nn.Module):
    def __init__(self, args, conv=basenet.default_conv):
        super(EEDSR3, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body_1 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_1.append(conv(n_feats, n_feats, kernel_size))

        m_body_2 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_2.append(conv(n_feats, n_feats, kernel_size))

        m_body_3 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_3.append(conv(n_feats, n_feats, kernel_size))

        m_body_4 = [
            basenet.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body_4.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            basenet.Upsampler(conv, scale, n_feats*5, act=False),
            conv(n_feats*5, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)

        self.body1 = nn.Sequential(*m_body_1)
        self.body2 = nn.Sequential(*m_body_2)
        self.body3 = nn.Sequential(*m_body_3)
        self.body4 = nn.Sequential(*m_body_4)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x1_out = self.body1(x) + x
        x2_out = self.body2(x1_out) + x1_out
        x3_out = self.body3(x2_out) + x2_out
        x4_out = self.body4(x3_out) + x3_out

        #x4_gated_out = self.gate4(x3_out, x4_out)
        #x3_gated_out = self.gate3(x2_out, x3_out)
        #x2_gated_out = self.gate2(x1_out, x2_out)
        #x1_gated_out = self.gate1(x, x1_out)

        #x_concat = torch.cat((x1_gated_out, x2_gated_out, x3_gated_out, x4_gated_out, x), 1)
        x_concat = torch.cat((x, x1_out, x2_out, x3_out, x4_out), 1)
        x_out = self.tail(x_concat)
        
        return x_out

class Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
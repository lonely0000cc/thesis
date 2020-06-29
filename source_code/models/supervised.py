import torch
import torch.nn as nn
import os

from models.basemodel import BaseModel
from models.losses import *

from networks.edsr import *
from networks.eedsr import *
from networks.drln import DRLN
from networks.rcan import RCAN
from networks.flow import PWCDCNet, UpPWCDCNet, Backward_warp
from networks.vgg import VGG19
from networks.basenet import Encoder, Decoder
from networks.critic import *
from networks.generator import *

from utils.metrics import PSNR, SSIM
from utils.functions import generate_2D_grid, grid_gradient_central_diff
from Blurrer_layer import FlowWarpMask

class CircleFlowSRNet(BaseModel):
    def __init__(self, opts):
        super(CircleFlowSRNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'enc', 'dec']
        self.net_sr   = EDSR(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D    = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        self.net_enc  = Encoder().cuda()
        self.net_dec  = Decoder().cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D)
        self.print_networks(self.net_enc)
        self.print_networks(self.net_dec)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size, 2, im_crop_H, im_crop_W]

        print('grid size', grid.size())
        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        #self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training: 
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_sr.parameters()},
                {'params': self.net_enc.parameters()},
                {'params': self.net_dec.parameters()},
                {'params': self.net_flow.parameters()}], lr=opts.lr)
            
            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr/10.0)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self',            10))
                f.write('{} : {}\n'.format('loss_self_deblur',     10))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       10))

                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.5))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None


    def forward(self):

        self.sr_img_ref         = self.net_sr(self.lr_img_ref)
        self.flows_ref_to_other = self.net_flow(self.hr_img_ref_gt, self.hr_img_oth_gt)
        self.flows_other_to_ref = self.net_flow(self.hr_img_oth_gt, self.hr_img_ref_gt)

        flow_12_1 = self.flows_ref_to_other[0]*20.0
        flow_12_2 = self.flows_ref_to_other[1]*10.0
        flow_12_3 = self.flows_ref_to_other[2]*5.0
        flow_12_4 = self.flows_ref_to_other[3]*2.5

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.net_enc(self.sr_img_ref)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.net_enc(self.hr_img_oth_gt)

        warp_21_conv1 = self.Backward_warper(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warper(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warper(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warper(HR2_conv4, flow_12_4)

        sythsis_output = self.net_dec(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        return sythsis_output

    def optimize_G(self):

        self.sythsis_output = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)

        # compute synthetic hr images
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth_gt, self.flows_ref_to_other[0]*20.0)

        # compute self consistency losses
        self.loss_self = self.loss_L1(self.sythsis_output, self.hr_img_ref_gt, mean=True) * 10

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True)
        
        # compute perceptual loss
        self.loss_perceptural = self.loss_content.get_loss(self.sythsis_output, self.hr_img_ref_gt) * 0.02

        # compute frequency loss
        self.loss_frequency   = self.loss_Fre(self.sythsis_output, self.hr_img_ref_gt)

        # compute texture loss
        sr_prevlayer, _ = self.vgg(self.sythsis_output)
        hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-5

        # compute GAN loss
        dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.sythsis_output)
        self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
                       
        self.loss_G =  self.loss_self \
                    + self.loss_perceptural \
                    + self.loss_frequency \
                    + self.loss_texture_matching \
                    + self.loss_G_D
                    #self.loss_flow \

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_D(self):

        dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, hr_img_real):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        self.hr_img_real = hr_img_real

    def optimize(self):
        self.optimize_G()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',      label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',       label, self.opts.checkpoint_dir)
        self.save_network(self.net_enc,  'Encoder', label, self.opts.checkpoint_dir)
        self.save_network(self.net_dec,  'Decoder', label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',      label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',       label, self.opts.checkpoint_dir)
        self.load_network(self.net_enc,  'Encoder', label, self.opts.checkpoint_dir)
        self.load_network(self.net_dec,  'Decoder', label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_flow'] = self.loss_flow.item()
        losses['loss_self'] = self.loss_self.item()
        losses['loss_perceptural'] = self.loss_perceptural.item()
        losses['loss_frequency'] = self.loss_frequency.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()        
        losses['loss_D'] = self.loss_D.item()
        losses['loss_G'] = self.loss_G.item()

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.sythsis_output.data, self.hr_img_ref_gt)
        return losses




#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class FusionFlowSRNet(BaseModel):
    def __init__(self, opts):
        super(FusionFlowSRNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'enc', 'dec']
        self.net_sr   = EDSR(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D    = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        #self.net_G1   = FusionFlowGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors).cuda()
        self.net_G1   = FeatureFusionGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors).cuda()
        #self.net_G1   = CrossFusionGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors).cuda()
        #self.net_enc  = Encoder().cuda()
        #self.net_dec  = Decoder().cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D)
        self.print_networks(self.net_G1)
        #self.print_networks(self.net_enc)
        #self.print_networks(self.net_dec)

        #grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        #grid = grid.int().cuda().unsqueeze(0)
        #grid = grid.repeat(opts.batch_size-1, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]
        #grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]

        #print('grid size', grid.size())
        #self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        #self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)
            #self.load_network(self.net_sr, 'SR', opts.epoch_to_load, self.opts.model_dir)

        if opts.is_training: 
            self.optimizer_G = torch.optim.Adam([
                #{'params': self.net_sr.parameters()},
                {'params': self.net_G1.parameters()},
                {'params': self.net_flow.parameters()}], lr=opts.lr)
            
            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr/10.0)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self',            10))
                f.write('{} : {}\n'.format('loss_self_deblur',     10))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       10))

                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.5))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None


    def forward(self):

        self.sr_img_ref = self.net_sr(self.lr_img_ref)
        self.sr_img_oth = self.net_sr(self.lr_img_oth)

        #self.flows_ref_to_other = self.net_flow(self.sr_img_ref, self.sr_img_oth)
        self.flows_ref_to_other = self.net_flow(self.sr_img_ref, self.sr_img_oth)
        #self.flows_other_to_ref = self.net_flow(self.hr_img_oth_gt, self.hr_img_ref_gt)

        #flow_12_1 = self.flows_ref_to_other[0]*20.0
        #flow_12_2 = self.flows_ref_to_other[1]*10.0
        #flow_12_3 = self.flows_ref_to_other[2]*5.0
        #flow_12_4 = self.flows_ref_to_other[3]*2.5

        #SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.net_enc(self.sr_img_ref)
        #HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.net_enc(self.hr_img_oth_gt)

        #warp_21_conv1 = self.Backward_warper(HR2_conv1, flow_12_1)
        #warp_21_conv2 = self.Backward_warper(HR2_conv2, flow_12_2)
        #warp_21_conv3 = self.Backward_warper(HR2_conv3, flow_12_3)
        #warp_21_conv4 = self.Backward_warper(HR2_conv4, flow_12_4)

        #sythsis_output = self.net_dec(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        sythsis_output = self.net_G1(self.sr_img_ref, self.flows_ref_to_other, self.Backward_warper, self.sr_img_oth)

        return sythsis_output

    def optimize_G(self):

        self.sythsis_output = self.forward()

        # compute hr mask & lr mask
        #self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)

        # compute synthetic hr images
        #self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth_gt, self.flows_ref_to_other[0]*20.0)

        # compute self consistency losses
        self.loss_self = self.loss_L1(self.sythsis_output, self.hr_img_ref_gt, mean=True) * 10
        #+ self.loss_L1(self.sr_img_ref,     self.hr_img_ref_gt, mean=True) \
        #+ self.loss_L1(self.sr_img_oth,     self.hr_img_oth_gt, mean=True)

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True)
        
        # compute perceptual loss
        self.loss_perceptural = self.loss_content.get_loss(self.sythsis_output, self.hr_img_ref_gt) * 0.05
        #+ self.loss_content.get_loss(self.sr_img_ref,     self.hr_img_ref_gt) * 0.01 \
        #+ self.loss_content.get_loss(self.sr_img_oth,     self.hr_img_oth_gt) * 0.01


        # compute frequency loss
        self.loss_frequency   = self.loss_Fre(self.sythsis_output, self.hr_img_ref_gt) * 10
        #+ self.loss_Fre(self.sr_img_ref,     self.hr_img_ref_gt) \
        #+ self.loss_Fre(self.sr_img_oth,     self.hr_img_oth_gt)

        # compute texture loss
        sr_prevlayer, _ = self.vgg(self.sythsis_output)
        hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-3

        # compute GAN loss
        dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.sythsis_output)
        self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
                       
        self.loss_G =  self.loss_self \
                    + self.loss_perceptural \
                    + self.loss_frequency \
                    + self.loss_texture_matching \
                    + self.loss_G_D
                    #self.loss_flow \

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_D(self):

        dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, hr_img_real):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        self.hr_img_real = hr_img_real

    def optimize(self):
        self.optimize_G()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',      label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',       label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',       label, self.opts.checkpoint_dir)
        #self.save_network(self.net_enc,  'Encoder', label, self.opts.checkpoint_dir)
        #self.save_network(self.net_dec,  'Decoder', label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',      label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',       label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',       label, self.opts.checkpoint_dir)
        #self.load_network(self.net_enc,  'Encoder', label, self.opts.checkpoint_dir)
        #self.load_network(self.net_dec,  'Decoder', label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_flow'] = self.loss_flow.item()
        losses['loss_self'] = self.loss_self.item()
        losses['loss_perceptural'] = self.loss_perceptural.item()
        losses['loss_frequency'] = self.loss_frequency.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()        
        losses['loss_D'] = self.loss_D.item()
        losses['loss_G'] = self.loss_G.item()

        if self.hr_img_ref_gt is not None:
            losses['PSNR_sythesis'] = PSNR(self.sythsis_output.data, self.hr_img_ref_gt)
            losses['PSNR_sr']       = PSNR(self.sr_img_ref.data, self.hr_img_ref_gt)
        return losses









#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


class FlowSRNet(BaseModel):
    def __init__(self, opts):
        super(FlowSRNet, self).__init__()
        self.opts = opts
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'G1', 'G2']
        self.net_sr   = EDSR(opts).cuda()
        #self.net_sr   = DRLN(opts).cuda()
        #self.net_sr   = RCAN(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D1   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_D2   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        #self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()
        #self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=7).cuda()
        self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        #self.net_G2   = EDLR(opts).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D1)
        self.print_networks(self.net_D2)
        self.print_networks(self.net_G1)
        self.print_networks(self.net_G2)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size, 2, im_crop_H, im_crop_W]

        print('grid size', grid.size())
        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        #self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                #self.load_network(self.net_sr,   'SR', opts.epoch_to_load, self.opts.model_dir)
                self.load_network(self.net_flow, 'Flow', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            self.optimizer_G_sr = torch.optim.Adam([
                {'params': self.net_sr.parameters()}], lr=opts.lr)

            self.optimizer_G_style = torch.optim.Adam([
                {'params': self.net_G1.parameters()},
                {'params': self.net_G2.parameters()}], lr=opts.lr)
            
            self.optimizer_D1 = torch.optim.Adam([
                {'params': self.net_D1.parameters()}], lr=opts.lr)

            self.optimizer_D2 = torch.optim.Adam([
                {'params': self.net_D2.parameters()}], lr=opts.lr)

            self.scheduler_G_sr    = torch.optim.lr_scheduler.StepLR(self.optimizer_G_sr,    step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_G_style = torch.optim.lr_scheduler.StepLR(self.optimizer_G_style, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D1 = torch.optim.lr_scheduler.StepLR(self.optimizer_D1, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D2 = torch.optim.lr_scheduler.StepLR(self.optimizer_D2, step_size=opts.lr_step, gamma=opts.lr_decay)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            #self.loss_tv_flow = VariationLoss(nc=2, grad_fn=grid_gradient_central_diff)
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self',            10))
                f.write('{} : {}\n'.format('loss_self_deblur',     10))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       10))

                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.5))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        #self.noise_amp = 0.001
        #self.using_noise = False

    def forward(self):

        hr_img_ref = self.net_sr(self.lr_img_ref)
        hr_img_oth = self.net_sr(self.lr_img_oth)

        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.hr_img_ref.shape[2], self.hr_img_ref.shape[3]).cuda() * self.noise_amp
        #res_ref = self.net_G1(self.hr_img_ref + noise)
        #res_oth = self.net_G1(self.hr_img_oth + noise)
        #deblur_hr_img_ref = self.net_G1(self.hr_img_ref)
        #deblur_hr_img_oth = self.net_G1(self.hr_img_oth)

        #deblur_hr_img_ref = self.hr_img_ref + res_ref
        #deblur_hr_img_oth = self.hr_img_oth + res_oth
        #deblur_hr_img_ref = self.net_G1(self.hr_img_ref)

        #self.cyc_img_ref = self.net_G2(deblur_hr_img_ref)
        #self.cyc_img_oth = self.net_G2(deblur_hr_img_oth)

        self.flows_ref_to_other = self.net_flow(hr_img_ref, hr_img_oth)
        self.flows_other_to_ref = self.net_flow(hr_img_oth, hr_img_ref)

        #flow_ref_to_other = self.upsample_4(flows_ref_to_other[0])*20.0
        #flow_other_to_ref = self.upsample_4(flows_other_to_ref[0])*20.0 

        #return deblur_hr_img_ref, deblur_hr_img_oth
        return hr_img_ref, hr_img_oth

    def optimize_G_sr(self):

        #self.lr_img_ref     = lr_img_ref
        #self.hr_img_others  = hr_img_others        
        #self.hr_img_ref, \
        #self.deblur_hr_img_ref, \
        #self.cyc_img_ref, \
        #self.flow_ref_to_other, \
        #self.flow_other_to_ref = self.forward(self.lr_img_ref, self.hr_img_others)
        #self.deblur_hr_img_ref, self.deblur_hr_img_oth = self.forward()
        self.hr_img_ref, self.hr_img_oth = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)
        self.hr_mask_oth = self.mask_fn(self.flows_ref_to_other[0]*20.0)

        # compute synthetic hr images
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_others, self.flow_ref_to_other)
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_others, self.flow_ref_to_other)
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth, self.flows_ref_to_other[0]*20.0)
        self.syn_hr_img_oth = self.Backward_warper(self.hr_img_ref, self.flows_other_to_ref[0]*20.0)

        # compute self consistency losses
        self.loss_self_sr = self.loss_L1(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
                            self.loss_L1(self.hr_img_oth, self.hr_img_oth_gt, mean=True) * 10
        #self.loss_L1(self.deblur_hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
        #self.loss_L1(self.deblur_hr_img_oth, self.hr_img_oth_gt, mean=True) * 10
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True)
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 10 + \
        #                self.loss_L2(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 10
        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01
        #self.loss_cyc_xy = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True) + \
        #                   self.loss_L2(self.cyc_img_oth,    self.hr_img_oth, mean=True)
        #self.loss_cyc_yx = self.loss_L2(self.net_G1(self.net_G2(self.img_real_cen)), self.img_real_cen, mean=True) + \
        #                   self.loss_L2(self.net_G1(self.net_G2(self.img_real_ref)), self.img_real_ref, mean=True)

        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01 + \
        #                     self.loss_L1(self.deblur_hr_img_oth, self.net_G1(self.deblur_hr_img_oth), mean=True) * 0.01

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref, self.hr_mask_ref, mean=True)
        self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True) * 0.5 + \
                         self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth_gt, self.hr_mask_oth, mean=True) * 0.5
        #self.loss_tv_flow(self.flow_ref_to_other, mean=True) * 0.01 + \
        #self.loss_tv_flow(self.flow_other_to_ref, mean=True) * 0.01

        # compute perceptual loss
        self.loss_perceptural_sr = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
                                   self.loss_content.get_loss(self.hr_img_oth, self.hr_img_oth_gt) * 0.01
        #self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #self.loss_content.get_loss(self.deblur_hr_img_oth, self.hr_img_oth_gt) * 0.01

        # compute frequency loss
        self.loss_frequency_sr   = self.loss_Fre(self.hr_img_ref, self.hr_img_ref_gt) + \
                                self.loss_Fre(self.hr_img_oth, self.hr_img_oth_gt)
        #self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) + \
        #self.loss_Fre(self.deblur_hr_img_oth, self.hr_img_oth_gt)

        # compute texture loss
        sr_prevlayer_ref, _ = self.vgg(self.hr_img_ref)
        sr_prevlayer_oth, _ = self.vgg(self.hr_img_oth)
        #dr_prevlayer_ref, _ = self.vgg(self.deblur_hr_img_ref)
        #dr_prevlayer_oth, _ = self.vgg(self.deblur_hr_img_oth)
        hr_prevlayer_ref, _ = self.vgg(self.hr_img_ref_gt)
        hr_prevlayer_oth, _ = self.vgg(self.hr_img_oth_gt)

        self.loss_texture_matching_sr = self.loss_texture(hr_prevlayer_ref, sr_prevlayer_ref) * 1e-4 + \
                                        self.loss_texture(hr_prevlayer_oth, sr_prevlayer_oth) * 1e-4
        #self.loss_texture(hr_prevlayer_ref, dr_prevlayer_ref) * 1e-4 + \
        #self.loss_texture(hr_prevlayer_oth, dr_prevlayer_oth) * 1e-4

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(torch.cat([self.deblur_hr_img_ref, self.deblur_hr_img_oth], dim=0))
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        #self.loss_G_D1 = -self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean()
        #self.loss_G_D2 = -self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean()

        self.loss_G_sr =  self.loss_flow \
                        + self.loss_self_sr \
                        + self.loss_perceptural_sr \
                        + self.loss_frequency_sr \
                        + self.loss_texture_matching_sr

        self.optimizer_G_sr.zero_grad()
        self.loss_G_sr.backward()
        #self.gradient_clip(['sr', 'flow', 'G1', 'G2'])
        self.optimizer_G_sr.step()

    def optimize_G_style(self):
        self.deblur_hr_img_ref = self.net_G1(self.hr_img_ref.detach())
        self.deblur_hr_img_oth = self.net_G1(self.hr_img_oth.detach())
        self.cyc_img_ref = self.net_G2(self.deblur_hr_img_ref)
        self.cyc_img_oth = self.net_G2(self.deblur_hr_img_oth)

        # compute self consistency losses
        self.loss_self_style = self.loss_L1(self.deblur_hr_img_ref, self.hr_img_ref_gt, mean=True) * 100 + \
                               self.loss_L1(self.deblur_hr_img_oth, self.hr_img_oth_gt, mean=True) * 100
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True)
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 10 + \
        #                self.loss_L2(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 10
        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01
        self.loss_cyc_xy = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref.detach(), mean=True) + \
                           self.loss_L2(self.cyc_img_oth,    self.hr_img_oth.detach(), mean=True)
        self.loss_cyc_yx = self.loss_L2(self.net_G1(self.net_G2(self.img_real_cen)), self.img_real_cen, mean=True) + \
                           self.loss_L2(self.net_G1(self.net_G2(self.img_real_ref)), self.img_real_ref, mean=True)

        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01 + \
        #                     self.loss_L1(self.deblur_hr_img_oth, self.net_G1(self.deblur_hr_img_oth), mean=True) * 0.01

        # compute perceptual loss
        #self.loss_perceptural_style = self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #                              self.loss_content.get_loss(self.deblur_hr_img_oth, self.hr_img_oth_gt) * 0.01

        # compute frequency loss
        #self.loss_frequency_style = self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) + \
        #                            self.loss_Fre(self.deblur_hr_img_oth, self.hr_img_oth_gt)

        # compute texture loss
        #dr_prevlayer_ref, _ = self.vgg(self.deblur_hr_img_ref)
        #dr_prevlayer_oth, _ = self.vgg(self.deblur_hr_img_oth)
        #hr_prevlayer_ref, _ = self.vgg(self.hr_img_ref_gt)
        #hr_prevlayer_oth, _ = self.vgg(self.hr_img_oth_gt)

        #self.loss_texture_matching_style = self.loss_texture(hr_prevlayer_ref, dr_prevlayer_ref) * 1e-4 + \
        #                                   self.loss_texture(hr_prevlayer_oth, dr_prevlayer_oth) * 1e-4

        # compute GAN loss
        self.loss_G_D1 = (-self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean())
        self.loss_G_D2 = (-self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean())

        self.loss_G_style = self.loss_self_style \
                    + self.loss_cyc_xy \
                    + self.loss_cyc_yx \
                    + self.loss_G_D1 \
                    + self.loss_G_D2

        self.optimizer_G_style.zero_grad()
        self.loss_G_style.backward()
        self.optimizer_G_style.step()

    def optimize_D(self):
        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(torch.cat([self.hr_img_ref_gt, self.hr_img_oth_gt, self.hr_img_real], dim=0))
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        for i in range(self.opts.d_step):
            self.optimizer_D1.zero_grad()
            self.loss_D1 = self.loss_GAN(self.net_D1, self.img_real_cen, self.deblur_hr_img_ref, use_gp=True) + \
                           self.loss_GAN(self.net_D1, self.img_real_ref, self.deblur_hr_img_oth, use_gp=True)
            self.loss_D1.backward()
            self.optimizer_D1.step()

            self.optimizer_D2.zero_grad()
            self.loss_D2 = self.loss_GAN(self.net_D2, self.hr_img_ref.detach(), self.net_G2(self.img_real_cen), use_gp=True) + \
                           self.loss_GAN(self.net_D2, self.hr_img_oth.detach(), self.net_G2(self.img_real_ref), use_gp=True)
            self.loss_D2.backward()
            self.optimizer_D2.step()

        #self.optimizer_D.zero_grad()
        #self.loss_D.backward()
        #self.gradient_clip(['D'])
        #self.optimizer_D.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, img_real_cen, img_real_ref):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        self.img_real_cen = img_real_cen
        self.img_real_ref = img_real_ref

    def optimize(self):
        self.optimize_G_sr()
        self.optimize_G_style()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G_sr.step()
        self.scheduler_G_style.step()
        self.scheduler_D1.step()
        self.scheduler_D2.step()

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.save_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.load_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_flow'] = self.loss_flow.item()
        losses['loss_self_sr'] = self.loss_self_sr.item()
        #losses['loss_identity'] = self.loss_identity.item()
        losses['loss_perceptural_sr'] = self.loss_perceptural_sr.item()
        losses['loss_frequency_sr'] = self.loss_frequency_sr.item()
        losses['loss_texture_matching_sr'] = self.loss_texture_matching_sr.item()
        losses['loss_G_sr'] = self.loss_G_sr.item()

        losses['loss_self_style'] = self.loss_self_style.item()
        losses['loss_cyc_xy'] = self.loss_cyc_xy.item()
        losses['loss_cyc_yx'] = self.loss_cyc_yx.item()
        losses['loss_G_D1'] = self.loss_G_D1.item()
        losses['loss_G_D2'] = self.loss_G_D2.item()
        losses['loss_G_style'] = self.loss_G_style.item()

        losses['loss_D1'] = self.loss_D1.item()
        losses['loss_D2'] = self.loss_D2.item()
        

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
            losses['PSNR_deblur'] = PSNR(self.deblur_hr_img_ref.data, self.hr_img_ref_gt)
        return losses




#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


class FlowSRGNet(BaseModel):
    def __init__(self, opts):
        super(FlowSRGNet, self).__init__()
        self.opts = opts
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'G1', 'G2']
        self.net_sr   = EDSR(opts).cuda()
        #self.net_sr   = DRLN(opts).cuda()
        #self.net_sr   = RCAN(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D1   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_D2   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        #self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()
        #self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=7).cuda()
        self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        #self.net_G2   = EDLR(opts).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D1)
        self.print_networks(self.net_D2)
        self.print_networks(self.net_G1)
        self.print_networks(self.net_G2)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size, 2, im_crop_H, im_crop_W]

        print('grid size', grid.size())
        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        #self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                #self.load_network(self.net_sr,   'SR', opts.epoch_to_load, self.opts.model_dir)
                self.load_network(self.net_flow, 'Flow', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            self.optimizer_G_sr = torch.optim.Adam([
                {'params': self.net_sr.parameters()}], lr=opts.lr)

            self.optimizer_G_style = torch.optim.Adam([
                {'params': self.net_G1.parameters()},
                {'params': self.net_G2.parameters()}], lr=opts.lr)
            
            self.optimizer_D1 = torch.optim.Adam([
                {'params': self.net_D1.parameters()}], lr=opts.lr)

            self.optimizer_D2 = torch.optim.Adam([
                {'params': self.net_D2.parameters()}], lr=opts.lr)

            self.scheduler_G_sr    = torch.optim.lr_scheduler.StepLR(self.optimizer_G_sr,    step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_G_style = torch.optim.lr_scheduler.StepLR(self.optimizer_G_style, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D1 = torch.optim.lr_scheduler.StepLR(self.optimizer_D1, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D2 = torch.optim.lr_scheduler.StepLR(self.optimizer_D2, step_size=opts.lr_step, gamma=opts.lr_decay)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            #self.loss_tv_flow = VariationLoss(nc=2, grad_fn=grid_gradient_central_diff)
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self',            10))
                f.write('{} : {}\n'.format('loss_self_deblur',     10))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       10))

                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.5))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        #self.noise_amp = 0.001
        #self.using_noise = False

    def forward(self):

        hr_img_ref = self.net_sr(self.lr_img_ref)
        hr_img_oth = self.net_sr(self.lr_img_oth)

        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.hr_img_ref.shape[2], self.hr_img_ref.shape[3]).cuda() * self.noise_amp
        #res_ref = self.net_G1(self.hr_img_ref + noise)
        #res_oth = self.net_G1(self.hr_img_oth + noise)
        #deblur_hr_img_ref = self.net_G1(self.hr_img_ref)
        #deblur_hr_img_oth = self.net_G1(self.hr_img_oth)

        #deblur_hr_img_ref = self.hr_img_ref + res_ref
        #deblur_hr_img_oth = self.hr_img_oth + res_oth
        #deblur_hr_img_ref = self.net_G1(self.hr_img_ref)

        #self.cyc_img_ref = self.net_G2(deblur_hr_img_ref)
        #self.cyc_img_oth = self.net_G2(deblur_hr_img_oth)

        self.flows_ref_to_other = self.net_flow(hr_img_ref, hr_img_oth)
        self.flows_other_to_ref = self.net_flow(hr_img_oth, hr_img_ref)

        #flow_ref_to_other = self.upsample_4(flows_ref_to_other[0])*20.0
        #flow_other_to_ref = self.upsample_4(flows_other_to_ref[0])*20.0 

        #return deblur_hr_img_ref, deblur_hr_img_oth
        return hr_img_ref, hr_img_oth

    def optimize_G_sr(self):

        #self.lr_img_ref     = lr_img_ref
        #self.hr_img_others  = hr_img_others        
        #self.hr_img_ref, \
        #self.deblur_hr_img_ref, \
        #self.cyc_img_ref, \
        #self.flow_ref_to_other, \
        #self.flow_other_to_ref = self.forward(self.lr_img_ref, self.hr_img_others)
        #self.deblur_hr_img_ref, self.deblur_hr_img_oth = self.forward()
        self.hr_img_ref, self.hr_img_oth = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)
        self.hr_mask_oth = self.mask_fn(self.flows_ref_to_other[0]*20.0)

        # compute synthetic hr images
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_others, self.flow_ref_to_other)
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_others, self.flow_ref_to_other)
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth, self.flows_ref_to_other[0]*20.0)
        self.syn_hr_img_oth = self.Backward_warper(self.hr_img_ref, self.flows_other_to_ref[0]*20.0)

        # compute self consistency losses
        self.loss_self_sr = self.loss_L1(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
                            self.loss_L1(self.hr_img_oth, self.hr_img_oth_gt, mean=True) * 10
        #self.loss_L1(self.deblur_hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
        #self.loss_L1(self.deblur_hr_img_oth, self.hr_img_oth_gt, mean=True) * 10
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True)
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 10 + \
        #                self.loss_L2(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 10
        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01
        #self.loss_cyc_xy = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True) + \
        #                   self.loss_L2(self.cyc_img_oth,    self.hr_img_oth, mean=True)
        #self.loss_cyc_yx = self.loss_L2(self.net_G1(self.net_G2(self.img_real_cen)), self.img_real_cen, mean=True) + \
        #                   self.loss_L2(self.net_G1(self.net_G2(self.img_real_ref)), self.img_real_ref, mean=True)

        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01 + \
        #                     self.loss_L1(self.deblur_hr_img_oth, self.net_G1(self.deblur_hr_img_oth), mean=True) * 0.01

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref, self.hr_mask_ref, mean=True)
        self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True) * 0.5 + \
                         self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth_gt, self.hr_mask_oth, mean=True) * 0.5
        #self.loss_tv_flow(self.flow_ref_to_other, mean=True) * 0.01 + \
        #self.loss_tv_flow(self.flow_other_to_ref, mean=True) * 0.01

        # compute perceptual loss
        self.loss_perceptural_sr = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
                                   self.loss_content.get_loss(self.hr_img_oth, self.hr_img_oth_gt) * 0.01
        #self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #self.loss_content.get_loss(self.deblur_hr_img_oth, self.hr_img_oth_gt) * 0.01

        # compute frequency loss
        self.loss_frequency_sr   = self.loss_Fre(self.hr_img_ref, self.hr_img_ref_gt) + \
                                self.loss_Fre(self.hr_img_oth, self.hr_img_oth_gt)
        #self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) + \
        #self.loss_Fre(self.deblur_hr_img_oth, self.hr_img_oth_gt)

        # compute texture loss
        sr_prevlayer_ref, _ = self.vgg(self.hr_img_ref)
        sr_prevlayer_oth, _ = self.vgg(self.hr_img_oth)
        #dr_prevlayer_ref, _ = self.vgg(self.deblur_hr_img_ref)
        #dr_prevlayer_oth, _ = self.vgg(self.deblur_hr_img_oth)
        hr_prevlayer_ref, _ = self.vgg(self.hr_img_ref_gt)
        hr_prevlayer_oth, _ = self.vgg(self.hr_img_oth_gt)

        self.loss_texture_matching_sr = self.loss_texture(hr_prevlayer_ref, sr_prevlayer_ref) * 1e-4 + \
                                        self.loss_texture(hr_prevlayer_oth, sr_prevlayer_oth) * 1e-4
        #self.loss_texture(hr_prevlayer_ref, dr_prevlayer_ref) * 1e-4 + \
        #self.loss_texture(hr_prevlayer_oth, dr_prevlayer_oth) * 1e-4

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(torch.cat([self.deblur_hr_img_ref, self.deblur_hr_img_oth], dim=0))
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        #self.loss_G_D1 = -self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean()
        #self.loss_G_D2 = -self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean()

        self.loss_G_sr =  self.loss_flow \
                        + self.loss_self_sr \
                        + self.loss_perceptural_sr \
                        + self.loss_frequency_sr \
                        + self.loss_texture_matching_sr

        self.optimizer_G_sr.zero_grad()
        self.loss_G_sr.backward()
        #self.gradient_clip(['sr', 'flow', 'G1', 'G2'])
        self.optimizer_G_sr.step()

    def optimize_G_style(self):
        self.deblur_hr_img_ref = self.net_G1(self.hr_img_ref.detach())
        self.deblur_hr_img_oth = self.net_G1(self.hr_img_oth.detach())
        self.cyc_img_ref = self.net_G2(self.deblur_hr_img_ref)
        self.cyc_img_oth = self.net_G2(self.deblur_hr_img_oth)

        # compute self consistency losses
        self.loss_self_style = self.loss_L1(self.deblur_hr_img_ref, self.hr_img_ref_gt, mean=True) * 100 + \
                               self.loss_L1(self.deblur_hr_img_oth, self.hr_img_oth_gt, mean=True) * 100
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True)
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 10 + \
        #                self.loss_L2(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 10
        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01
        self.loss_cyc_xy = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref.detach(), mean=True) + \
                           self.loss_L2(self.cyc_img_oth,    self.hr_img_oth.detach(), mean=True)
        self.loss_cyc_yx = self.loss_L2(self.net_G1(self.net_G2(self.img_real_cen)), self.img_real_cen, mean=True) + \
                           self.loss_L2(self.net_G1(self.net_G2(self.img_real_ref)), self.img_real_ref, mean=True)

        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01 + \
        #                     self.loss_L1(self.deblur_hr_img_oth, self.net_G1(self.deblur_hr_img_oth), mean=True) * 0.01

        # compute perceptual loss
        #self.loss_perceptural_style = self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #                              self.loss_content.get_loss(self.deblur_hr_img_oth, self.hr_img_oth_gt) * 0.01

        # compute frequency loss
        #self.loss_frequency_style = self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) + \
        #                            self.loss_Fre(self.deblur_hr_img_oth, self.hr_img_oth_gt)

        # compute texture loss
        #dr_prevlayer_ref, _ = self.vgg(self.deblur_hr_img_ref)
        #dr_prevlayer_oth, _ = self.vgg(self.deblur_hr_img_oth)
        #hr_prevlayer_ref, _ = self.vgg(self.hr_img_ref_gt)
        #hr_prevlayer_oth, _ = self.vgg(self.hr_img_oth_gt)

        #self.loss_texture_matching_style = self.loss_texture(hr_prevlayer_ref, dr_prevlayer_ref) * 1e-4 + \
        #                                   self.loss_texture(hr_prevlayer_oth, dr_prevlayer_oth) * 1e-4

        # compute GAN loss
        self.loss_G_D1 = (-self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean())
        self.loss_G_D2 = (-self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean())

        self.loss_G_style = self.loss_self_style \
                    + self.loss_cyc_xy \
                    + self.loss_cyc_yx \
                    + self.loss_G_D1 \
                    + self.loss_G_D2

        self.optimizer_G_style.zero_grad()
        self.loss_G_style.backward()
        self.optimizer_G_style.step()

    def optimize_D(self):
        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(torch.cat([self.hr_img_ref_gt, self.hr_img_oth_gt, self.hr_img_real], dim=0))
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        for i in range(self.opts.d_step):
            self.optimizer_D1.zero_grad()
            self.loss_D1 = self.loss_GAN(self.net_D1, self.img_real_cen, self.deblur_hr_img_ref, use_gp=True) + \
                           self.loss_GAN(self.net_D1, self.img_real_ref, self.deblur_hr_img_oth, use_gp=True)
            self.loss_D1.backward()
            self.optimizer_D1.step()

            self.optimizer_D2.zero_grad()
            self.loss_D2 = self.loss_GAN(self.net_D2, self.hr_img_ref.detach(), self.net_G2(self.img_real_cen), use_gp=True) + \
                           self.loss_GAN(self.net_D2, self.hr_img_oth.detach(), self.net_G2(self.img_real_ref), use_gp=True)
            self.loss_D2.backward()
            self.optimizer_D2.step()

        #self.optimizer_D.zero_grad()
        #self.loss_D.backward()
        #self.gradient_clip(['D'])
        #self.optimizer_D.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, img_real_cen, img_real_ref):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        self.img_real_cen = img_real_cen
        self.img_real_ref = img_real_ref

    def optimize(self):
        self.optimize_G_sr()
        self.optimize_G_style()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G_sr.step()
        self.scheduler_G_style.step()
        self.scheduler_D1.step()
        self.scheduler_D2.step()

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.save_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.load_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_flow'] = self.loss_flow.item()
        losses['loss_self_sr'] = self.loss_self_sr.item()
        #losses['loss_identity'] = self.loss_identity.item()
        losses['loss_perceptural_sr'] = self.loss_perceptural_sr.item()
        losses['loss_frequency_sr'] = self.loss_frequency_sr.item()
        losses['loss_texture_matching_sr'] = self.loss_texture_matching_sr.item()
        losses['loss_G_sr'] = self.loss_G_sr.item()

        losses['loss_self_style'] = self.loss_self_style.item()
        losses['loss_cyc_xy'] = self.loss_cyc_xy.item()
        losses['loss_cyc_yx'] = self.loss_cyc_yx.item()
        losses['loss_G_D1'] = self.loss_G_D1.item()
        losses['loss_G_D2'] = self.loss_G_D2.item()
        losses['loss_G_style'] = self.loss_G_style.item()

        losses['loss_D1'] = self.loss_D1.item()
        losses['loss_D2'] = self.loss_D2.item()
        

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
            losses['PSNR_deblur'] = PSNR(self.deblur_hr_img_ref.data, self.hr_img_ref_gt)
        return losses


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


class FlowCircleSRGNet(BaseModel):
    def __init__(self, opts):
        super(FlowCircleSRGNet, self).__init__()
        self.opts = opts
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'G1', 'G2']
        #self.net_sr   = EDSR(opts).cuda()
        #self.net_sr   = DRLN(opts).cuda()
        #self.net_sr   = RCAN(opts).cuda()
        self.net_Feature_Head = Feature_Head(opts).cuda()
        self.net_Feature_extractor = Feature_extractor(opts).cuda()
        self.net_Upscalar = Upscalar(opts).cuda()
        self.net_Downscalar = Downscalar(opts).cuda()

        self.net_flow = UpPWCDCNet().cuda()
        self.net_D1   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_D2   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        #self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()
        #self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=7).cuda()
        self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        #self.net_G2   = EDLR(opts).cuda()

        # print network
        #self.print_networks(self.net_sr)
        self.print_networks(self.net_Feature_Head)
        self.print_networks(self.net_Feature_extractor)
        self.print_networks(self.net_Upscalar)
        self.print_networks(self.net_Downscalar)
        #self.print_networks(self.net_flow)
        self.print_networks(self.net_D1)
        self.print_networks(self.net_D2)
        self.print_networks(self.net_G1)
        self.print_networks(self.net_G2)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size, 2, im_crop_H, im_crop_W]

        print('grid size', grid.size())
        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                #self.load_network(self.net_sr,   'SR', opts.epoch_to_load, self.opts.model_dir)
                self.load_network(self.net_flow, 'Flow', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_Feature_Head.parameters()},
                {'params': self.net_Feature_extractor.parameters()},
                {'params': self.net_Upscalar.parameters()},
                {'params': self.net_Downscalar.parameters()},
                {'params': self.net_G1.parameters()},
                {'params': self.net_G2.parameters()}], lr=opts.lr)

            self.optimizer_D1 = torch.optim.Adam([
                {'params': self.net_D1.parameters()}], lr=opts.lr)

            self.optimizer_D2 = torch.optim.Adam([
                {'params': self.net_D2.parameters()}], lr=opts.lr)

            self.scheduler_G  = torch.optim.lr_scheduler.StepLR(self.optimizer_G,  step_size=opts.lr_step, gamma=opts.lr_decay)
            
            self.scheduler_D1 = torch.optim.lr_scheduler.StepLR(self.optimizer_D1, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D2 = torch.optim.lr_scheduler.StepLR(self.optimizer_D2, step_size=opts.lr_step, gamma=opts.lr_decay)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            #self.loss_tv_flow = VariationLoss(nc=2, grad_fn=grid_gradient_central_diff)
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self',            10))
                f.write('{} : {}\n'.format('loss_self_deblur',     10))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       10))

                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.5))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        #self.noise_amp = 0.001
        #self.using_noise = False

    def forward(self):

        #hr_img_ref = self.net_sr(self.lr_img_ref)
        #hr_img_oth = self.net_sr(self.lr_img_oth)

        lr_feature_head_ref    = self.net_Feature_Head(self.lr_img_ref)
        lr_feature_head_oth    = self.net_Feature_Head(self.lr_img_oth)
        lr_content_feature_ref = self.net_Feature_extractor(lr_feature_head_ref)
        lr_content_feature_oth = self.net_Feature_extractor(lr_feature_head_oth)
        lr_content_output_ref = lr_feature_head_ref + lr_content_feature_ref
        lr_content_output_oth = lr_feature_head_oth + lr_content_feature_oth
        hr_img_ref = self.net_Upscalar(lr_content_output_ref)
        hr_img_oth = self.net_Upscalar(lr_content_output_oth)
        
        hr_feature_head_ref    = self.net_Feature_Head(hr_img_ref)
        hr_feature_head_oth    = self.net_Feature_Head(hr_img_oth)
        hr_content_feature_ref = self.net_Feature_extractor(hr_feature_head_ref)
        hr_content_feature_oth = self.net_Feature_extractor(hr_feature_head_oth)
        hr_content_output_ref = hr_feature_head_ref + hr_content_feature_ref
        hr_content_output_oth = hr_feature_head_oth + hr_content_feature_oth
        lr_img_syn_ref = self.net_Downscalar(hr_content_output_ref)
        lr_img_syn_oth = self.net_Downscalar(hr_content_output_oth)

        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.hr_img_ref.shape[2], self.hr_img_ref.shape[3]).cuda() * self.noise_amp
        #res_ref = self.net_G1(self.hr_img_ref + noise)
        #res_oth = self.net_G1(self.hr_img_oth + noise)
        self.deblur_hr_img_ref = self.net_G1(hr_img_ref)
        self.deblur_hr_img_oth = self.net_G1(hr_img_oth)

        #deblur_hr_img_ref = self.hr_img_ref + res_ref
        #deblur_hr_img_oth = self.hr_img_oth + res_oth
        #deblur_hr_img_ref = self.net_G1(self.hr_img_ref)

        self.cyc_img_ref = self.net_G2(self.deblur_hr_img_ref)
        self.cyc_img_oth = self.net_G2(self.deblur_hr_img_oth)

        self.flows_ref_to_other = self.net_flow(hr_img_ref, hr_img_oth)
        self.flows_other_to_ref = self.net_flow(hr_img_oth, hr_img_ref)

        #flow_ref_to_other = self.upsample_4(flows_ref_to_other[0])*20.0
        #flow_other_to_ref = self.upsample_4(flows_other_to_ref[0])*20.0 

        #return deblur_hr_img_ref, deblur_hr_img_oth
        return hr_img_ref, hr_img_oth, lr_img_syn_ref, lr_img_syn_oth

    def optimize_G(self):

        #self.lr_img_ref     = lr_img_ref
        #self.hr_img_others  = hr_img_others        
        #self.hr_img_ref, \
        #self.deblur_hr_img_ref, \
        #self.cyc_img_ref, \
        #self.flow_ref_to_other, \
        #self.flow_other_to_ref = self.forward(self.lr_img_ref, self.hr_img_others)
        #self.deblur_hr_img_ref, self.deblur_hr_img_oth = self.forward()
        self.hr_img_ref, self.hr_img_oth, self.lr_img_syn_ref, self.lr_img_syn_oth = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)
        self.hr_mask_oth = self.mask_fn(self.flows_ref_to_other[0]*20.0)

        # compute synthetic hr images
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_others, self.flow_ref_to_other)
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_others, self.flow_ref_to_other)
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth, self.flows_ref_to_other[0]*20.0)
        self.syn_hr_img_oth = self.Backward_warper(self.hr_img_ref, self.flows_other_to_ref[0]*20.0)

        # compute self consistency losses
        self.loss_self = self.loss_L1(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
                         self.loss_L1(self.hr_img_oth, self.hr_img_oth_gt, mean=True) * 10 + \
                         self.loss_L1(self.lr_img_syn_ref, self.lr_img_ref, mean=True) * 10 + \
                         self.loss_L1(self.lr_img_syn_oth, self.lr_img_oth, mean=True) * 10 + \
                         self.loss_L1(self.deblur_hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
                         self.loss_L1(self.deblur_hr_img_oth, self.hr_img_oth_gt, mean=True) * 10

        #self.loss_L1(self.deblur_hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
        #self.loss_L1(self.deblur_hr_img_oth, self.hr_img_oth_gt, mean=True) * 10
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True)
        #self.loss_cyc = self.loss_L2(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 10 + \
        #                self.loss_L2(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 10
        #self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01
        self.loss_cyc_xy = self.loss_L2(self.cyc_img_ref,    self.hr_img_ref, mean=True) + \
                           self.loss_L2(self.cyc_img_oth,    self.hr_img_oth, mean=True)
        self.loss_cyc_yx = self.loss_L2(self.net_G1(self.net_G2(self.img_real_cen)), self.img_real_cen, mean=True) + \
                           self.loss_L2(self.net_G1(self.net_G2(self.img_real_ref)), self.img_real_ref, mean=True)
        self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01 + \
                             self.loss_L1(self.deblur_hr_img_oth, self.net_G1(self.deblur_hr_img_oth), mean=True) * 0.01

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref, self.hr_mask_ref, mean=True)
        self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True) * 0.5 + \
                         self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth_gt, self.hr_mask_oth, mean=True) * 0.5
        #self.loss_tv_flow(self.flow_ref_to_other, mean=True) * 0.01 + \
        #self.loss_tv_flow(self.flow_other_to_ref, mean=True) * 0.01

        # compute perceptual loss
        self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.04 + \
                                self.loss_content.get_loss(self.hr_img_oth, self.hr_img_oth_gt) * 0.04
        #self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #self.loss_content.get_loss(self.deblur_hr_img_oth, self.hr_img_oth_gt) * 0.01

        # compute frequency loss
        self.loss_frequency   = self.loss_Fre(self.hr_img_ref, self.hr_img_ref_gt) + \
                                self.loss_Fre(self.hr_img_oth, self.hr_img_oth_gt)
        #self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) + \
        #self.loss_Fre(self.deblur_hr_img_oth, self.hr_img_oth_gt)

        # compute texture loss
        sr_prevlayer_ref, _ = self.vgg(self.hr_img_ref)
        sr_prevlayer_oth, _ = self.vgg(self.hr_img_oth)
        #dr_prevlayer_ref, _ = self.vgg(self.deblur_hr_img_ref)
        #dr_prevlayer_oth, _ = self.vgg(self.deblur_hr_img_oth)
        hr_prevlayer_ref, _ = self.vgg(self.hr_img_ref_gt)
        hr_prevlayer_oth, _ = self.vgg(self.hr_img_oth_gt)

        self.loss_texture_matching = self.loss_texture(hr_prevlayer_ref, sr_prevlayer_ref) * 1e-4 + \
                                     self.loss_texture(hr_prevlayer_oth, sr_prevlayer_oth) * 1e-4
        #self.loss_texture(hr_prevlayer_ref, dr_prevlayer_ref) * 1e-4 + \
        #self.loss_texture(hr_prevlayer_oth, dr_prevlayer_oth) * 1e-4

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(torch.cat([self.deblur_hr_img_ref, self.deblur_hr_img_oth], dim=0))
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        #self.loss_G_D1 = -self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean()
        #self.loss_G_D2 = -self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean()
        self.loss_G_D1 = (-self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean()) * 1e-4
        self.loss_G_D2 = (-self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean()) * 1e-4

        self.loss_G =  self.loss_flow \
                        + self.loss_self \
                        + self.loss_cyc_xy \
                        + self.loss_cyc_yx \
                        + self.loss_identity \
                        + self.loss_perceptural \
                        + self.loss_frequency \
                        + self.loss_texture_matching \
                        + self.loss_G_D1 \
                        + self.loss_G_D2


        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()


    def optimize_D(self):
        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(torch.cat([self.hr_img_ref_gt, self.hr_img_oth_gt, self.hr_img_real], dim=0))
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        for i in range(self.opts.d_step):
            self.optimizer_D1.zero_grad()
            self.loss_D1 = self.loss_GAN(self.net_D1, self.img_real_cen, self.deblur_hr_img_ref, use_gp=True) + \
                           self.loss_GAN(self.net_D1, self.img_real_ref, self.deblur_hr_img_oth, use_gp=True)
            self.loss_D1.backward()
            self.optimizer_D1.step()

            self.optimizer_D2.zero_grad()
            self.loss_D2 = self.loss_GAN(self.net_D2, self.hr_img_ref.detach(), self.net_G2(self.img_real_cen), use_gp=True) + \
                           self.loss_GAN(self.net_D2, self.hr_img_oth.detach(), self.net_G2(self.img_real_ref), use_gp=True)
            self.loss_D2.backward()
            self.optimizer_D2.step()

        #self.optimizer_D.zero_grad()
        #self.loss_D.backward()
        #self.gradient_clip(['D'])
        #self.optimizer_D.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, img_real_cen, img_real_ref):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        self.img_real_cen = img_real_cen
        self.img_real_ref = img_real_ref

    def optimize(self):
        self.optimize_G()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D1.step()
        self.scheduler_D2.step()

    def save_checkpoint(self, label):
        #self.save_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_Feature_Head,        'Feature_Head',        label, self.opts.checkpoint_dir)
        self.save_network(self.net_Feature_extractor,   'Feature_extractor',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_Upscalar,            'Upscalar',            label, self.opts.checkpoint_dir)
        self.save_network(self.net_Downscalar,          'Downscalar',          label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.save_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        #self.load_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_Feature_Head,        'Feature_Head',        label, self.opts.checkpoint_dir)
        self.load_network(self.net_Feature_extractor,   'Feature_extractor',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_Upscalar,            'Upscalar',            label, self.opts.checkpoint_dir)
        self.load_network(self.net_Downscalar,          'Downscalar',          label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.load_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_flow'] = self.loss_flow.item()
        losses['loss_self'] = self.loss_self.item()
        losses['loss_cyc_xy'] = self.loss_cyc_xy.item()
        losses['loss_cyc_yx'] = self.loss_cyc_yx.item()
        losses['loss_identity'] = self.loss_identity.item()
        losses['loss_perceptural'] = self.loss_perceptural.item()
        losses['loss_frequency'] = self.loss_frequency.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D1'] = self.loss_G_D1.item()
        losses['loss_G_D2'] = self.loss_G_D2.item()
        losses['loss_G'] = self.loss_G.item()

        losses['loss_D1'] = self.loss_D1.item()
        losses['loss_D2'] = self.loss_D2.item()
        

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
            losses['PSNR_deblur'] = PSNR(self.deblur_hr_img_ref.data, self.hr_img_ref_gt)
        return losses


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class SRGNet(BaseModel):
    def __init__(self, opts):
        super(SRGNet, self).__init__()
        self.opts = opts
 
        # create network
        self.model_names = ['sr', 'D', 'G1', 'G2']
        self.net_sr = EDSR(opts).cuda()
        self.net_D  = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        #self.net_G1 = ResnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats).cuda()
        #self.net_G2 = ResnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats).cuda()
        self.net_G1 = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()
        self.net_G2 = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_D)
        self.print_networks(self.net_G1)
        self.print_networks(self.net_G2)

        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)

        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training: 
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_sr.parameters()},
                {'params': self.net_G1.parameters()},
                {'params': self.net_G2.parameters()}], lr=opts.lr)

            self.optimizer_D = torch.optim.Adam([{'params': self.net_D.parameters()}], lr=opts.lr/10.0)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self',            10))
                f.write('{} : {}\n'.format('loss_identity',        1))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_G_D',             0.1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.01))
                #f.write('{} : {}\n'.format('loss_img_smooth',      0.01))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None


    def forward(self, lr_img_ref, lr_img_others):

        hr_img_ref = self.net_sr(lr_img_ref)
        hr_img_others = self.net_sr(lr_img_others)

        deblur_hr_img_ref = self.net_G1(hr_img_ref)
        deblur_hr_img_others = self.net_G1(hr_img_others)

        cyc_hr_img_ref = self.net_G2(deblur_hr_img_ref)
        cyc_hr_img_others = self.net_G2(deblur_hr_img_others)

        return hr_img_ref, hr_img_others, deblur_hr_img_ref, deblur_hr_img_others, cyc_hr_img_ref, cyc_hr_img_others

    def optimize_G(self, lr_img_ref, lr_img_others):

        self.lr_img_ref     = lr_img_ref
        self.lr_img_others  = lr_img_others

        self.hr_img_ref, \
        self.hr_img_others, \
        self.deblur_hr_img_ref, \
        self.deblur_hr_img_others, \
        self.cyc_hr_img_ref, \
        self.cyc_hr_img_others = self.forward(self.lr_img_ref, self.lr_img_others)

        # compute self consistency losses
        self.loss_self_ref = self.loss_L1(self.hr_img_ref_gt, self.hr_img_ref, mean=True) * 10
        self.loss_self_others = self.loss_L1(self.hr_img_oth_gt, self.hr_img_others, mean=True) * 10

        self.loss_self_deblur_ref = self.loss_L1(self.hr_img_ref_gt, self.deblur_hr_img_ref, mean=True) * 10
        self.loss_self_deblur_others = self.loss_L1(self.hr_img_oth_gt, self.deblur_hr_img_others, mean=True) * 10

        self.loss_cyc_ref    = self.loss_L2(self.cyc_hr_img_ref, self.hr_img_ref, mean=True)
        self.loss_cyc_others = self.loss_L2(self.cyc_hr_img_others, self.hr_img_others, mean=True)

        self.loss_identity_ref    = self.loss_L1(self.deblur_hr_img_ref,    self.net_G1(self.deblur_hr_img_ref), mean=True)
        self.loss_identity_others = self.loss_L1(self.deblur_hr_img_others, self.net_G1(self.deblur_hr_img_others), mean=True)

        # compute smoothness loss
        #self.loss_img_smooth         = (self.loss_tv_img(self.hr_img_ref, mean=True) + self.loss_tv_img(self.hr_img_others, mean=True)) * 0.01
        #self.loss_img_smooth_deblur  = (self.loss_tv_img(self.deblur_hr_img_ref, mean=True) + self.loss_tv_img(self.deblur_hr_img_others, mean=True)) * 0.01
        
        # compute perceptual loss
        self.loss_perceptural        = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.01
        self.loss_perceptural_deblur = self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01
        #self.loss_perceptural        = self.loss_content(self.hr_img_ref, self.hr_img_ref_gt) * 0.01
        #self.loss_perceptural_deblur = self.loss_content(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01

        # compute frequency loss
        self.loss_frequency          = self.loss_Fre(self.hr_img_ref, self.hr_img_ref_gt) * 10
        self.loss_frequency_deblur   = self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 10

        # compute GAN loss
        #fake_hr_image = torch.cat([self.hr_img_others, self.hr_img_others], dim=0)
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(fake_hr_image)
        #fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        
        self.fake_hr_image = torch.cat([self.deblur_hr_img_ref, self.deblur_hr_img_others], dim=0)
        dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.fake_hr_image)
        self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        #self.loss_G_D = self.loss_GAN(self.net_D(self.net_G(self.hr_img_ref)), target_is_real=True)

        #self.deblur_hr_img_ref = self.net_G(self.hr_img_ref)
        #self.loss_D_G = self.loss_GAN(self.net_D(self.deblur_hr_img_ref), target_is_real=True)
        #self.loss_D_G_content = self.loss_content.get_loss(self.hr_img_ref, self.deblur_hr_img_ref) * 0.02
        #self.loss_D_G_total = self.loss_D_G + self.loss_D_G_content
                       
        self.loss_G = self.loss_self_ref \
                    + self.loss_self_others \
                    + self.loss_self_deblur_ref \
                    + self.loss_self_deblur_others \
                    + self.loss_cyc_ref \
                    + self.loss_cyc_others \
                    + self.loss_identity_ref \
                    + self.loss_identity_others \
                    + self.loss_perceptural \
                    + self.loss_perceptural_deblur \
                    + self.loss_frequency \
                    + self.loss_frequency_deblur \
                    + self.loss_G_D
                    #+ self.loss_img_smooth \
                    #+ self.loss_img_smooth_deblur \
                    

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_D(self, hr_img_real):

        real_hr_image = torch.cat([hr_img_real, hr_img_real], dim=0)
        dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(real_hr_image)
        real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)

        self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_G2,   'G2',   label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_G2,   'G2',   label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_self_ref'] = self.loss_self_ref.item()
        losses['loss_self_others'] = self.loss_self_others.item()
        losses['loss_self_deblur_ref'] = self.loss_self_deblur_ref.item()
        losses['loss_self_deblur_others'] = self.loss_self_deblur_others.item()
        losses['loss_cyc_ref'] = self.loss_cyc_ref.item()
        losses['loss_cyc_others'] = self.loss_cyc_others.item()
        losses['loss_identity_ref'] = self.loss_identity_ref.item()
        losses['loss_identity_others'] = self.loss_identity_others.item()
        #losses['loss_img_smooth'] = self.loss_img_smooth.item()
        #losses['loss_img_smooth_deblur'] = self.loss_img_smooth_deblur.item()
        losses['loss_perceptural'] = self.loss_perceptural.item()
        losses['loss_perceptural_deblur'] = self.loss_perceptural_deblur.item()
        losses['loss_frequency'] = self.loss_frequency.item()
        losses['loss_frequency_deblur'] = self.loss_frequency_deblur.item()
        losses['loss_G_D'] = self.loss_G_D.item()
        losses['loss_G'] = self.loss_G.item()
        losses['loss_D'] = self.loss_D.item()
        
        if self.hr_img_ref_gt is not None:
            #losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
            losses['PSNR'] = (PSNR(self.deblur_hr_img_ref.data, self.hr_img_ref_gt) + \
                            PSNR(self.deblur_hr_img_others.data, self.hr_img_oth_gt)) * 0.5
        return losses






#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class SRNet(BaseModel):
    def __init__(self, opts):
        super(SRNet, self).__init__()
        self.opts = opts
 
        # create network
        self.model_names = ['sr', 'D']
        
        self.net_sr = EDSR(opts).cuda()
        #self.net_sr = RCAN(opts).cuda()
        #self.net_sr = DRLN(opts).cuda()
        #self.net_sr = EEDSR(opts).cuda()
        #self.net_sr = EEDSR2(opts).cuda()
        #self.net_sr = EEDSR3(opts).cuda()

        #self.net_sr = nn.Upsample(scale_factor=4, mode='bilinear')
        self.net_D = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        #self.net_G = GeneratorConcatSkip2CleanAdd(opts).cuda()
        #self.net_G = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=5).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_D)
        #self.print_networks(self.net_G)

        #grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        #grid = grid.int().cuda().unsqueeze(0)
        #grid = grid.repeat(opts.batch_size-1, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]

        #print('grid size', grid.size())
        #self.mask_fn = FlowWarpMask(grid)

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                self.load_network(self.net_sr, 'SR', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)
            #pass

        if opts.is_training:
            # initialize optimizers
            #self.optimizer_G = torch.optim.Adam([{'params': self.net_sr.parameters()}], lr=opts.lr)
            self.optimizer_G = torch.optim.Adam([{'params': self.net_sr.parameters()}], lr=opts.lr)
            self.optimizer_D = torch.optim.Adam([{'params': self.net_D.parameters()}], lr=opts.lr)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_data = L1()
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
            self.loss_GAN = DiscLossWGANGP(LAMBDA=self.opts.LAMBDA)
            self.loss_content = PerceptualLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self_ref',        10))               
                f.write('{} : {}\n'.format('loss_self_others',     10))
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.1))
                f.write('{} : {}\n'.format('loss_img_smooth',      0.01))
                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        self.noise_amp = 0.001

    def forward(self):
        #assert(lr_ref_img.size(0)==lr_other_imgs.size(0))
        #self.noise       = torch.randn(self.opts.batch_size, self.opts.n_colors, self.opts.im_crop_H, self.opts.im_crop_W).cuda() * self.noise_amp
        #hr_img_ref = self.net_sr(self.lr_img_ref) + self.upsample_4(self.lr_img_ref)
        hr_img_ref = self.net_sr(self.lr_img_ref)
        #hr_other_imgs = self.net_sr(lr_other_imgs)
        #self.res         = self.net_G(self.hr_img_ref + self.noise)
        #synthesis_output = self.hr_img_ref + self.res

        #return synthesis_output
        return hr_img_ref

    def optimize_G(self):
        #assert(lr_img_ref.size(0)==lr_img_others.size(0))

        #self.lr_img_ref     = lr_img_ref
        #self.lr_img_others  = lr_img_others
        #self.hr_img_ref, self.hr_img_others = self.forward(self.lr_img_ref, self.lr_img_others)
        self.hr_img_ref = self.forward()

        # compute self consistency losses
        #self.loss_self_ref = self.loss_data(self.hr_img_ref_gt, self.hr_img_ref, mean=True) * 10
        #self.loss_self_others = self.loss_data(self.hr_img_oth_gt, self.hr_img_others, mean=True) * 10
        #self.loss_self = self.loss_data(self.synthesis_output, self.hr_img_ref_gt, mean=True) * 10
        self.loss_self = self.loss_data(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 10
                         

        # compute smoothness loss
        #self.loss_img_smooth  = (self.loss_tv_img(self.hr_img_ref, mean=True) + self.loss_tv_img(self.hr_img_others, mean=True)) * 0.01
        
        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.04
        #self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.04
        self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.04

        # compute texture matching loss
        #sr_prevlayer, _ = self.vgg(self.hr_img_ref)
        #hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        #self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-5
        
        sr_prevlayer, _ = self.vgg(self.hr_img_ref)
        hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-3

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.synthesis_output)
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        self.loss_G_D = -self.net_D(self.hr_img_ref).mean()

        self.loss_G = self.loss_self \
                    + self.loss_G_D \
                    + self.loss_perceptural \
                    + self.loss_texture_matching

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        #self.scheduler_G.step()
        #for param_group in self.optimizer_G.param_groups:
        #    print('G ', param_group['lr'])

    def optimize_D(self):

        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.fake_hr_image.detach())
        #fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)

        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)

        for i in range(self.opts.d_step):
            self.optimizer_D.zero_grad()
            self.loss_D = self.loss_GAN(self.net_D, self.hr_img_ref_gt, self.hr_img_ref, use_gp=True)
            self.loss_D.backward()
            self.optimizer_D.step()
        #self.scheduler_D.step()
        #for param_group in self.optimizer_D.param_groups:
        #    print('D ', param_group['lr'])

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, img_real_cen, img_real_ref):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        #self.hr_img_real = hr_img_real
        self.img_real_cen = img_real_cen
        self.img_real_ref = img_real_ref        

    def optimize(self):
        self.optimize_G()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
        #self.noise_amp = self.noise_amp * 0.995

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        #self.save_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        #self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_self_ref'] = self.loss_self_ref.item()
        #losses['loss_self_others'] = self.loss_self_others.item()
        #losses['loss_img_smooth'] = self.loss_img_smooth.item()
        losses['loss_self'] = self.loss_self.item()
        losses['loss_perceptural'] = self.loss_perceptural.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()
        losses['loss_D'] = self.loss_D.item()
        losses['loss_G'] = self.loss_G.item()

        if self.hr_img_ref_gt is not None:
            #losses['PSNR'] = PSNR(self.synthesis_output.data, self.hr_img_ref_gt)
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
        return losses




#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class CircleSRNet(BaseModel):
    def __init__(self, opts):
        super(CircleSRNet, self).__init__()
        self.opts = opts
 
        # create network
        self.model_names = ['Feature_Head', 'Feature_extractor', 'D', 'Upscalar', 'Downscalar']
        self.net_Feature_Head = Feature_Head(opts).cuda()
        self.net_Feature_extractor = Feature_extractor(opts).cuda()
        self.net_Upscalar = Upscalar(opts).cuda()
        self.net_Downscalar = Downscalar(opts).cuda()

        self.net_D = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()

        # print network
        self.print_networks(self.net_Feature_Head)
        self.print_networks(self.net_Feature_extractor)
        self.print_networks(self.net_Upscalar)
        self.print_networks(self.net_Downscalar)
        self.print_networks(self.net_D)

        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)

        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_Feature_Head.parameters()},
                {'params': self.net_Feature_extractor.parameters()},
                {'params': self.net_Upscalar.parameters()},
                {'params': self.net_Downscalar.parameters()}], lr=opts.lr)

            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_data = L1()
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_self_ref',        10))               
                f.write('{} : {}\n'.format('loss_self_others',     10))
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.1))
                f.write('{} : {}\n'.format('loss_img_smooth',      0.01))
                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        #self.hr_img_gt = None

    def forward(self):
        lr_feature_head    = self.net_Feature_Head(self.lr_img_ref)
        lr_content_feature = self.net_Feature_extractor(lr_feature_head)
        lr_content_output = lr_feature_head + lr_content_feature
        hr_img = self.net_Upscalar(lr_content_output)
        
        hr_feature_head    = self.net_Feature_Head(hr_img)
        hr_content_feature = self.net_Feature_extractor(hr_feature_head)
        hr_content_output = hr_feature_head + hr_content_feature
        lr_img_syn = self.net_Downscalar(hr_content_output)

        return hr_img, lr_img_syn

    def optimize_G(self):
        
        #self.lr_img = torch.cat([lr_img_ref, lr_img_others], dim=0)

        #self.lr_img_ref     = lr_img_ref
        #self.lr_img_others  = lr_img_others
        #self.hr_img_ref, self.hr_img_others = self.forward(self.lr_img_ref, self.lr_img_others)
        self.hr_img_ref, self.lr_img_syn = self.forward()

        # compute self consistency losses
        #self.loss_self_ref = self.loss_data(self.hr_img_ref_gt, self.hr_img_ref, mean=True) * 10
        #self.loss_self_others = self.loss_data(self.hr_img_oth_gt, self.hr_img_others, mean=True) * 10
        self.loss_consistent_hr = self.loss_data(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 10
        self.loss_consistent_lr = self.loss_data(self.lr_img_syn, self.lr_img_ref, mean=True) * 10

        # compute smoothness loss
        #self.loss_img_smooth  = (self.loss_tv_img(self.hr_img_ref, mean=True) + self.loss_tv_img(self.lr_img_syn, mean=True)) * 0.01
        
        # compute perceptual loss
        self.loss_perceptural_hr = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.04
        #self.loss_perceptural_lr = self.loss_content.get_loss(self.lr_img_syn, self.lr_img_ref) * 0.1

        # compute texture matching loss
        sr_prevlayer, _ = self.vgg(self.hr_img_ref)
        hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-3

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.hr_img_ref)
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        self.loss_G_D = -self.net_D(self.hr_img_ref).mean()

        self.loss_G = self.loss_consistent_hr \
                    + self.loss_consistent_lr \
                    + self.loss_perceptural_hr \
                    + self.loss_texture_matching \
                    + self.loss_G_D

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        #self.scheduler_G.step()
        #for param_group in self.optimizer_G.param_groups:
        #    print('G ', param_group['lr'])

    def optimize_D(self):

        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.hr_img_ref.detach())
        #fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)

        #hr_img_real = torch.cat([hr_img_real, hr_img_real], dim=0)
        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(hr_img_real)
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)

        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        #self.optimizer_D.zero_grad()
        #self.loss_D.backward()
        #self.optimizer_D.step()
        for i in range(self.opts.d_step):
            self.optimizer_D.zero_grad()
            self.loss_D = self.loss_GAN(self.net_D, self.hr_img_ref_gt, self.hr_img_ref, use_gp=True)
            self.loss_D.backward()
            self.optimizer_D.step()
        #self.scheduler_D.step()
        #for param_group in self.optimizer_D.param_groups:
        #    print('D ', param_group['lr'])

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        #self.hr_img_ref_gt = hr_img_ref_gt
        #self.hr_img_oth_gt = hr_img_oth_gt
        #self.hr_img_gt = torch.cat([hr_img_ref_gt, hr_img_oth_gt], dim=0)
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, img_real_cen, img_real_ref):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        #self.hr_img_real = hr_img_real
        self.img_real_cen = img_real_cen
        self.img_real_ref = img_real_ref

    def optimize(self):
        self.optimize_G()
        self.optimize_D()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def save_checkpoint(self, label):
        self.save_network(self.net_Feature_Head,        'Feature_Head',        label, self.opts.checkpoint_dir)
        self.save_network(self.net_Feature_extractor,   'Feature_extractor',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_Upscalar,            'Upscalar',            label, self.opts.checkpoint_dir)
        self.save_network(self.net_Downscalar,          'Downscalar',          label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,                   'D',                   label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_Feature_Head,        'Feature_Head',        label, self.opts.checkpoint_dir)
        self.load_network(self.net_Feature_extractor,   'Feature_extractor',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_Upscalar,            'Upscalar',            label, self.opts.checkpoint_dir)
        self.load_network(self.net_Downscalar,          'Downscalar',          label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,                   'D',                   label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_consistent_hr'] = self.loss_consistent_hr.item()
        losses['loss_consistent_lr'] = self.loss_consistent_lr.item()
        #losses['loss_img_smooth'] = self.loss_img_smooth.item()
        losses['loss_perceptural_hr'] = self.loss_perceptural_hr.item()
        #losses['loss_perceptural_lr'] = self.loss_perceptural_lr.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()
        losses['loss_D'] = self.loss_D.item()
        losses['loss_G'] = self.loss_G.item()

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
        return losses






#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
class FlowNet(BaseModel):
    def __init__(self, opts):
        super(FlowNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['flow', 'mask']
        self.model_names = ['flow']
        self.net_flow = PWCDCNet().cuda()
        #self.net_flow = UpPWCDCNet().cuda()
        #self.net_mask = MaskNet().cuda()

        # print network
        self.print_networks(self.net_flow)
        #self.print_networks(self.net_mask)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]

        print('grid size', grid.size())
        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)

        if opts.use_pretrained_model:
            #self.load_network(self.net_flow, 'Flow', opts.epoch_to_load, self.opts.model_dir)
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training: 
            self.optimizer_flow = torch.optim.Adam([
                {'params': self.net_flow.parameters()}], lr=opts.lr)

            self.scheduler_flow = torch.optim.lr_scheduler.StepLR(self.optimizer_flow, step_size=opts.lr_step, gamma=0.618)

            self.loss_data = L1()
            self.loss_tv_flow = VariationLoss(nc=2, grad_fn=grid_gradient_central_diff)
            #self.loss_tv_mask = VariationLoss(nc=1, grad_fn=grid_gradient_central_diff)

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_lr',              0.05))
                f.write('{} : {}\n'.format('loss_flow_smooth',     0.1))
                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None

    def forward(self):

        flows_ref_to_other = self.net_flow(self.hr_img_ref_gt, self.hr_img_oth_gt)
        flows_other_to_ref = self.net_flow(self.hr_img_oth_gt, self.hr_img_ref_gt)

        #flow_ref_to_other = self.upsample_4(flows_ref_to_other[0])*20.0
        #flow_other_to_ref = self.upsample_4(flows_other_to_ref[0])*20.0
        flow_ref_to_other = flows_ref_to_other[0]*20.0
        flow_other_to_ref = flows_other_to_ref[0]*20.0

        return flow_ref_to_other, flow_other_to_ref

    def optimize_flow(self):

        self.flow_ref_to_other, \
        self.flow_other_to_ref = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_others = self.mask_fn(self.flow_ref_to_other)
        self.hr_mask_ref = self.mask_fn(self.flow_other_to_ref)
        #self.hr_mask_others = self.net_mask(self.flow_ref_to_other)
        #self.hr_mask_ref = self.net_mask(self.flow_other_to_ref)

        # compute synthetic hr images
        #self.syn_hr_img_ref, _ = warp_image_flow(self.hr_img_oth_gt, self.flow_ref_to_other)
        #self.syn_hr_img_others, _ = warp_image_flow(self.hr_img_ref_gt, self.flow_other_to_ref)
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth_gt, self.flow_ref_to_other)
        self.syn_hr_img_others = self.Backward_warper(self.hr_img_ref_gt, self.flow_other_to_ref)

        # compute left-right consistency loss
        self.loss_lr = (self.loss_data(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True) \
                    + self.loss_data(self.syn_hr_img_others, self.hr_img_oth_gt, self.hr_mask_others, mean=True)) * 10
        #self.loss_lr = (self.loss_data(self.syn_hr_img_ref * self.hr_mask_ref, self.hr_img_ref_gt * self.hr_mask_ref, mean=True) \
        #            + self.loss_data(self.syn_hr_img_others * self.hr_mask_others, self.hr_img_oth_gt * self.hr_mask_others, mean=True)) * 10

        # compute smoothness loss
        self.loss_flow_smooth = (self.loss_tv_flow(self.flow_ref_to_other, mean=True) + self.loss_tv_flow(self.flow_other_to_ref, mean=True)) * 0.1
        
        # compute mask loss
        #self.loss_mask_ref = self.loss_data(torch.tensor(1.0).cuda().expand_as(self.hr_mask_ref), self.hr_mask_ref, mean=True) * 19
        #self.loss_mask_oth = self.loss_data(torch.tensor(1.0).cuda().expand_as(self.hr_mask_others), self.hr_mask_others, mean=True) * 10
        #self.loss_mask_ref = (1 - self.hr_mask_ref.mean()) * 10
        #self.loss_mask_oth = (1 - self.hr_mask_others.mean()) * 10
        #self.loss_mask_smooth = (self.loss_tv_mask(self.hr_mask_ref, mean=True) + self.loss_tv_mask(self.hr_mask_others, mean=True)) * 0.1
                       
        self.loss =  self.loss_lr + self.loss_flow_smooth
        #self.loss =  self.loss_lr

        self.optimizer_flow.zero_grad()
        self.loss.backward()
        self.optimizer_flow.step()

    def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
        self.hr_img_ref_gt = hr_img_ref_gt
        self.hr_img_oth_gt = hr_img_oth_gt

    def set_train_data(self, lr_img_ref, lr_img_oth, hr_img_real):
        self.lr_img_ref  = lr_img_ref
        self.lr_img_oth  = lr_img_oth
        self.hr_img_real = hr_img_real

    def optimize(self):
        self.optimize_flow()

    def update_lr(self):
        self.scheduler_flow.step()

    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        #self.save_network(self.net_mask, 'Mask', label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        #self.load_network(self.net_mask, 'Mask', label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_lr'] = self.loss_lr.item()
        losses['loss_flow_smooth'] = self.loss_flow_smooth.item()
        #losses['loss_mask_ref'] = self.loss_mask_ref.item()
        #losses['loss_mask_oth'] = self.loss_mask_oth.item()
        #losses['loss_mask_smooth'] = self.loss_mask_smooth.item()
        return losses

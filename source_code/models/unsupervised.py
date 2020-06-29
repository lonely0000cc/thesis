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
from utils.functions import *
from Blurrer_layer import FlowWarpMask

class UnFusionFlowSRNet(BaseModel):
    def __init__(self, opts):
        super(UnFusionFlowSRNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'G1', 'G2']
        self.net_sr   = EDSR(opts).cuda()
        #self.net_flow = PWCDCNet().cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D    = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        self.net_G1   = FeatureFusionGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors).cuda()
        #self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=5).cuda()
        #self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=5).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D)
        self.print_networks(self.net_G1)
        #self.print_networks(self.net_G2)

        #grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        #grid = grid.int().cuda().unsqueeze(0)
        #grid = grid.repeat(opts.batch_size-1, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]

        #print('grid size', grid.size())
        #self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training: 
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_sr.parameters()},
                {'params': self.net_G1.parameters()},
                {'params': self.net_flow.parameters()}], lr=opts.lr)
            
            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            #self.bp_loss = BackProjectionLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_bp',              1000))
                f.write('{} : {}\n'.format('loss_bp_deblur',       1000))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       100))
                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     10))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None


    def forward(self):

        #self.sr_img_ref = self.net_sr(self.lr_img_ref)
        #self.sr_img_oth = self.net_sr(self.lr_img_oth)
        self.hr_img_ref = self.net_sr(self.lr_img_ref) + self.upsample_4(self.lr_img_ref)
        self.hr_img_oth = self.net_sr(self.lr_img_oth) + self.upsample_4(self.lr_img_oth)

        #deblur_sr_img_ref = self.net_G1(self.sr_img_ref)
        #self.cyc_sr_img_ref = self.net_G2(deblur_sr_img_ref)

        self.flows_ref_to_other = self.net_flow(self.hr_img_ref, self.hr_img_oth)
        self.flows_other_to_ref = self.net_flow(self.hr_img_oth, self.hr_img_ref)

        sythsis_output_ref = self.net_G1(self.hr_img_ref, self.flows_ref_to_other, self.Backward_warper, self.hr_img_oth)
        sythsis_output_oth = self.net_G1(self.hr_img_oth, self.flows_other_to_ref, self.Backward_warper, self.hr_img_ref)
        #self.cyc_sr_img_ref = self.net_G2(sythsis_output)

        return sythsis_output_ref, sythsis_output_oth

    def optimize_G(self):

        self.sythsis_output_ref, self.sythsis_output_oth = self.forward()

        # compute hr mask & lr mask
        #self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)

        # compute synthetic hr images
        #self.syn_hr_img_ref = self.Backward_warper(self.sr_img_oth, self.flows_ref_to_other[0]*20.0)

        # compute self consistency losses
        self.loss_bp = self.loss_L1(nn.functional.avg_pool2d(self.sythsis_output_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100 + \
                       self.loss_L1(nn.functional.avg_pool2d(self.sythsis_output_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True) * 100

        #self.loss_bp_deblur   = self.loss_L1(nn.functional.avg_pool2d(self.deblur_sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100
        #self.loss_cyc         = self.loss_L2(self.cyc_sr_img_ref,    self.sr_img_ref, mean=True) * 10
        #self.loss_identity    = self.loss_L1(self.deblur_sr_img_ref, self.net_G1(self.deblur_sr_img_ref), mean=True) * 0.01

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(nn.functional.avg_pool2d(self.syn_hr_img_ref, kernel_size=self.opts.scale), \
        #                        self.lr_img_ref, nn.functional.avg_pool2d(self.hr_mask_ref.float(), kernel_size=self.opts.scale), mean=True)
        
        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.1
        #self.loss_perceptural = self.loss_content.get_loss(nn.functional.avg_pool2d(self.sythsis_output, kernel_size=self.opts.scale), self.lr_img_ref) * 0.1

        # compute frequency loss
        #self.loss_frequency   = self.loss_Fre(nn.functional.avg_pool2d(self.sythsis_output, kernel_size=self.opts.scale), self.lr_img_ref) * 10

        # compute texture loss
        #sr_prevlayer, _ = self.vgg(nn.functional.avg_pool2d(self.sythsis_output, kernel_size=self.opts.scale))
        #hr_prevlayer, _ = self.vgg(self.lr_img_ref)
        #self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer)

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.sythsis_output)
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True) * 10
        self.loss_G_D = (-self.net_D(self.hr_img_ref).mean() - self.net_D(self.hr_img_oth).mean())
                       
        self.loss_G = self.loss_bp \
                    + self.loss_G_D
                    #+ self.loss_perceptural \
                    #+ self.loss_texture_matching \
                    #+ self.loss_frequency \
                    

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        #self.gradient_clip(['sr', 'flow', 'G1', 'G2'])
        self.optimizer_G.step()

    def gradient_clip(self, names):
        for net in names:
            assert isinstance(net, str)
            net = getattr(self, 'net_' + net)
            torch.nn.utils.clip_grad_value_(net.parameters(), 1)        

    def optimize_D(self):

        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())
        for i in range(self.opts.d_step):
            self.optimizer_D.zero_grad()
            self.loss_D = self.loss_GAN(self.net_D, self.img_real_cen, self.hr_img_ref, use_gp=True) + \
                          self.loss_GAN(self.net_D, self.img_real_ref, self.hr_img_oth, use_gp=True)
            self.loss_D.backward()
            self.optimizer_D.step()

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
        self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        #self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_flow'] = self.loss_flow.item()
        losses['loss_bp'] = self.loss_bp.item()
        #losses['loss_bp_deblur'] = self.loss_bp_deblur.item()
        #losses['loss_cyc'] = self.loss_cyc.item()
        #losses['loss_identity'] = self.loss_identity.item()
        #losses['loss_perceptural'] = self.loss_perceptural.item()
        #losses['loss_frequency'] = self.loss_frequency.item()
        #losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()        
        losses['loss_G'] = self.loss_G.item()
        losses['loss_D'] = self.loss_D.item()

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.sythsis_output.data, self.hr_img_ref_gt)
            #losses['PSNR_deblur'] = PSNR(self.deblur_sr_img_ref.data, self.hr_img_ref_gt)
        return losses








#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class UnFlowSRNet(BaseModel):
    def __init__(self, opts):
        super(UnFlowSRNet, self).__init__()
        self.opts = opts
 
        # create network
        self.net_sr   = EDSR(opts).cuda()
        #self.net_sr   = DRLN(opts).cuda()
        #self.net_sr   = EEDSR(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D    = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        #self.net_D    = PatchGAN_net(input_nc=opts.n_colors).cuda()
        #self.net_G2   = EDLR(opts).cuda()
        #self.net_G2   = UnetFlowGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D)
        #self.print_networks(self.net_G1)
        #self.print_networks(self.net_G2)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]

        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                self.load_network(self.net_flow, 'Flow', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training: 
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_sr.parameters()}], lr=opts.lr)
            
            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=opts.lr_decay)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP(LAMBDA=self.opts.LAMBDA)
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_bp',              1000))
                f.write('{} : {}\n'.format('loss_bp_deblur',       1000))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       100))
                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     10))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        self.noise_amp = 0.1


    def forward(self):

        hr_img_ref = self.net_sr(self.lr_img_ref) + self.upsample_4(self.lr_img_ref)
        hr_img_oth = self.net_sr(self.lr_img_oth) + self.upsample_4(self.lr_img_oth)

        #self.d_mask= zero_out_pixels(list_shape=list(self.hr_img_ref.shape), prop=self.opts.mask_prop)
        #d_hr_img_ref = self.hr_img_ref * self.d_mask
        #d_hr_img_oth = self.hr_img_oth * self.d_mask
        #self.cyc_img_ref = self.net_G2(d_hr_img_ref)
        #self.cyc_img_oth = self.net_G2(d_hr_img_oth)
        #self.cyc_img_ref = self.net_G2(hr_img_ref)
        #self.cyc_img_oth = self.net_G2(hr_img_oth)

        self.flows_ref_to_other = self.net_flow(hr_img_ref, hr_img_oth)
        self.flows_other_to_ref = self.net_flow(hr_img_oth, hr_img_ref)

        return hr_img_ref, hr_img_oth

    def optimize_G(self):

        self.hr_img_ref, self.hr_img_oth = self.forward()
        self.cyc_img_ref = nn.functional.avg_pool2d(self.hr_img_ref, kernel_size=self.opts.scale)
        self.cyc_img_oth = nn.functional.avg_pool2d(self.hr_img_oth, kernel_size=self.opts.scale)

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)
        self.hr_mask_oth = self.mask_fn(self.flows_ref_to_other[0]*20.0)

        # compute synthetic hr images
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth, self.flows_ref_to_other[0]*20.0)
        self.syn_hr_img_oth = self.Backward_warper(self.hr_img_ref, self.flows_other_to_ref[0]*20.0)

        # compute self consistency losses
        #self.loss_bp = self.loss_L1(nn.functional.avg_pool2d(self.hr_img_real, kernel_size=self.opts.scale), self.net_G2(self.hr_img_real), mean=True)*100 + \
        #               self.loss_L1(nn.functional.avg_pool2d(self.hr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True)*10 + \
        #               self.loss_L1(nn.functional.avg_pool2d(self.hr_img_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True)*10
        self.loss_bp = self.loss_L1(self.cyc_img_ref, self.lr_img_ref, mean=True)*100 + \
                       self.loss_L1(self.cyc_img_oth, self.lr_img_oth, mean=True)*100

        #self.loss_img_smooth = (self.loss_tv_img(self.hr_img_ref, mean=True) + self.loss_tv_img(self.hr_img_oth, mean=True))*1e-2
        #self.loss_bp_deblur   = self.loss_L1(nn.functional.avg_pool2d(self.deblur_sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100
        #self.loss_cyc       = self.loss_L1(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 100 + \
        #                      self.loss_L1(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 100
        #noise = generator_noise(list_shape=list(self.hr_img_ref.shape)) * self.noise_amp
        
        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.hr_img_ref.shape[2], self.hr_img_ref.shape[3]).cuda() * self.noise_amp * self.loss_bp.item()
        #lr_noise = nn.functional.avg_pool2d(noise, kernel_size=self.opts.scale)
        #syn_noise = self.net_sr(lr_noise) + self.upsample_4(lr_noise)
        #self.loss_sr = self.loss_L1(syn_noise, noise, mean=True)

        #lr_img_real_ref = nn.functional.avg_pool2d(self.img_real_ref + noise, kernel_size=self.opts.scale)
        #lr_img_real_cen = nn.functional.avg_pool2d(self.img_real_cen + noise, kernel_size=self.opts.scale)
        
        # loss_sr included
        #lr_img_real_ref = nn.functional.avg_pool2d(self.img_real_ref, kernel_size=self.opts.scale)
        #lr_img_real_cen = nn.functional.avg_pool2d(self.img_real_cen, kernel_size=self.opts.scale)
        #syn_img_real_ref = self.net_sr(lr_img_real_ref) + self.upsample_4(lr_img_real_ref)
        #syn_img_real_cen = self.net_sr(lr_img_real_cen) + self.upsample_4(lr_img_real_cen)
        #self.loss_sr = self.loss_L1(syn_img_real_ref, self.img_real_ref, mean=True) + \
        #               self.loss_L1(syn_img_real_cen, self.img_real_cen, mean=True)

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(nn.functional.avg_pool2d(self.syn_hr_img_ref, kernel_size=self.opts.scale), \
        #                        self.lr_img_ref, nn.functional.avg_pool2d(self.hr_mask_ref.float(), kernel_size=self.opts.scale), mean=True)
        self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref, self.hr_mask_ref, mean=True)*0.1 + \
                         self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth, self.hr_mask_oth, mean=True)*0.1
        
        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.1
        #self.loss_perceptural = self.loss_content.get_loss(nn.functional.avg_pool2d(self.sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref)
        #self.loss_perceptural = self.loss_content.get_loss(self.cyc_img_ref, self.lr_img_ref) + \
        #                        self.loss_content.get_loss(self.cyc_img_oth, self.lr_img_oth)

        # compute frequency loss
        #self.loss_frequency   = self.loss_Fre(nn.functional.avg_pool2d(self.sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref) * 10
        #self.loss_frequency   = self.loss_Fre(self.cyc_img_ref, self.lr_img_ref) + \
        #                        self.loss_Fre(self.cyc_img_oth, self.lr_img_oth)

        # compute texture loss
        #lr_prevlayer_ref, _ = self.vgg(self.cyc_img_ref)
        #lr_prevlayer_oth, _ = self.vgg(self.cyc_img_oth)
        #tr_prevlayer_ref, _ = self.vgg(self.lr_img_ref)
        #tr_prevlayer_oth, _ = self.vgg(self.lr_img_oth)

        #self.loss_texture_matching = self.loss_texture(tr_prevlayer_ref, lr_prevlayer_ref) + \
        #                             self.loss_texture(tr_prevlayer_oth, lr_prevlayer_oth)

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(torch.cat([self.hr_img_ref, self.hr_img_oth], dim=0))
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        self.loss_G_D = (-self.net_D(self.hr_img_ref).mean() - self.net_D(self.hr_img_oth).mean())

        self.loss_G =  self.loss_flow \
                    + self.loss_G_D \
                    + self.loss_bp
                    #+ self.loss_sr
                    #+ self.loss_img_smooth \
                    


        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        #self.gradient_clip(['sr', 'flow', 'G1', 'G2'])
        self.optimizer_G.step()

    def gradient_clip(self, names):
        for net in names:
            assert isinstance(net, str)
            net = getattr(self, 'net_' + net)
            torch.nn.utils.clip_grad_value_(net.parameters(), 1)        

    def optimize_D(self):

        #mask= zero_out_pixels(list_shape=list(self.hr_img_real.shape), prop=self.opts.mask_prop)
        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())
        for i in range(self.opts.d_step):
            self.optimizer_D.zero_grad()
            self.loss_D = self.loss_GAN(self.net_D, self.img_real_cen, self.hr_img_ref, use_gp=True) + \
                          self.loss_GAN(self.net_D, self.img_real_ref, self.hr_img_oth, use_gp=True)
            self.loss_D.backward()
            self.optimizer_D.step()

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

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        #self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        #self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        #self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_bp'] = self.loss_bp.item()
        #losses['loss_sr'] = self.loss_sr.item()
        losses['loss_flow'] = self.loss_flow.item()
        losses['loss_G_D'] = self.loss_G_D.item()
        #losses['loss_img_smooth'] = self.loss_img_smooth.item()
        losses['loss_G'] = self.loss_G.item()
        #losses['loss_identity'] = self.loss_identity.item()
        #losses['loss_perceptural'] = self.loss_perceptural.item()
        #losses['loss_frequency'] = self.loss_frequency.item()
        #losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_D'] = self.loss_D.item()
        
        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
        return losses





#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class UnFlowSRGNet(BaseModel):
    def __init__(self, opts):
        super(UnFlowSRGNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['sr', 'flow', 'D', 'G1', 'G2']
        #self.net_flow = PWCDCNet().cuda()
        self.net_sr   = EDSR(opts).cuda()        
        #self.net_sr   = DRLN(opts).cuda()
        #self.net_sr   = RCAN(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D1   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_D2   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_flow)
        self.print_networks(self.net_D1)
        self.print_networks(self.net_D2)
        self.print_networks(self.net_G1)
        self.print_networks(self.net_G2)

        grid = generate_2D_grid(opts.im_crop_H, opts.im_crop_W) #[2, im_crop_H, im_crop_W]
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(opts.batch_size, 1, 1, 1) #[batch_size-1, 2, im_crop_H, im_crop_W]

        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()

        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                self.load_network(self.net_flow, 'Flow', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_sr.parameters()},
                {'params': self.net_G1.parameters()},
                {'params': self.net_G2.parameters()}], lr=opts.lr)
            
            self.optimizer_D1 = torch.optim.Adam([
                {'params': self.net_D1.parameters()}], lr=opts.lr)

            self.optimizer_D2 = torch.optim.Adam([
                {'params': self.net_D2.parameters()}], lr=opts.lr)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=opts.lr_decay)
            #self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D1 = torch.optim.lr_scheduler.StepLR(self.optimizer_D1, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D2 = torch.optim.lr_scheduler.StepLR(self.optimizer_D2, step_size=opts.lr_step, gamma=opts.lr_decay)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP(LAMBDA=self.opts.LAMBDA)
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_bp',              1000))
                f.write('{} : {}\n'.format('loss_bp_deblur',       1000))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       100))
                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     10))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        #self.noise_amp = 0.01


    def forward(self):

        self.hr_img_ref = self.net_sr(self.lr_img_ref) + self.upsample_4(self.lr_img_ref)
        self.hr_img_oth = self.net_sr(self.lr_img_oth) + self.upsample_4(self.lr_img_oth)

        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.hr_img_ref.shape[2], self.hr_img_ref.shape[3]).cuda() * self.noise_amp
        #res_ref = self.net_G1(self.hr_img_ref + noise)
        #res_oth = self.net_G1(self.hr_img_oth + noise)
        #deblur_hr_img_ref = self.hr_img_ref + res_ref
        #deblur_hr_img_oth = self.hr_img_oth + res_oth
        deblur_hr_img_ref = self.net_G1(self.hr_img_ref)
        deblur_hr_img_oth = self.net_G1(self.hr_img_oth)

        self.cyc_img_ref = self.net_G2(deblur_hr_img_ref)
        self.cyc_img_oth = self.net_G2(deblur_hr_img_oth)

        self.flows_ref_to_other = self.net_flow(self.hr_img_ref, self.hr_img_oth)
        self.flows_other_to_ref = self.net_flow(self.hr_img_oth, self.hr_img_ref)

        return deblur_hr_img_ref, deblur_hr_img_oth

    def optimize_G(self):

        self.deblur_hr_img_ref, self.deblur_hr_img_oth = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)
        self.hr_mask_oth = self.mask_fn(self.flows_ref_to_other[0]*20.0)

        # compute synthetic hr images
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth, self.flows_ref_to_other[0]*20.0)
        self.syn_hr_img_oth = self.Backward_warper(self.hr_img_ref, self.flows_other_to_ref[0]*20.0)

        # compute self consistency losses
        self.loss_bp = self.loss_L1(nn.functional.avg_pool2d(self.hr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100 + \
                       self.loss_L1(nn.functional.avg_pool2d(self.hr_img_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True) * 100 + \
                       self.loss_L1(nn.functional.avg_pool2d(self.deblur_hr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100 + \
                       self.loss_L1(nn.functional.avg_pool2d(self.deblur_hr_img_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True) * 100

        # loss_sr included
        lr_img_real_ref = nn.functional.avg_pool2d(self.img_real_ref, kernel_size=self.opts.scale)
        lr_img_real_cen = nn.functional.avg_pool2d(self.img_real_cen, kernel_size=self.opts.scale)
        syn_img_real_ref = self.net_sr(lr_img_real_ref) + self.upsample_4(lr_img_real_ref)
        syn_img_real_cen = self.net_sr(lr_img_real_cen) + self.upsample_4(lr_img_real_cen)
        self.loss_sr = self.loss_L1(syn_img_real_ref, self.img_real_ref, mean=True) + \
                       self.loss_L1(syn_img_real_cen, self.img_real_cen, mean=True)
        self.loss_perceptural_sr = self.loss_content.get_loss(syn_img_real_ref, self.img_real_ref) * 0.01 + \
                                   self.loss_content.get_loss(syn_img_real_cen, self.img_real_cen) * 0.01

        self.loss_img_smooth  = (self.loss_tv_img(self.hr_img_ref, mean=True) + self.loss_tv_img(self.hr_img_oth, mean=True)) * 0.001 + \
                                (self.loss_tv_img(self.deblur_hr_img_ref, mean=True) + self.loss_tv_img(self.deblur_hr_img_oth, mean=True)) * 0.001

        #self.loss_bp_deblur   = self.loss_L1(nn.functional.avg_pool2d(self.deblur_sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100
        self.loss_cyc_xy  = self.loss_L1(self.cyc_img_ref,    self.hr_img_ref, mean=True) * 10 + \
                            self.loss_L1(self.cyc_img_oth,    self.hr_img_oth, mean=True) * 10
        self.loss_cyc_yx  = self.loss_L1(self.net_G1(self.net_G2(self.img_real_cen)),    self.img_real_cen, mean=True) * 10 + \
                            self.loss_L1(self.net_G1(self.net_G2(self.img_real_ref)),    self.img_real_ref, mean=True) * 10

        #self.loss_cyc_deblur = self.loss_L1(self.deblur_hr_img_ref,    self.net_G1(self.cyc_img_ref), mean=True) * 10 + \
        #                       self.loss_L1(self.deblur_hr_img_oth,    self.net_G1(self.cyc_img_oth), mean=True) * 10

        self.loss_identity = self.loss_L1(self.deblur_hr_img_ref, self.net_G1(self.deblur_hr_img_ref), mean=True) * 0.01 + \
                             self.loss_L1(self.deblur_hr_img_oth, self.net_G1(self.deblur_hr_img_oth), mean=True) * 0.01

        # compute left-right consistency loss
        #self.loss_flow = self.loss_L1(nn.functional.avg_pool2d(self.syn_hr_img_ref, kernel_size=self.opts.scale), \
        #                        self.lr_img_ref, nn.functional.avg_pool2d(self.hr_mask_ref.float(), kernel_size=self.opts.scale), mean=True)
        self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref, self.hr_mask_ref, mean=True) + \
                         self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth, self.hr_mask_oth, mean=True)

        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.1
        #self.loss_perceptural = self.loss_content.get_loss(nn.functional.avg_pool2d(self.sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref)
        #self.loss_perceptural = self.loss_content.get_loss(self.cyc_img_ref, self.lr_img_ref) + \
        #                        self.loss_content.get_loss(self.cyc_img_oth, self.lr_img_oth)

        # compute frequency loss
        #self.loss_frequency   = self.loss_Fre(nn.functional.avg_pool2d(self.sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref) * 10
        #self.loss_frequency   = self.loss_Fre(self.cyc_img_ref, self.lr_img_ref) * 10 + \
        #                        self.loss_Fre(self.cyc_img_oth, self.lr_img_oth) * 10

        # compute texture loss
        #lr_prevlayer_ref, _ = self.vgg(self.cyc_img_ref)
        #lr_prevlayer_oth, _ = self.vgg(self.cyc_img_oth)
        #hr_prevlayer_ref, _ = self.vgg(self.lr_img_ref)
        #hr_prevlayer_oth, _ = self.vgg(self.lr_img_oth)

        #self.loss_texture_matching = self.loss_texture(hr_prevlayer_ref, lr_prevlayer_ref) * 1e-2 + \
        #                             self.loss_texture(hr_prevlayer_oth, lr_prevlayer_oth) * 1e-2

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.deblur_sr_img_ref)
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True) * 10
        self.loss_G_D1 = (-self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean()) * 1e-4
        self.loss_G_D2 = (-self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean()) * 1e-4

        self.loss_G =  self.loss_flow \
                    + self.loss_bp \
                    + self.loss_cyc_xy \
                    + self.loss_cyc_yx \
                    + self.loss_G_D1 \
                    + self.loss_G_D2 \
                    + self.loss_img_smooth \
                    + self.loss_identity \
                    + self.loss_sr \
                    + self.loss_perceptural_sr

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        #self.gradient_clip(['sr', 'flow', 'G1', 'G2'])
        self.optimizer_G.step()

    def gradient_clip(self, names):
        for net in names:
            assert isinstance(net, str)
            net = getattr(self, 'net_' + net)
            torch.nn.utils.clip_grad_value_(net.parameters(), 1)        

    def optimize_D(self):

        #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        #real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        #self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())
        
        #for i in range(self.opts.d_step):
        #    self.optimizer_D.zero_grad()
        #    self.loss_D1 = self.loss_GAN(self.net_D1, self.img_real_cen, self.deblur_hr_img_ref, use_gp=True) + \
        #                   self.loss_GAN(self.net_D1, self.img_real_ref, self.deblur_hr_img_oth, use_gp=True)
        #    self.loss_D2 = self.loss_GAN(self.net_D2, self.hr_img_ref.detach(), self.net_G2(self.img_real_cen), use_gp=True) + \
        #                   self.loss_GAN(self.net_D2, self.hr_img_oth.detach(), self.net_G2(self.img_real_ref), use_gp=True)

        #    self.loss_D = self.loss_D1 + self.loss_D2
        #    self.loss_D.backward()
        #    self.optimizer_D.step()

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
        #self.hr_img_real = hr_img_real
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
        self.save_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        self.save_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_flow, 'Flow',  label, self.opts.checkpoint_dir)
        #self.load_network(self.net_D1,   'D1',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_D2,   'D2',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)

    def get_current_scalars(self):
        losses = {}
        losses['loss_flow'] = self.loss_flow.item()
        losses['loss_bp'] = self.loss_bp.item()
        losses['loss_cyc_xy'] = self.loss_cyc_xy.item()
        losses['loss_cyc_yx'] = self.loss_cyc_yx.item()
        losses['loss_img_smooth'] = self.loss_img_smooth.item()
        losses['loss_identity'] = self.loss_identity.item()
        losses['loss_sr'] = self.loss_sr.item()
        losses['loss_perceptural_sr'] = self.loss_perceptural_sr.item()
        losses['loss_G_D1'] = self.loss_G_D1.item()
        losses['loss_G_D2'] = self.loss_G_D2.item()
        losses['loss_G'] = self.loss_G.item()
        losses['loss_D1'] = self.loss_D1.item()
        losses['loss_D2'] = self.loss_D2.item()
        #losses['loss_D'] = self.loss_D.item()
        

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
            losses['PSNR_deblur'] = PSNR(self.deblur_hr_img_ref.data, self.hr_img_ref_gt)
        return losses



#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#




class UnFlowSRG2Net(BaseModel):
    def __init__(self, opts):
        super(UnFlowSRG2Net, self).__init__()
        self.opts = opts
        # create network
        #self.net_sr   = EDSR(opts).cuda()
        self.net_sr   = DRLN(opts).cuda()
        #self.net_sr   = RCAN(opts).cuda()
        self.net_flow = UpPWCDCNet().cuda()
        self.net_D1   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_D2   = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()
        self.net_G2   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats//2).cuda()

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

        #print('grid size', grid.size())
        self.mask_fn = FlowWarpMask(grid)
        self.Backward_warper = Backward_warp()
        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        

        if opts.use_pretrained_model:
            if opts.is_training:
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
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
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

        hr_img_ref = self.net_sr(self.lr_img_ref) + self.upsample_4(self.lr_img_ref)
        hr_img_oth = self.net_sr(self.lr_img_oth) + self.upsample_4(self.lr_img_oth)

        self.flows_ref_to_other = self.net_flow(hr_img_ref, hr_img_oth)
        self.flows_other_to_ref = self.net_flow(hr_img_oth, hr_img_ref)

        return hr_img_ref, hr_img_oth

    def optimize_G_sr(self):

        self.hr_img_ref, self.hr_img_oth = self.forward()

        # compute hr mask & lr mask
        self.hr_mask_ref = self.mask_fn(self.flows_other_to_ref[0]*20.0)
        self.hr_mask_oth = self.mask_fn(self.flows_ref_to_other[0]*20.0)

        # compute synthetic hr images
        self.syn_hr_img_ref = self.Backward_warper(self.hr_img_oth, self.flows_ref_to_other[0]*20.0)
        self.syn_hr_img_oth = self.Backward_warper(self.hr_img_ref, self.flows_other_to_ref[0]*20.0)

        # compute self consistency losses
        #self.loss_self_sr = self.loss_L1(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 10 + \
        #                    self.loss_L1(self.hr_img_oth, self.hr_img_oth_gt, mean=True) * 10
        self.loss_bp_sr = self.loss_L1(nn.functional.avg_pool2d(self.hr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100 + \
                          self.loss_L1(nn.functional.avg_pool2d(self.hr_img_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True) * 100
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
        #self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref_gt, self.hr_mask_ref, mean=True) * 0.5 + \
        #                 self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth_gt, self.hr_mask_oth, mean=True) * 0.5

        self.loss_flow = self.loss_L1(self.syn_hr_img_ref, self.hr_img_ref, self.hr_mask_ref, mean=True) + \
                         self.loss_L1(self.syn_hr_img_oth, self.hr_img_oth, self.hr_mask_oth, mean=True)

        #self.loss_tv_flow(self.flow_ref_to_other, mean=True) * 0.01 + \
        #self.loss_tv_flow(self.flow_other_to_ref, mean=True) * 0.01

        # compute perceptual loss
        #self.loss_perceptural_sr = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #                           self.loss_content.get_loss(self.hr_img_oth, self.hr_img_oth_gt) * 0.01
        #self.loss_content.get_loss(self.deblur_hr_img_ref, self.hr_img_ref_gt) * 0.01 + \
        #self.loss_content.get_loss(self.deblur_hr_img_oth, self.hr_img_oth_gt) * 0.01

        # compute frequency loss
        #self.loss_frequency_sr   = self.loss_Fre(self.hr_img_ref, self.hr_img_ref_gt) + \
        #                        self.loss_Fre(self.hr_img_oth, self.hr_img_oth_gt)
        #self.loss_Fre(self.deblur_hr_img_ref, self.hr_img_ref_gt) + \
        #self.loss_Fre(self.deblur_hr_img_oth, self.hr_img_oth_gt)

        # compute texture loss
        #sr_prevlayer_ref, _ = self.vgg(self.hr_img_ref)
        #sr_prevlayer_oth, _ = self.vgg(self.hr_img_oth)
        #dr_prevlayer_ref, _ = self.vgg(self.deblur_hr_img_ref)
        #dr_prevlayer_oth, _ = self.vgg(self.deblur_hr_img_oth)
        #hr_prevlayer_ref, _ = self.vgg(self.hr_img_ref_gt)
        #hr_prevlayer_oth, _ = self.vgg(self.hr_img_oth_gt)

        #self.loss_texture_matching_sr = self.loss_texture(hr_prevlayer_ref, sr_prevlayer_ref) * 1e-4 + \
        #                                self.loss_texture(hr_prevlayer_oth, sr_prevlayer_oth) * 1e-4
        #self.loss_texture(hr_prevlayer_ref, dr_prevlayer_ref) * 1e-4 + \
        #self.loss_texture(hr_prevlayer_oth, dr_prevlayer_oth) * 1e-4

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(torch.cat([self.deblur_hr_img_ref, self.deblur_hr_img_oth], dim=0))
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        #self.loss_G_D1 = -self.net_D1(self.deblur_hr_img_ref).mean() - self.net_D1(self.deblur_hr_img_oth).mean()
        #self.loss_G_D2 = -self.net_D2(self.net_G2(self.img_real_cen)).mean() - self.net_D2(self.net_G2(self.img_real_ref)).mean()

        self.loss_G_sr =  self.loss_flow + self.loss_bp_sr

        self.optimizer_G_sr.zero_grad()
        self.loss_G_sr.backward()
        self.optimizer_G_sr.step()

    def optimize_G_style(self):
        self.deblur_hr_img_ref = self.net_G1(self.hr_img_ref.detach())
        self.deblur_hr_img_oth = self.net_G1(self.hr_img_oth.detach())
        self.cyc_img_ref = self.net_G2(self.deblur_hr_img_ref)
        self.cyc_img_oth = self.net_G2(self.deblur_hr_img_oth)

        # compute self consistency losses
        self.loss_bp_style = self.loss_L1(nn.functional.avg_pool2d(self.deblur_hr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100 + \
                             self.loss_L1(nn.functional.avg_pool2d(self.deblur_hr_img_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True) * 100
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

        self.loss_G_style = self.loss_bp_style \
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
        losses['loss_bp_sr'] = self.loss_bp_sr.item()
        #losses['loss_identity'] = self.loss_identity.item()
        #losses['loss_perceptural_sr'] = self.loss_perceptural_sr.item()
        #losses['loss_frequency_sr'] = self.loss_frequency_sr.item()
        #losses['loss_texture_matching_sr'] = self.loss_texture_matching_sr.item()
        losses['loss_G_sr'] = self.loss_G_sr.item()

        losses['loss_bp_style'] = self.loss_bp_style.item()
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




class UnSRNet(BaseModel):
    def __init__(self, opts):
        super(UnSRNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.net_sr   = EDSR(opts).cuda()
        self.net_sr   = DRLN(opts).cuda()
        self.net_D    = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        #self.net_G2   = EDLR(opts).cuda()
        self.net_G1   = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()

        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_D)
        self.print_networks(self.net_G1)

        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        
        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training: 
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_sr.parameters()},
                {'params': self.net_G1.parameters()}], lr=opts.lr)
            
            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr/10.0)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=opts.lr_decay)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=opts.lr_decay)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
                f.write('{} : {}\n'.format('loss_bp',              1000))
                f.write('{} : {}\n'.format('loss_bp_deblur',       1000))
                f.write('{} : {}\n'.format('loss_cyc',             1))
                f.write('{} : {}\n'.format('loss_identity',        0.1))
                f.write('{} : {}\n'.format('loss_frequency',       100))
                f.write('{} : {}\n'.format('loss_flow',            1)) #0.5
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     10))

                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        self.noise_amp = 0.01


    def forward(self):

        hr_img_ref = self.net_sr(self.lr_img_ref)
        hr_img_oth = self.net_sr(self.lr_img_oth)

        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.opts.im_crop_H, self.opts.im_crop_W).cuda() * self.noise_amp
        #noise = torch.randn(1, 1, self.opts.im_crop_H, self.opts.im_crop_W).cuda() * self.noise_amp
        noise = generator_noise(list_shape=list(hr_img_ref.shape)) * self.noise_amp
        res_ref   = self.net_G1(hr_img_ref + noise)
        res_oth   = self.net_G1(hr_img_oth + noise)
        hr_img_ref = hr_img_ref + res_ref
        hr_img_oth = hr_img_oth + res_oth

        self.d_mask= zero_out_pixels(list_shape=list(hr_img_ref.shape), prop=self.opts.mask_prop)
        self.d_hr_img_ref = hr_img_ref * self.d_mask
        self.d_hr_img_oth = hr_img_oth * self.d_mask

        #d_hr_img_ref = self.hr_img_ref * self.d_mask
        #d_hr_img_oth = self.hr_img_oth * self.d_mask
        #self.cyc_img_ref = self.net_G2(d_hr_img_ref)
        #self.cyc_img_oth = self.net_G2(d_hr_img_oth)
        #self.cyc_img_ref = nn.functional.avg_pool2d(self.net_G2(hr_img_ref), kernel_size=self.opts.scale)
        #self.cyc_img_oth = nn.functional.avg_pool2d(self.net_G2(hr_img_oth), kernel_size=self.opts.scale)

        return hr_img_ref, hr_img_oth

    def optimize_G(self):

        self.hr_img_ref, self.hr_img_oth = self.forward()
        self.cyc_img_ref = nn.functional.avg_pool2d(self.d_hr_img_ref, kernel_size=self.opts.scale)
        self.cyc_img_oth = nn.functional.avg_pool2d(self.d_hr_img_oth, kernel_size=self.opts.scale)

        # compute self consistency losses
        #self.loss_bp = self.loss_L1(nn.functional.avg_pool2d(self.hr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True)*100 + \
        #               self.loss_L1(nn.functional.avg_pool2d(self.hr_img_oth, kernel_size=self.opts.scale), self.lr_img_oth, mean=True)*100
        #self.loss_bp_deblur   = self.loss_L1(nn.functional.avg_pool2d(self.deblur_sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref, mean=True) * 100
        self.loss_cyc       = self.loss_L1(self.cyc_img_ref,    self.lr_img_ref, mean=True) * 100 + \
                              self.loss_L1(self.cyc_img_oth,    self.lr_img_oth, mean=True) * 100
        
        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.1
        #self.loss_perceptural = self.loss_content.get_loss(nn.functional.avg_pool2d(self.sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref)
        self.loss_perceptural = self.loss_content.get_loss(self.cyc_img_ref, self.lr_img_ref) + \
                                self.loss_content.get_loss(self.cyc_img_oth, self.lr_img_oth)

        # compute frequency loss
        #self.loss_frequency   = self.loss_Fre(nn.functional.avg_pool2d(self.sr_img_ref, kernel_size=self.opts.scale), self.lr_img_ref) * 10
        self.loss_frequency   = self.loss_Fre(self.cyc_img_ref, self.lr_img_ref) + \
                                self.loss_Fre(self.cyc_img_oth, self.lr_img_oth)

        # compute texture loss
        lr_prevlayer_ref, _ = self.vgg(self.cyc_img_ref)
        lr_prevlayer_oth, _ = self.vgg(self.cyc_img_oth)
        tr_prevlayer_ref, _ = self.vgg(self.lr_img_ref)
        tr_prevlayer_oth, _ = self.vgg(self.lr_img_oth)

        self.loss_texture_matching = self.loss_texture(tr_prevlayer_ref, lr_prevlayer_ref) + \
                                     self.loss_texture(tr_prevlayer_oth, lr_prevlayer_oth)

        # compute GAN loss
        dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(torch.cat([self.hr_img_ref, self.hr_img_oth], dim=0))
        self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True) * 100

        self.loss_G = self.loss_cyc \
                    + self.loss_G_D \
                    + self.loss_perceptural \
                    + self.loss_texture_matching \
                    + self.loss_frequency

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        #self.gradient_clip(['sr', 'flow', 'G1', 'G2'])
        self.optimizer_G.step()

    def gradient_clip(self, names):
        for net in names:
            assert isinstance(net, str)
            net = getattr(self, 'net_' + net)
            torch.nn.utils.clip_grad_value_(net.parameters(), 1)        

    def optimize_D(self):

        #mask= zero_out_pixels(list_shape=list(self.hr_img_real.shape), prop=self.opts.mask_prop)
        dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
        self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        #self.gradient_clip(['D'])
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
        self.noise_amp = self.noise_amp * 0.995

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        #self.save_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        #self.load_network(self.net_flow, 'Flow', label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G1,   'G1',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G2,   'G2',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_flow'] = self.loss_flow.item()
        #losses['loss_bp'] = self.loss_bp.item()
        #losses['loss_bp_deblur'] = self.loss_bp_deblur.item()
        losses['loss_cyc'] = self.loss_cyc.item()
        #losses['loss_identity'] = self.loss_identity.item()
        losses['loss_perceptural'] = self.loss_perceptural.item()
        losses['loss_frequency'] = self.loss_frequency.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()        
        losses['loss_D'] = self.loss_D.item()
        losses['loss_G'] = self.loss_G.item()

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
        return losses
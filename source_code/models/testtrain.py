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

class TestSRNet(BaseModel):
    def __init__(self, opts):
        super(TestSRNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['sr', 'D']
        self.net_sr = EDSR(opts).cuda()
        #self.net_sr = DRLN(opts).cuda()
        #self.net_sr = EEDSR(opts).cuda()
        self.net_D = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=False).cuda()
        #self.net_D = PatchGAN_net(input_nc=opts.n_colors).cuda()
        self.net_G = GeneratorConcatSkip2CleanAdd(opts, num_layer=1).cuda()
        #self.net_G = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=5).cuda()
        # print network
        self.print_networks(self.net_sr)
        self.print_networks(self.net_D)
        self.print_networks(self.net_G)

        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        

        if opts.use_pretrained_model:
            if opts.is_training:
                self.load_network(self.net_sr, 'SR', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_G.parameters()},
                {'params': self.net_sr.parameters()}], lr=opts.lr)
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

            self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')

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
        self.noise_amp = 0.01

    def forward(self):

        
        hr_img_ref  = self.net_sr(self.lr_img_ref) + self.upsample_4(self.lr_img_ref)
        noise = torch.randn(self.opts.batch_size, self.opts.n_colors, hr_img_ref.shape[2], hr_img_ref.shape[3]).cuda() * self.noise_amp
        #hr_other_imgs = self.net_sr(lr_other_imgs)
        res   = self.net_G(hr_img_ref + noise)
        #self.res = self.net_G(self.noise)
        hr_img_ref = hr_img_ref + res

        #output = self.hr_img_ref.cpu()[0].permute(1,2,0).detach().numpy()
        #output[output>1]=1
        #output[output<-1]=-1
        #output = skimage.img_as_ubyte(output)
        #skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'SR_{}.png'.format('chain_EDSR')), output)

        #hr_img_ref  = self.net_sr(self.lr_img_ref)
        return hr_img_ref

    def optimize_G(self):
        self.hr_img_ref = self.forward()

        # compute self consistency losses
        self.loss_self = self.loss_data(self.hr_img_ref, self.hr_img_ref_gt, mean=True) * 100
        #self.loss_self = self.loss_data(self.synthesis_output, self.hr_img_ref_gt, mean=True) * 10 + \
        #                 self.loss_data(self.hr_img_ref,       self.hr_img_ref_gt, mean=True) * 10

        # compute smoothness loss
        #self.loss_img_smooth  = (self.loss_tv_img(self.hr_img_ref, mean=True) + self.loss_tv_img(self.hr_img_others, mean=True)) * 0.01
        #self.loss_img_smooth  = self.loss_tv_img(self.hr_img_ref, mean=True) * 0.1
        
        # compute perceptual loss
        self.loss_perceptural = self.loss_content.get_loss(self.hr_img_ref, self.hr_img_ref_gt) * 0.02
        #self.loss_perceptural = self.loss_content.get_loss(self.synthesis_output, self.hr_img_ref_gt) * 0.04 + \
        #                        self.loss_content.get_loss(self.hr_img_ref,       self.hr_img_ref_gt) * 0.04

        # compute texture matching loss
        sr_prevlayer, _ = self.vgg(self.hr_img_ref)
        hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-5
        
        #sr_prevlayer, _ = self.vgg(self.hr_img_ref)
        #op_prevlayer, _ = self.vgg(self.synthesis_output)
        #hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        #self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-3 + \
        #                             self.loss_texture(hr_prevlayer, op_prevlayer) * 1e-3

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.hr_img_ref)
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)
        self.loss_G_D = -self.net_D(self.hr_img_ref).mean()

        self.loss_G = self.loss_self \
                    + self.loss_G_D \
                    + self.loss_perceptural \
                    + self.loss_texture_matching
                    #+ self.loss_img_smooth
                    #+ self.loss_self_others \

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
            self.loss_D = self.loss_GAN(self.net_D, self.hr_img_real, self.hr_img_ref, use_gp=True)
            self.loss_D.backward()
            self.optimizer_D.step()
        #self.scheduler_D.step()
        #for param_group in self.optimizer_D.param_groups:
        #    print('D ', param_group['lr'])

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
        #self.noise_amp = self.noise_amp/2.0

    def save_checkpoint(self, label):
        self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.load_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
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


class TestCircleSRNet(BaseModel):
    def __init__(self, opts):
        super(TestCircleSRNet, self).__init__()
        self.opts = opts
 
        # create network
        self.model_names = ['Feature_Head', 'Feature_extractor', 'D', 'Upscalar', 'Downscalar', 'G']
        self.net_Feature_Head = Feature_Head(opts).cuda()
        self.net_Feature_extractor = Feature_extractor(opts).cuda()
        self.net_Upscalar = Upscalar(opts).cuda()
        self.net_Downscalar = Downscalar(opts).cuda()

        self.net_D = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        self.net_Ghr = GeneratorConcatSkip2CleanAdd(opts, num_layer=1).cuda()
        #self.net_Ghr = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=5).cuda()
        self.net_Glr = GeneratorConcatSkip2CleanAdd(opts, num_layer=1).cuda()

        # print network
        self.print_networks(self.net_Feature_Head)
        self.print_networks(self.net_Feature_extractor)
        self.print_networks(self.net_Upscalar)
        self.print_networks(self.net_Downscalar)
        self.print_networks(self.net_D)
        self.print_networks(self.net_Ghr)
        self.print_networks(self.net_Glr)

        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)

        if opts.use_pretrained_model:
            self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_Feature_Head.parameters()},
                {'params': self.net_Feature_extractor.parameters()},
                {'params': self.net_Upscalar.parameters()},
                {'params': self.net_Ghr.parameters()},
                {'params': self.net_Glr.parameters()},
                {'params': self.net_Downscalar.parameters()}], lr=opts.lr)

            self.optimizer_D = torch.optim.Adam([
                {'params': self.net_D.parameters()}], lr=opts.lr/10.0)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_data = L1()
            self.loss_tv_img = VariationLoss(nc=3, grad_fn=grid_gradient_central_diff)
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.vgg = VGG19(final_layer='relu_5-1', prev_layer=['relu_1-1', 'relu_2-1', 'relu_3-1'], pretrain=True).cuda()
            self.loss_texture = TextureLoss()

            #with open(os.path.join(self.opts.checkpoint_dir, 'log.txt'), 'a') as f:
            #    f.write('{} : {}\n'.format('loss_self_ref',        10))               
            #    f.write('{} : {}\n'.format('loss_self_others',     10))
            #    f.write('{} : {}\n'.format('loss_G_D',             1))
            #    f.write('{} : {}\n'.format('loss_D',               1))
            #    f.write('{} : {}\n'.format('loss_perceptural',     0.1))
            #    f.write('{} : {}\n'.format('loss_img_smooth',      0.01))
            #    f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        #self.hr_img_gt = None
        self.noise_amp = 0.001

    def forward(self):
        lr_feature_head    = self.net_Feature_Head(self.lr_img_ref)
        lr_content_feature = self.net_Feature_extractor(lr_feature_head)
        lr_content_output = lr_feature_head + lr_content_feature
        hr_img = self.net_Upscalar(lr_content_output)

        noise = torch.randn(self.opts.batch_size, self.opts.n_colors, hr_img.shape[2], hr_img.shape[3]).cuda() * self.noise_amp
        res   = self.net_Ghr(hr_img + noise)
        hr_img = hr_img + res
        
        #hr_feature_head    = self.net_Feature_Head(hr_img)
        hr_feature_head    = self.net_Feature_Head(self.hr_img_ref_gt) #TODO change it!
        hr_content_feature = self.net_Feature_extractor(hr_feature_head)
        hr_content_output = hr_feature_head + hr_content_feature
        lr_img_syn = self.net_Downscalar(hr_content_output)
        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, lr_img_syn.shape[2], lr_img_syn.shape[3]).cuda() * self.noise_amp
        res   = self.net_Glr(lr_img_syn)
        lr_img_syn = lr_img_syn + res

        return hr_img, lr_img_syn

    def optimize_G(self):
        
        #self.lr_img_ref = torch.cat([lr_img_ref, lr_img_others], dim=0)

        #self.lr_img_ref     = lr_img_ref
        #self.lr_img_others  = lr_img_others
        #self.hr_img_ref, self.hr_img_others = self.forward(self.lr_img_ref, self.lr_img_others)
        self.hr_img, self.lr_img_syn = self.forward()

        # compute self consistency losses
        #self.loss_self_ref = self.loss_data(self.hr_img_ref_gt, self.hr_img_ref, mean=True) * 10
        #self.loss_self_others = self.loss_data(self.hr_img_oth_gt, self.hr_img_others, mean=True) * 10
        self.loss_consistent_hr = self.loss_data(self.hr_img, self.hr_img_ref_gt, mean=True) * 100
        self.loss_consistent_lr = self.loss_data(self.lr_img_syn, self.lr_img_ref, mean=True) * 7

        # compute smoothness loss
        self.loss_img_smooth  = (self.loss_tv_img(self.hr_img, mean=True) + self.loss_tv_img(self.lr_img_syn, mean=True)) * 0.01
        
        # compute perceptual loss
        self.loss_perceptural_hr = self.loss_content.get_loss(self.hr_img, self.hr_img_ref_gt) * 0.1
        self.loss_perceptural_lr = self.loss_content.get_loss(self.lr_img_syn, self.lr_img_ref) * 0.1

        # compute texture matching loss
        sr_prevlayer, _ = self.vgg(self.hr_img)
        hr_prevlayer, _ = self.vgg(self.hr_img_ref_gt)
        self.loss_texture_matching = self.loss_texture(hr_prevlayer, sr_prevlayer) * 1e-4

        # compute GAN loss
        dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.hr_img)
        self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)

        self.loss_G = self.loss_consistent_hr \
                    + self.loss_consistent_lr \
                    + self.loss_perceptural_hr \
                    + self.loss_perceptural_lr \
                    + self.loss_img_smooth \
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
        dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = self.gradient_fn(self.hr_img_real)
        real_hr_image = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)

        self.loss_D = self.loss_GAN(self.net_D, real_hr_image, self.fake_hr_image.detach())

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()
        #self.scheduler_D.step()
        #for param_group in self.optimizer_D.param_groups:
        #    print('D ', param_group['lr'])

    #def set_ground_truth(self, hr_img_ref_gt, hr_img_oth_gt):
    #    #self.hr_img_ref_gt = hr_img_ref_gt
    #    #self.hr_img_oth_gt = hr_img_oth_gt
    #    self.hr_img_gt = torch.cat([hr_img_ref_gt, hr_img_oth_gt], dim=0)

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
        self.noise_amp = self.noise_amp/2.0

    def save_checkpoint(self, label):
        self.save_network(self.net_Feature_Head,        'Feature_Head',        label, self.opts.checkpoint_dir)
        self.save_network(self.net_Feature_extractor,   'Feature_extractor',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_Upscalar,            'Upscalar',            label, self.opts.checkpoint_dir)
        self.save_network(self.net_Downscalar,          'Downscalar',          label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,                   'D',                   label, self.opts.checkpoint_dir)
        self.save_network(self.net_G,                   'G',                   label, self.opts.checkpoint_dir)
        
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
        losses['loss_img_smooth'] = self.loss_img_smooth.item()
        losses['loss_perceptural_hr'] = self.loss_perceptural_hr.item()
        losses['loss_perceptural_lr'] = self.loss_perceptural_lr.item()
        losses['loss_texture_matching'] = self.loss_texture_matching.item()
        losses['loss_G_D'] = self.loss_G_D.item()
        losses['loss_D'] = self.loss_D.item()
        losses['loss_G'] = self.loss_G.item()

        if self.hr_img_ref_gt is not None:
            losses['PSNR'] = PSNR(self.hr_img.data, self.hr_img_ref_gt)
        return losses




#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

class TestGNet(BaseModel):
    def __init__(self, opts):
        super(TestGNet, self).__init__()
        self.opts = opts
 
        # create network
        #self.model_names = ['sr', 'D']
        #self.net_sr = EDSR(opts).cuda()
        #self.net_D = WGANGP_net(input_nc=opts.n_colors, use_sigmoid=True).cuda()
        self.net_D = PatchGAN_net(input_nc=opts.n_colors).cuda()
        #self.net_G = GeneratorConcatSkip2CleanAdd(opts, num_layer=3).cuda()
        self.net_G = UnetGenerator(input_nc=opts.n_colors, output_nc=opts.n_colors, ngf=opts.n_feats, num_downs=6).cuda()
        # print network
        #self.print_networks(self.net_sr)
        self.print_networks(self.net_D)
        self.print_networks(self.net_G)

        self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        
        if opts.use_pretrained_model:
            if opts.is_training:
                self.load_network(self.net_sr, 'SR', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_G.parameters()}], lr=opts.lr)

            self.optimizer_D = torch.optim.Adam([{'params': self.net_D.parameters()}], lr=opts.lr)

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
                f.write('{} : {}\n'.format('loss_self_ref',        10))               
                f.write('{} : {}\n'.format('loss_self_others',     10))
                f.write('{} : {}\n'.format('loss_G_D',             1))
                f.write('{} : {}\n'.format('loss_D',               1))
                f.write('{} : {}\n'.format('loss_perceptural',     0.1))
                f.write('{} : {}\n'.format('loss_img_smooth',      0.01))
                f.write('\n')
            
        self.hr_img_ref_gt = None
        self.hr_img_oth_gt = None
        self.noise_amp = 0.0001

    def forward(self):

        noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.opts.im_crop_H, self.opts.im_crop_W).cuda() * self.noise_amp
        hr_img_ref = self.net_G(noise)
        self.d_mask= zero_out_pixels(list_shape=list(hr_img_ref.shape), prop=self.opts.mask_prop)
        return hr_img_ref

    def optimize_G(self):
        self.hr_img_ref = self.forward()

        #self.cyc_img = nn.functional.avg_pool2d(self.hr_img_ref*self.d_mask, kernel_size=self.opts.scale)

        # compute self consistency losses
        #self.loss_cyc = self.loss_L1(self.cyc_img, self.lr_img_ref, mean=True) * 100
        self.loss_cyc = self.loss_L1(self.hr_img_ref * self.d_mask, self.hr_img_ref_gt * self.d_mask, mean=True) * 100
        
        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.cyc_img, self.lr_img_ref)

        # compute frequency loss
        #self.loss_frequency   = self.loss_Fre(self.cyc_img, self.lr_img_ref)

        # compute texture loss
        #lr_prevlayer_ref, _ = self.vgg(self.cyc_img)
        #tr_prevlayer_ref, _ = self.vgg(self.lr_img_ref)

        #self.loss_texture_matching = self.loss_texture(tr_prevlayer_ref, lr_prevlayer_ref)

        # compute GAN loss
        dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.hr_img_ref)
        self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)

        self.loss_G = self.loss_cyc \
                    + self.loss_G_D
                    #+ self.loss_perceptural \
                    #+ self.loss_texture_matching
                    #+ self.loss_img_smooth
                    #+ self.loss_self_others \

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
        #self.noise_amp = self.noise_amp*0.999

    def save_checkpoint(self, label):
        #self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        #self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_self_ref'] = self.loss_self_ref.item()
        #losses['loss_self_others'] = self.loss_self_others.item()
        #losses['loss_img_smooth'] = self.loss_img_smooth.item()
        losses['loss_cyc'] = self.loss_cyc.item()
        #losses['loss_perceptural'] = self.loss_perceptural.item()
        #losses['loss_texture_matching'] = self.loss_texture_matching.item()
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

class TestContinueNet(BaseModel):
    def __init__(self, opts):
        super(TestContinueNet, self).__init__()
        self.opts = opts
 
        self.net_G = GradualGenerator(n_blocks=30).cuda()

        #self.print_networks(self.net_D)
        self.print_networks(self.net_G)

        #self.gradient_fn = grid_gradient_central_diff(nc=opts.n_colors, diagonal=True)
        
        if opts.use_pretrained_model:
            if opts.is_training:
                self.load_network(self.net_sr, 'SR', opts.epoch_to_load, self.opts.model_dir)
            else:
                self.load_checkpoint(opts.epoch_to_load)

        if opts.is_training:
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_G.parameters()}], lr=opts.lr)

            #self.optimizer_D = torch.optim.Adam([{'params': self.net_D.parameters()}], lr=opts.lr)

            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opts.lr_step, gamma=0.618)
            #self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opts.lr_step, gamma=0.618)

            # define loss functions
            self.loss_L1 = L1()
            self.loss_L2 = L2()
            self.loss_GAN = DiscLossWGANGP()
            self.loss_content = PerceptualLoss()
            self.loss_Fre = FreqLoss()
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
            
        #self.hr_img_ref_gt = None
        #self.hr_img_oth_gt = None
        #self.noise_amp = 0.0001
        self.gt_list = None

    def forward(self):

        #noise = torch.randn(self.opts.batch_size, self.opts.n_colors, self.opts.im_crop_H, self.opts.im_crop_W).cuda() * self.noise_amp
        #hr_img_ref = self.net_G(noise)
        #self.d_mask= zero_out_pixels(list_shape=list(hr_img_ref.shape), prop=self.opts.mask_prop)
        #return hr_img_ref
        self.out = []
        #self.out.append(self.net_G.upscale_layers[0](self.lr_list[0]))
        for i in range(0, self.net_G.n_blocks):
            if i==0:
                x = torch.cat([self.lr_list[i], self.lr_list[i]], dim=1)
                self.out.append(self.net_G.upscale_layers[i](x))
            else:
                x = torch.cat([self.out[-1], self.lr_list[i]], dim=1)
                self.out.append(self.net_G.upscale_layers[i](x))
        #self.out.append(self.net_G.upscale_layers[-1](self.out[-1]))
        return self.out

    def optimize_G(self):
        #self.hr_img_ref = self.forward()
        self.out = self.forward()

        #self.cyc_img = nn.functional.avg_pool2d(self.hr_img_ref*self.d_mask, kernel_size=self.opts.scale)

        # compute self consistency losses
        #self.loss_cyc = self.loss_L1(self.cyc_img, self.lr_img_ref, mean=True) * 100
        #self.loss_cyc = self.loss_L1(self.hr_img_ref * self.d_mask, self.hr_img_ref_gt * self.d_mask, mean=True) * 100
        self.loss_self = 0
        for i in range(self.net_G.n_blocks):
            self.loss_self += self.loss_L1(self.out[i], self.gt_list[i], mean=True)
        
        # compute perceptual loss
        #self.loss_perceptural = self.loss_content.get_loss(self.cyc_img, self.lr_img_ref)

        # compute frequency loss
        #self.loss_frequency   = self.loss_Fre(self.cyc_img, self.lr_img_ref)

        # compute texture loss
        #lr_prevlayer_ref, _ = self.vgg(self.cyc_img)
        #tr_prevlayer_ref, _ = self.vgg(self.lr_img_ref)

        #self.loss_texture_matching = self.loss_texture(tr_prevlayer_ref, lr_prevlayer_ref)

        # compute GAN loss
        #dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image = self.gradient_fn(self.hr_img_ref)
        #self.fake_hr_image = torch.cat([dx_fake_hr_image, dy_fake_hr_image, dxy_fake_hr_image], dim=0)
        #self.loss_G_D = self.loss_GAN.get_g_loss(self.net_D(self.fake_hr_image), target_is_real=True)

        self.loss_G = self.loss_self
        #self.loss_G = self.loss_cyc \
        #            + self.loss_G_D
                    #+ self.loss_perceptural \
                    #+ self.loss_texture_matching
                    #+ self.loss_img_smooth
                    #+ self.loss_self_others \

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

    def set_ground_truth(self, gt_list):
        self.gt_list = gt_list

    def set_train_data(self, lr_list):
        self.lr_list = lr_list

    def optimize(self):
        self.optimize_G()
        #self.optimize_D()

    def update_lr(self):
        self.scheduler_G.step()
        #self.scheduler_D.step()
        #self.noise_amp = self.noise_amp*0.999

    def save_checkpoint(self, label):
        #self.save_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        #self.save_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        self.save_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def load_checkpoint(self, label):
        #self.load_network(self.net_sr,   'SR',   label, self.opts.checkpoint_dir)
        self.load_network(self.net_D,    'D',    label, self.opts.checkpoint_dir)
        #self.load_network(self.net_G,    'G',    label, self.opts.checkpoint_dir)
        
    def get_current_scalars(self):
        losses = {}
        #losses['loss_self_ref'] = self.loss_self_ref.item()
        #losses['loss_self_others'] = self.loss_self_others.item()
        #losses['loss_img_smooth'] = self.loss_img_smooth.item()
        #losses['loss_cyc'] = self.loss_cyc.item()
        #losses['loss_perceptural'] = self.loss_perceptural.item()
        #losses['loss_texture_matching'] = self.loss_texture_matching.item()
        
        #losses['loss_G_D'] = self.loss_G_D.item()
        #losses['loss_D'] = self.loss_D.item()
        losses['loss_self'] = self.loss_self.item()

        #if self.hr_img_ref_gt is not None:
        #    losses['PSNR'] = PSNR(self.synthesis_output.data, self.hr_img_ref_gt)
        #    losses['PSNR'] = PSNR(self.hr_img_ref.data, self.hr_img_ref_gt)
        return losses
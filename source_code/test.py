import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import skimage
import torchvision
from PIL import Image
from option import get_arguments

from models.supervised import *
from models.unsupervised import *
from models.testtrain import *
import utils.image_transform as transforms

from utils.metrics import PSNR, SSIM
from networks.flow import Backward_warp
from models.losses import *
from dataloaders.loading_satellite_image import Dataset_Loader
from dataloaders.loading_unsupervised import Unsupervised_Dataset_Loader
from dataloaders.loading_test import Test_Loader

def test(args, model, test_dataloader):

    PSNR_total = []
    SSIM_total = []
    #model.eval()
    print('=====> test sr begin!')
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            #torch.Size([1, 3, 320, 320])
            img_ref = data['image_center']
            img_oth = data['image_others']
            #img_oth = torch.squeeze(img_oth)
            img_adv_cen = data['img_adv_cen']
            img_adv_ref = data['img_adv_ref']
            img_oth = img_oth.squeeze(0)
            img_adv_ref = img_adv_ref.squeeze(0)
            img_ref = img_ref.expand(args.batch_size, -1, -1, -1)
            img_adv_cen = img_adv_cen.expand(args.batch_size, -1, -1, -1)

            image_others = (img_oth.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
            
            #print(img_ref.shape)
            #image_ref = img_ref.expand(args.batch_size-1, -1, -1, -1)
            #image_ref = (image_ref.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
            image_ref = (img_ref.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
            lr_image_ref = nn.functional.avg_pool2d(image_ref, kernel_size=args.scale)
            lr_image_others = nn.functional.avg_pool2d(image_others, kernel_size=args.scale)
            image_adv_cen = img_adv_cen.cuda().clone().float()
            image_adv_ref = img_adv_ref.cuda().clone().float()

            '''
            hr_val = model.net_sr(lr_image_ref)
            hr_ref = model.net_sr(lr_image_others)
            #flows_ref_to_other = model.net_flow(image_ref, image_others)
            flows_ref_to_other = model.net_flow(hr_val, hr_ref)
            #flows_other_to_ref = model.net_flow(image_others, image_ref)
            #flow_12_1 = flows_ref_to_other[0]*20.0
            #flow_12_2 = flows_ref_to_other[1]*10.0
            #flow_12_3 = flows_ref_to_other[2]*5.0
            #flow_12_4 = flows_ref_to_other[3]*2.5
            #SR_conv1, SR_conv2, SR_conv3, SR_conv4 = model.net_enc(hr_val)
            #HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = model.net_enc(hr_ref)

            #warp_21_conv1 = model.Backward_warper(HR2_conv1, flow_12_1)
            #warp_21_conv2 = model.Backward_warper(HR2_conv2, flow_12_2)
            #warp_21_conv3 = model.Backward_warper(HR2_conv3, flow_12_3)
            #warp_21_conv4 = model.Backward_warper(HR2_conv4, flow_12_4)

            #hr_val = model.net_dec(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)
            #hr_val = model.net_G1(hr_val, flows_ref_to_other, model.Backward_warper, image_others)
            hr_val = model.net_G1(hr_val, flows_ref_to_other, model.Backward_warper, hr_ref)
            #print(hr_val.min(), hr_val.max())
            '''
            #hr_val = model.net_sr(lr_image_ref) + model.upsample_4(lr_image_ref)
            #hr_ref = model.net_sr(lr_image_others) + model.upsample_4(lr_image_others)
            #flows_ref_to_other = model.net_flow(hr_val, hr_ref)
            #hr_val = model.net_G1(hr_val, flows_ref_to_other, model.Backward_warper, hr_ref)

            #noise  = torch.randn(args.batch_size, args.n_colors, args.im_crop_H, args.im_crop_W).cuda() * 1e-4
            #hr_val = model.net_sr(image_ref)
            #hr_val = model.net_sr(image_ref) + model.upsample_4(image_ref)
            #hr_val = model.net_sr(image_ref)
            #hr_val = model.net_G1(hr_val)
            #hr_val = model.net_G2(image_adv_cen)
            #res    = model.net_G(hr_val)
            #res    = model.net_G1(hr_val)
            #hr_val = hr_val + res

            #hr_other_imgs = self.net_sr(lr_other_imgs)
            #hr_val = model.net_sr(lr_image_ref)
            #noise  = torch.randn(args.batch_size, args.n_colors, args.im_crop_H, args.im_crop_W).cuda() * 0.0001
            #hr_val = hr_val + model.net_G(hr_val)
            #hr_val = model.net_G1(hr_val)

            lr_feature_head    = model.net_Feature_Head(lr_image_ref)
            lr_content_feature = model.net_Feature_extractor(lr_feature_head)
            lr_content_output = lr_feature_head + lr_content_feature
            hr_val = model.net_Upscalar(lr_content_output)
            hr_val = model.net_G1(hr_val)


            hr_val_numpy = hr_val.cpu()[0].permute(1,2,0).numpy()
            hr_val_numpy[hr_val_numpy>1]=1
            hr_val_numpy[hr_val_numpy<-1]=-1

            img_sr = skimage.img_as_ubyte(hr_val_numpy)
            skimage.io.imsave(os.path.join(args.result_dir, 'tempo', 'SR_{}.png'.format(i)), img_sr)
            #skimage.io.imsave(os.path.join(args.result_dir, 'tempo', 'SR_{}.png'.format(i)), hr_val_numpy)

            if args.have_gt:
                PSNR_value = PSNR(hr_val.data, image_ref)
                SSIM_value = SSIM(hr_val.data, image_ref)
                PSNR_total.append(PSNR_value)
                SSIM_total.append(SSIM_value)

                print('PSNR: {} for patch {}'.format(PSNR_value, i))
                print('SSIM: {} for patch {}'.format(SSIM_value, i))
                print('Average PSNR: {} for {} patches'.format(sum(PSNR_total) / len(PSNR_total), i))
                print('Average SSIM: {} for {} patches'.format(sum(SSIM_total) / len(SSIM_total), i))

            if args.save_result:
                os.makedirs(os.path.join(args.result_dir, 'HR'), exist_ok=True)
                os.makedirs(os.path.join(args.result_dir, 'LR'), exist_ok=True)
                os.makedirs(os.path.join(args.result_dir, 'REF'), exist_ok=True)
                os.makedirs(os.path.join(args.result_dir, 'ADV_CEN'), exist_ok=True)
                os.makedirs(os.path.join(args.result_dir, 'ADV_REF'), exist_ok=True)

                #img_gt = skimage.img_as_float(torch.squeeze(img_ref).permute(1,2,0).numpy())
                img_gt = skimage.img_as_ubyte(torch.squeeze(img_ref).permute(1,2,0).numpy())
                skimage.io.imsave(os.path.join(args.result_dir, 'HR', '{}.png'.format(i)), img_gt)
                skimage.io.imsave(os.path.join(args.result_dir, 'HR', '{}.png'.format(i)), img_gt)
                
                img_lr = skimage.img_as_ubyte(lr_image_ref.cpu()[0].permute(1,2,0).numpy())
                skimage.io.imsave(os.path.join(args.result_dir, 'LR', '{}.png'.format(i)), img_lr)
                skimage.io.imsave(os.path.join(args.result_dir, 'LR', '{}.png'.format(i)), img_lr)
                
                img_adv_center = skimage.img_as_ubyte(image_adv_cen.cpu()[0].permute(1,2,0).numpy())
                skimage.io.imsave(os.path.join(args.result_dir, 'ADV_CEN', '{}.png'.format(i)), img_adv_center)

                for j in range(args.batch_size):
                    os.makedirs(os.path.join(args.result_dir, 'ADV_REF', '{}'.format(j)), exist_ok=True)
                    img_adv_reference = skimage.img_as_ubyte(image_adv_ref.cpu()[j].permute(1,2,0).numpy())
                    skimage.io.imsave(os.path.join(args.result_dir, 'ADV_REF', '{}'.format(j), '{}.png'.format(i)), img_adv_reference)

                    os.makedirs(os.path.join(args.result_dir, 'REF', '{}'.format(j)), exist_ok=True)
                    img_reference = skimage.img_as_ubyte(image_others.cpu()[j].permute(1,2,0).numpy())
                    skimage.io.imsave(os.path.join(args.result_dir, 'REF', '{}'.format(j), '{}.png'.format(i)), img_reference)

            
def test_lr(args, model, test_dataloader):
        
    #model.eval()
    print('=====> test existing lr begin!')
    PSNR_total = []
    SSIM_total = []

    fake_total = []
    real_total = []
    Loss_function = GANLoss()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            
            img_lr = data['lr_image']
            img_lr = img_lr.expand(args.batch_size, -1, -1, -1)
            img_lr = img_lr.cuda().clone().float()


            #hr_val = model.net_sr(img_lr)
            #flows_ref_to_other = model.net_flow(self.hr_img_ref_gt, self.hr_img_oth_gt)
            #flows_other_to_ref = model.net_flow(self.hr_img_oth_gt, self.hr_img_ref_gt)
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

            #lr_feature_head    = model.net_Feature_Head(img_lr)
            #lr_content_feature = model.net_Feature_extractor(lr_feature_head)
            #lr_content_output = lr_feature_head + lr_content_feature
            #hr_val = model.net_Upscalar(lr_content_output)

            hr_val = model.net_sr(img_lr) + model.upsample_4(img_lr)
            #hr_val = model.upsample_4(img_lr)
            #hr_val = model.net_sr(img_lr)
            #noise  = torch.randn(args.batch_size, args.n_colors, args.im_crop_H, args.im_crop_W).cuda() * 1e-4
            #hr_val = hr_val + model.net_G1(hr_val)
            hr_val = model.net_G1(hr_val)
            #hr_val = model.net_G1(hr_val)
            #hr_val = model.net_G2(hr_val)
            #m = nn.Upsample(size=[args.im_crop_H*3, args.im_crop_W*3],mode='bilinear',align_corners=True)
            #hr_val = m(hr_val)
            hr_val_numpy = hr_val.cpu()[0].permute(1,2,0).numpy()
            hr_val_numpy[hr_val_numpy>1]=1
            hr_val_numpy[hr_val_numpy<-1]=-1
            
            img_sr = skimage.img_as_ubyte(hr_val_numpy)
            skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'SR_{}.png'.format(i)), img_sr)
            #skimage.io.imsave(os.path.join(args.result_dir, 'SR_{}.png'.format(i)), img_sr)

            #dx_hr_img_fake, dy_hr_img_fake, dxy_hr_img_fake = model.gradient_fn(hr_val)
            #hr_img_fake = torch.cat([dx_hr_img_fake, dy_hr_img_fake, dxy_hr_img_fake], dim=0)
            #fake = model.net_D(hr_img_fake)
            #fake = Loss_function(fake, target_is_real=False)

            #print('fake: {} for patch {}'.format(fake, i))
            #fake_total.append(fake)
            #print('Average fake: {} for {} patches'.format(sum(fake_total) / len(fake_total), i))

            if args.have_gt:
                
                img_hr = data['hr_image']
                img_hr = img_hr.expand(args.batch_size, -1, -1, -1)
                img_hr = img_hr.cuda().clone().float()

                #dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real = model.gradient_fn(img_hr)
                #hr_img_real = torch.cat([dx_hr_img_real, dy_hr_img_real, dxy_hr_img_real], dim=0)
                #real = model.net_D(hr_img_real)
                #real = Loss_function(real, target_is_real=True)
                
                #print('real: {} for patch {}'.format(real, i))
                #real_total.append(real)
                #print('Average real: {} for {} patches'.format(sum(real_total) / len(real_total), i))

                PSNR_value = PSNR(hr_val.data, img_hr)
                SSIM_value = SSIM(hr_val.data, img_hr)

                PSNR_total.append(PSNR_value)
                SSIM_total.append(SSIM_value)

                print('PSNR: {} for patch {}'.format(PSNR_value, i))
                print('SSIM: {} for patch {}'.format(SSIM_value, i))
                print('Average PSNR: {} for {} patches'.format(sum(PSNR_total) / len(PSNR_total), i))
                print('Average SSIM: {} for {} patches'.format(sum(SSIM_total) / len(SSIM_total), i))


#def img_resize(img, t):
#    b, c, h, w = img.shape
#    size = 1 - 0.2*np.random.rand(1)
#    img = img.permute(2,3,1,0).cpu().numpy()
#    print(img.shape)
#    hr = skimage.transform.resize(img, (int(h*size), int(w*size), 3, 1)) 
#    hr = torch.from_numpy(hr).cuda().float()
#    lr = nn.functional.avg_pool2d(hr, kernel_size=4)
#    #hr = t(img)
#    #lr = nn.functional.avg_pool2d(hr, kernel_size=4)
#    return lr, hr

def test_train(args, model, test_dataloader):
    #model.eval()
    print('=====> test training begin!')
    #for i, data in enumerate(test_dataloader):
    img_lr = None
    #lr_list = []
    img_hr = None
    #t = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.RandomResizedCrop(size=224)])

    for i in range(args.start_epoch, args.end_epoch):
        if i>0 and i%99==0:
            #checkpoint(i)
            model.noise_amp = model.noise_amp/10.0
        model.update_lr()
        for j, data in enumerate(test_dataloader):
            img_lr = data['lr_image']
            img_lr = img_lr.expand(args.batch_size, -1, -1, -1)
            img_lr = img_lr.cuda().clone().float()
            
            img_hr = data['hr_image']
            img_hr = img_hr.expand(args.batch_size, -1, -1, -1)
            img_hr = img_hr.cuda().clone().float()

            #img_lr, img_hr = img_resize(img_hr, t)
            #print(img_hr.size())

            img_llr = nn.functional.avg_pool2d(img_lr, kernel_size=4)
            #img_hlr = nn.functional.avg_pool2d(img_hr, kernel_size=2)
            #img_lllr = nn.functional.avg_pool2d(img_llr, kernel_size=2)

            model.set_train_data(img_llr, img_llr, img_lr)
            model.set_ground_truth(img_lr, img_lr)
            model.optimize()

            #model.set_train_data(img_lr, img_lr, img_hr, img_hr)
            #model.set_ground_truth(img_hr, img_hr)
            #model.optimize()
            
            #if i==0:
            #    lr_list.append(img_llr)
            #    for j in range(30):
            #        m = nn.Upsample(size=[22+2*j, 22+2*j],mode='bilinear',align_corners=True)
            #        lr_list.append(m(img_lr))
            #        output = m(img_lr).cpu()[0].permute(1,2,0).detach().numpy()
            #        output = skimage.img_as_ubyte(output)
            #        skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'lr_list_{}.png'.format(j)), output)

            #model.set_train_data(lr_list[0:-1])
            #model.set_ground_truth(lr_list[1:])
            #model.optimize()


            scalars = model.get_current_scalars()
            print('epoches', i, 'step', j, scalars)
    
    #hr  = model.net_sr(img_lr)
    output  = model.net_sr(img_lr) + model.upsample_4(img_lr)
    res = model.net_G(output)
    output = output + res
    #output  = model.net_sr(img_hr)
    #noise = torch.randn(model.opts.batch_size, model.opts.n_colors, model.opts.im_crop_H, model.opts.im_crop_W).cuda() * model.noise_amp
    #print(model.noise_amp)
    #output = model.net_G(noise)

    #noise = torch.randn(model.opts.batch_size, model.opts.n_colors, model.opts.im_crop_H, model.opts.im_crop_W).cuda() * model.noise_amp
    

    #output  = model.net_sr(img_lr)
    #output  = model.net_sr(output)

    #lr_feature_head    = model.net_Feature_Head(img_hr)
    #lr_content_feature = model.net_Feature_extractor(lr_feature_head)
    #lr_content_output = lr_feature_head + lr_content_feature
    #hr_img = model.net_Upscalar(lr_content_output)    
    #res   = model.net_Ghr(hr_img)
    #output = hr_img + res

    #output = model.net_G.upscale_layers[0](lr_list[0])
    
    #for i in range(0, model.net_G.n_blocks):
    #    if i==0:
    #        x = torch.cat([lr_list[i], lr_list[i]], dim=1)
    #        output = model.net_G.upscale_layers[i](x)
    #    else:
    #        x = torch.cat([output, lr_list[i]], dim=1)
    #        output = model.net_G.upscale_layers[i](x)

    #output = model.net_G.upscale_layers[-1](output)
    #print(output.shape)
    


    PSNR_value = PSNR(output.data, img_hr)
    print('PSNR: {}'.format(PSNR_value))

    output = output.cpu()[0].permute(1,2,0).detach().numpy()
    output[output>1]=1
    output[output<-1]=-1
    output = skimage.img_as_ubyte(output)
    skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'SR_{}.png'.format('testtrain_SRNet_lr')), output)

    #mask = skimage.img_as_ubyte(model.d_mask[0].cpu().detach().numpy())
    #mask = skimage.img_as_ubyte(model.d_mask.float().cpu()[0].permute(1,2,0).numpy())
    #print(mask.shape)
    #skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'SR_{}.png'.format('mask')), mask)

    #mask = torch.rand([1,1,320,320]).cuda()
    #mask[mask<0.5] = 0
    #mask[mask!=0] = 1
    #print(mask)
    #mask = mask.repeat(1,3,1,1)
    #mask_hr = img_hr * model.d_mask
    #mask_hr = mask_hr.cpu()[0].permute(1,2,0).detach().numpy()
    #print(mask_hr)
    #skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'SR_{}.png'.format('mask_hr')), skimage.img_as_ubyte(mask_hr))


    #mask_lr = nn.functional.avg_pool2d(img_hr*model.d_mask, kernel_size=model.opts.scale).cpu()[0].permute(1,2,0).detach().numpy()
    #skimage.io.imsave(os.path.join(args.result_dir, 'SR', 'SR_{}.png'.format('mask_lr')), skimage.img_as_ubyte(mask_lr))


def test_flow(args, model, test_dataloader):
    MSE_total = []
    Loss_function = L2()
    print('=====> test flow begin!')
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            #torch.Size([1, args.n_colors, args.im_crop_H, args.im_crop_W])
            img_ref = data['image_center']
            image_ref = img_ref.expand(args.batch_size, -1, -1, -1)
            image_ref = (image_ref.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
            #print('image_ref', image_ref.size())
            
            #img_flow_gt = skimage.img_as_ubyte(torch.squeeze(img_ref).permute(1,2,0).numpy())
            img_flow_gt = skimage.img_as_ubyte(img_ref[0].permute(1,2,0).numpy())
            if args.save_result:
                os.makedirs(os.path.join(args.result_dir, 'flow_hr'), exist_ok=True)
                skimage.io.imsave(os.path.join(args.result_dir, 'flow_hr', '{}.png'.format(i)), img_flow_gt)

            #torch.Size([1, args.batch_size-1, args.n_colors, args.im_crop_H,, args.im_crop_W])
            img_oth = data['image_others']
            img_oth = torch.squeeze(img_oth, 0)
            image_oth = (img_oth.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
            #print(image_oth.size())

            #img_flow_ref = skimage.img_as_ubyte(torch.squeeze(img_oth).permute(1,2,0).numpy())
            img_flow_ref = skimage.img_as_ubyte(img_oth[0].permute(1,2,0).numpy())
            if args.save_result:
                os.makedirs(os.path.join(args.result_dir, 'flow_ref'), exist_ok=True)
                skimage.io.imsave(os.path.join(args.result_dir, 'flow_ref', '{}.png'.format(i)), img_flow_ref)

            # flow for groud truth
            # [80, 40, 20, 10, 5]
            flows_ref_to_other = model.net_flow(image_ref, image_oth)
            flows_other_to_ref = model.net_flow(image_oth, image_ref)
            
            flow_ref_to_other = model.upsample_4(flows_ref_to_other[0])*20.0
            flow_other_to_ref = model.upsample_4(flows_other_to_ref[0])*20.0
            print(flow_ref_to_other.shape)
            #flow_ref_to_other = flows_ref_to_other[0]*20.0
            #flow_other_to_ref = flows_other_to_ref[0]*20.0

            hr_mask_others = model.mask_fn(flow_ref_to_other)
            hr_mask_ref = model.mask_fn(flow_other_to_ref)
            #hr_mask_others = model.net_mask(flow_ref_to_other)
            #hr_mask_ref = model.net_mask(flow_other_to_ref)
            #print(hr_mask_ref.min(), hr_mask_ref.max())
            #hr_mask_ref[hr_mask_ref>1]=1
            #hr_mask_ref[hr_mask_ref<-1]=-1

            img_flow_mask = skimage.img_as_ubyte(hr_mask_ref.float().cpu()[0].permute(1,2,0).numpy())
            if args.save_result:
                os.makedirs(os.path.join(args.result_dir, 'flow_mask'), exist_ok=True)
                skimage.io.imsave(os.path.join(args.result_dir, 'flow_mask', '{}.png'.format(i)), img_flow_mask)

            #syn_hr_img_ref, _ = warp_image_flow(image_oth, flow_ref_to_other)
            #syn_hr_img_oth, _ = warp_image_flow(image_ref, flow_other_to_ref)
            Backward_warp_layer = Backward_warp()

            syn_hr_img_ref = Backward_warp_layer(image_oth, flow_ref_to_other)

            syn_numpy = syn_hr_img_ref.cpu()[0].permute(1,2,0).numpy()
            syn_numpy[syn_numpy>1]=1
            syn_numpy[syn_numpy<-1]=-1

            img_syn = skimage.img_as_ubyte(syn_numpy)
            img_mask_syn = skimage.img_as_ubyte(syn_numpy*(hr_mask_ref.float().cpu()[0].permute(1,2,0).numpy()))
            if args.save_result:
                os.makedirs(os.path.join(args.result_dir, 'flow_syn'), exist_ok=True)
                skimage.io.imsave(os.path.join(args.result_dir, 'flow_syn', 'syn_{}.png'.format(i)), img_syn)

                os.makedirs(os.path.join(args.result_dir, 'flow_mask_syn'), exist_ok=True)
                skimage.io.imsave(os.path.join(args.result_dir, 'flow_mask_syn', 'mask_syn_{}.png'.format(i)), img_mask_syn)
            
            MSE_loss = Loss_function(syn_hr_img_ref, image_ref, hr_mask_ref, mean=True)
            MSE_total.append(MSE_loss)
            print('MSE_loss: {} for patch {}'.format(MSE_loss, i))            
            print('Average MSE_loss: {} for {} patches'.format(sum(MSE_total) / len(MSE_total), i))

# ========================================================== #
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--is_training', type=bool, default=False)
    parser.add_argument('--use_pretrained_model', type=bool, default=False)
    parser.add_argument('--test_dir', type=str, default='/local/home/yuanhao/thesis/super_resolution_image/dataset/satellite/testing/')
    #parser.add_argument('--adv_data_dir', type=str, default= '/local/home/yuanhao/thesis/super_resolution_image/dataset/satellite/adv')
    parser.add_argument('--checkpoint_dir', type=str, default='/local/home/yuanhao/thesis/super_resolution_image/tem/EDSR_PWC_GAN')
    parser.add_argument('--model_dir', type=str, default='/local/home/yuanhao/scratch/tem/pretrained_models')
    parser.add_argument('--result_dir', type=str, default='/local/home/yuanhao/thesis/super_resolution_image/Results/image/')

    parser.add_argument('--test_flow', type=bool, default=False)
    parser.add_argument('--test_lr', type=bool, default=False)
    #parser.add_argument('--test_d', type=bool, default=False)
    parser.add_argument('--test_new', type=bool, default=False)
    parser.add_argument('--test_train', type=bool, default=False)
    parser.add_argument('--save_result', type=bool, default=False)
    parser.add_argument('--have_gt', type=bool, default=False)
   
    args = parser.parse_args()

    if args.model_name == 'FlowSRNet':
        model = FlowSRNet(args)
    elif args.model_name == 'FlowSRGNet':
        model = FlowSRGNet(args)
    elif args.model_name == 'FlowCircleSRGNet':
        model = FlowCircleSRGNet(args)
    elif args.model_name == 'SRNet':
        model = SRNet(args)
    elif args.model_name == 'SRGNet':
        model = SRGNet(args)
    elif args.model_name == 'UnSRNet':
        model = UnSRNet(args)
    elif args.model_name == 'UnFlowSRNet':
        model = UnFlowSRNet(args)
    elif args.model_name == 'UnFlowSRGNet':
        model = UnFlowSRGNet(args)
    elif args.model_name == 'UnFusionFlowSRNet':
        model = UnFusionFlowSRNet(args)
    elif args.model_name == 'FusionFlowSRNet':
        model = FusionFlowSRNet(args)
    elif args.model_name == 'CircleSRNet':
        model = CircleSRNet(args)
    elif args.model_name == 'FlowNet':
        model = FlowNet(args)
    elif args.model_name == 'CircleFlowSRNet':
        model = CircleFlowSRNet(args)
    elif args.model_name == 'TestSRNet':
        model = TestSRNet(args)
    elif args.model_name == 'TestCircleSRNet':
        model = TestCircleSRNet(args)
    elif args.model_name == 'TestGNet':
        model = TestGNet(args)
    elif args.model_name == 'TestContinueNet':
        model = TestContinueNet(args)
    else:
        raise ValueError('no model named: {}'.format(args.model_name))

    transform = transforms.Compose([transforms.Normalize()])
    transform = transforms.Compose([transforms.Normalize()])
    if args.model_name.find('Un') >= 0:
        dataset = Unsupervised_Dataset_Loader(args.test_dir,args.batch_size, args.scale, args.im_crop_H,args.im_crop_W,transform,args.random_crop)
    else:
        #dataset = Dataset_Loader(args.test_dir, args.adv_data_dir,args.batch_size,args.im_crop_H,args.im_crop_W,transform,args.random_crop)
        dataset = Dataset_Loader(args.test_dir, args.batch_size, args.im_crop_H, args.im_crop_W, transform, args.random_crop)

    if args.test_new:
        #dataset = Dataset_Loader(args.test_dir, args.adv_data_dir,args.batch_size,args.im_crop_H,args.im_crop_W,transform,args.random_crop)
        test_dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.batch_size, drop_last=True)
        test(args, model, test_dataloader)

    if args.test_lr:
        testset = Test_Loader(args.result_dir, transform)
        test_dataloader = torch.utils.data.DataLoader(testset, num_workers=args.batch_size, drop_last=True)
        test_lr(args, model, test_dataloader)

    if args.test_flow:
        #dataset = Dataset_Loader(args.test_dir, args.adv_data_dir,args.batch_size,args.im_crop_H,args.im_crop_W,transform,args.random_crop)
        test_dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.batch_size, drop_last=True)
        test_flow(args, model, test_dataloader)

    if args.test_train:
        testset = Test_Loader(args.result_dir, transform)
        test_dataloader = torch.utils.data.DataLoader(testset, num_workers=args.batch_size, drop_last=True)
        test_train(args, model, test_dataloader)

    print("=====> test done!")




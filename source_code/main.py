import torch
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import datetime
import os

from option import get_arguments

from models.supervised import *
from models.unsupervised import *
from models.testtrain import *
import utils.image_transform as transforms

from dataloaders.loading_satellite_image import Dataset_Loader
from dataloaders.loading_unsupervised import Unsupervised_Dataset_Loader
from torch.utils.data.sampler import SubsetRandomSampler

def train(args, model, train_dataloader, logger):
	total_steps = -1
	val_losses = []
	PSNR_total = []
	for i in range(args.start_epoch, args.end_epoch):

		if i>=49 and i%50==0:
			checkpoint(i)
		model.update_lr()

		for j, data in enumerate(train_dataloader):
			img_ref = data['image_center']
			img_oth = data['image_others']
			img_adv_cen = data['img_adv_cen']
			img_adv_ref = data['img_adv_ref']
			#img_adv = data['image_adversarial']
			img_oth     = img_oth.squeeze(0)
			img_adv_ref = img_adv_ref.squeeze(0)
			img_ref     = img_ref.expand(args.batch_size, -1, -1, -1)
			#img_adv    = img_adv.expand(args.batch_size, -1, -1, -1)
			img_adv_cen = img_adv_cen.expand(args.batch_size, -1, -1, -1)

			#image_ref = (img_ref.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
			#adv_images = (img_adv.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
			#image_others = (img_oth.cuda())[:, :, :args.im_crop_H, :args.im_crop_W].clone().float()
			image_ref = img_ref.cuda().clone().float()
			image_others = img_oth.cuda().clone().float()

			#adv_images = img_adv.cuda().clone().float()
			image_adv_cen = img_adv_cen.cuda().clone().float()
			image_adv_ref = img_adv_ref.cuda().clone().float()

			#lr_image_ref = nn.functional.avg_pool2d(image_ref, kernel_size=args.scale)
			#lr_image_others = nn.functional.avg_pool2d(image_others, kernel_size=args.scale)
			if args.model_name.find('Un') >= 0:
				model.set_train_data(image_ref, image_others, image_adv_cen, image_adv_ref)
			else:
				lr_image_ref = nn.functional.avg_pool2d(image_ref, kernel_size=args.scale)
				lr_image_others = nn.functional.avg_pool2d(image_others, kernel_size=args.scale)
				model.set_train_data(lr_image_ref, lr_image_others, image_adv_cen, image_adv_ref)
				model.set_ground_truth(image_ref, image_others)
			model.optimize()

			total_steps = total_steps + 1
			if total_steps%args.freq_visual==0:
				scalars = model.get_current_scalars()
				for tag, value in scalars.items():
					logger.add_scalar(tag, value, total_steps)
				print('epoches', i, 'step', j, scalars)

	checkpoint(args.end_epoch)

# ========================================================== #
def checkpoint(epoch):
	model.save_checkpoint(str(epoch))
	print("Checkpoint saved to {}".format(args.checkpoint_dir))


if __name__ == '__main__':
	parser = get_arguments()
	parser.add_argument('--is_training', type=bool, default=True)
	parser.add_argument('--use_pretrained_model', type=bool, default=False)
	parser.add_argument('--data_dir', type=str, default='/cluster/scratch/huangyu/satellite/training/')
	#parser.add_argument('--adv_data_dir', type=str, default= '/cluster/scratch/huangyu/satellite/adv/')
	parser.add_argument('--checkpoint_dir', type=str, default='/cluster/scratch/huangyu/tem/EDSR_PWC_GAN/')
	parser.add_argument('--model_dir', type=str, default='/cluster/scratch/huangyu/tem/pretrained_models/')
	parser.add_argument('--description', type=str, default='explain the setting of the model')
	args = parser.parse_args()

	now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
	args.checkpoint_dir = os.path.join(args.checkpoint_dir, now)
	os.makedirs(args.checkpoint_dir, exist_ok=True)
	with open(os.path.join(args.checkpoint_dir, 'log.txt'), 'w') as f:
		f.write(now + '\n')
		for arg in vars(args):
			f.write('{} : {}\n'.format(arg, getattr(args, arg)))
		f.write('\n')
	
	print(args.checkpoint_dir)
	print('number of epoches: {}'.format(args.end_epoch))

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
	elif args.model_name == 'UnFlowSRG2Net':
		model = UnFlowSRG2Net(args)
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
	else:
		raise ValueError('no model named: {}'.format(args.model_name))

	transform = transforms.Compose([transforms.Normalize()])
	if args.model_name.find('Un') >= 0:
		dataset = Unsupervised_Dataset_Loader(args.data_dir,args.batch_size, args.scale, args.im_crop_H,args.im_crop_W,transform,args.random_crop)
	else:
		#dataset = Dataset_Loader(args.data_dir, args.adv_data_dir,args.batch_size,args.im_crop_H,args.im_crop_W,transform,args.random_crop)
		dataset = Dataset_Loader(args.data_dir, args.batch_size, args.im_crop_H, args.im_crop_W, transform, args.random_crop)
	train_sampler = SubsetRandomSampler(list(range(len(dataset))))
	train_dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, num_workers=args.batch_size, drop_last=True)
	logger = SummaryWriter(args.checkpoint_dir)
	train(args, model, train_dataloader, logger)
	print("=====>  Training %d epochs completed"%(args.end_epoch))
	logger.close()
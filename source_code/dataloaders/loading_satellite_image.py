from torch.utils.data import Dataset
from utils.functions import *
import numpy as np
import os
import os.path
#import numbers
import torch
import random
#import skimage
#from skimage.transform import downscale_local_mean


#IMG_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG',
#    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
#]

#def is_image_file(filename):
#    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Dataset_Loader(Dataset):

    def __init__(self, root_dir, size, crop_size_H, crop_size_W, transform=None, random_crop=False):

        self.transform = transform
        self.random_crop = random_crop
        self.crop_size_H = crop_size_H
        self.crop_size_W = crop_size_W
        assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir
        self.images = []
        #self.adversarial = []#[[0 for x in range(64)] for y in range(75)]
        #self.size = size
        self.batch_size = size
        self.len_images = 0
        #self.folders = 0

        for folder in sorted(os.listdir(root_dir)):
            self.images.append([])
            for f in sorted(os.listdir(os.path.join(root_dir, folder))):
                if is_image_file(f):
                    self.images[-1].append(os.path.join(root_dir, folder, f))
            self.len_images += len(self.images[-1])

        #for folder in sorted(os.listdir(adv_dir)):
        #    self.adversarial.append([])
        #    for f in sorted(os.listdir(os.path.join(adv_dir, folder))):
        #        if is_image_file(f):
        #            self.adversarial[-1].append(os.path.join(adv_dir, folder, f))

    def __len__(self):
        #return (self.folders) *  self.len_images
        # TODO define the length of dataset
        #if this value didn't return, dataset would be NoneType
        return self.len_images

    #def is_low_contrast(self, image_array, h1, w1, scale):
    #    image_array = image_array[h1: h1 + self.crop_size_H, w1:w1 + self.crop_size_W, :]
    #    image_array = skimage.img_as_ubyte(image_array)
    #    return skimage.exposure.is_low_contrast(downscale_local_mean(image_array, (scale, scale, 1)).astype(np.uint8), fraction_threshold=0.2)

    def __getitem__(self, idx):
        #folder_idx_train = 0 if len(self.images)<=1 else np.random.randint(0, len(self.images) - 1)
        #folder_idx_adv = 0 if len(self.adversarial)<=1 else np.random.randint(0, len(self.adversarial) - 1)
        folder_idx_train = np.random.randint(0, len(self.images))
        #folder_idx_adv   = np.random.randint(0, len(self.adversarial))

        np.random.shuffle(self.images[folder_idx_train]) #TODO shaffule the order of the images
        #c = np.random.randint(0, len(self.images[folder_idx_train])-1)
        c = np.random.randint(0, len(self.images[folder_idx_train])-self.batch_size)

        img_file_name_center = self.images[folder_idx_train][c]
        #img_file_name_others = self.images[folder_idx_train][c+1] #next frame
        img_file_name_others = self.images[folder_idx_train][c+1:c+1+self.batch_size] #next frame

        #adv_c = np.random.randint(0, len(self.adversarial[folder_idx_adv]))        
        #img_file_name_adversarial = self.adversarial[folder_idx_adv][adv_c]

        img_c = load_as_PIL_image(img_file_name_center)
        img_c = np.array(img_c).astype(np.float)

        img_others_np = []
        for i in range(self.batch_size):
            #img_others_np = io.load_as_PIL_image(img_file_name_others)
            #img_others_np = np.array(img_others_np).astype(np.float)
            img_others_np.append(np.array(load_as_PIL_image(img_file_name_others[i])))
        img_others_np = np.array(img_others_np).astype(np.float)

        #img_adv = load_as_PIL_image(img_file_name_adversarial)
        #img_adv = np.array(img_adv).astype(np.float)
        
        if self.transform:
            img_c = self.transform(img_c)
            #img_adv = self.transform(img_adv)
            img_others_np = self.transform(img_others_np)

        th = self.crop_size_H
        tw = self.crop_size_W

        h,w,c = img_c.shape
        w1 = int(round((w - tw) / 2.))
        h1 = int(round((h - th) / 2.))
        w2 = int(round((w - tw) / 2.))
        h2 = int(round((h - th) / 2.))

        if self.random_crop:
            assert w > tw and h > th, 'cropped size larger than the image size'
            random.seed()
            h1 = random.randint(0, h - th)
            w1 = random.randint(0, w - tw)

            h2 = random.randint(0, h - th)
            w2 = random.randint(0, w - tw)

            #while(self.is_low_contrast(img_c, h1, w1, 8)):
            while(is_low_contrast(img_c, h1, w1, th, tw, 8)):
                h1 = random.randint(0, h - th)
                w1 = random.randint(0, w - tw)

            #while(self.is_low_contrast(img_adv, h2, w2, 8)):
            while(is_low_contrast(img_c, h2, w2, th, tw, 8)):
                h2 = random.randint(0, h - th)
                w2 = random.randint(0, w - tw)


        img_center = img_c[h1: h1 + th, w1:w1 + tw, :]
        img_center = torch.from_numpy(img_center).float()
        img_center = img_center.permute(2,0,1)

        img_others = img_others_np[:, h1: h1 + th, w1:w1 + tw, :]
        img_others = torch.from_numpy(img_others).float()
        #img_others = img_others.permute(2,0,1)
        img_others = img_others.permute(0, 3, 1, 2)

        #TODO change the crop for adv images
        #img_adversarial = img_adv[h2: h2 + th, w2: w2 + tw, :]
        #img_adversarial = torch.from_numpy(img_adversarial).float()
        #img_adversarial = img_adversarial.permute(2, 0, 1)
        img_adv_cen = img_c[h2: h2+th, w2: w2+tw, :]
        img_adv_cen = torch.from_numpy(img_adv_cen).float()
        img_adv_cen = img_adv_cen.permute(2, 0, 1)

        img_adv_ref = img_others_np[:, h2: h2+th, w2:w2+tw, :]
        img_adv_ref = torch.from_numpy(img_adv_ref).float()
        img_adv_ref = img_adv_ref.permute(0, 3, 1, 2)

        sample = {'image_center': img_center, 'image_others' : img_others, 'img_adv_cen' : img_adv_cen, 'img_adv_ref' : img_adv_ref}
        return sample

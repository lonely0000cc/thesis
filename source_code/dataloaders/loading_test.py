from torch.utils.data import Dataset
from utils.functions import *
import numpy as np
import os
#import deep_nets_test_EDSRPWC_GAN.utils.file_io as io
import torch

#IMG_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG',
#    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
#]

#def is_image_file(filename):
#    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Test_Loader(Dataset):

    def __init__(self, root_dir, transform=None):

        self.transform = transform
        assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir
        self.lr_images = []
        self.hr_images = []

        for f in sorted(os.listdir(os.path.join(root_dir, 'LR', '4'))):
            if is_image_file(f):
                self.lr_images.append(os.path.join(root_dir, 'LR', '4', f))

        for f in sorted(os.listdir(os.path.join(root_dir, 'HR'))):
            if is_image_file(f):
                self.hr_images.append(os.path.join(root_dir, 'HR', f))

        assert len(self.lr_images) == len(self.hr_images), 'hr_images and lr_images length not equal'
        self.len_images = len(self.lr_images)

    def __len__(self):
        return self.len_images

    def __getitem__(self, idx):
        f_lr_image = self.lr_images[idx]
        f_hr_image = self.hr_images[idx]
        #f_lr_image = self.lr_images[0]
        #f_hr_image = self.hr_images[0]

        #lr_image = io.load_as_PIL_image(f_lr_image)
        #hr_image = io.load_as_PIL_image(f_hr_image)
        lr_image = load_as_PIL_image(f_lr_image)
        hr_image = load_as_PIL_image(f_hr_image)
        
        lr_image = np.array(lr_image).astype(np.float)
        hr_image = np.array(hr_image).astype(np.float)
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        lr_image = torch.from_numpy(lr_image).float()
        lr_image = lr_image.permute(2,0,1)

        hr_image = torch.from_numpy(hr_image).float()
        hr_image = hr_image.permute(2,0,1)

        sample = {'lr_image': lr_image, 'hr_image': hr_image}
        return sample




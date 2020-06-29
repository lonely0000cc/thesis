from osgeo import gdal
from PIL import Image
import numpy as np
import skimage
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/local/home/yuanhao/thesis/super_resolution_image/dataset/satellite/training')
parser.add_argument('--h', type=int, default=1920)
parser.add_argument('--w', type=int, default=1920)
parser.add_argument('--c', type=int, default=3)
args = parser.parse_args()
print(args.data_dir)

files = []
IMG_EXTENSION = ['.tif', '.tiff']

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSION)

for folder in sorted(os.listdir(args.data_dir)):
	files.append([])
	for f in sorted(os.listdir(os.path.join(args.data_dir, folder))):
		if(is_image_file(f)):
			files[-1].append(os.path.join(args.data_dir, folder, f))

for i in range(len(files)):
	for j in range(len(files[i])):
		ds = gdal.Open(files[i][j]).ReadAsArray()[:args.c, :, :]
		image = skimage.transform.resize(np.transpose(ds, (1,2,0)), [args.h, args.w, args.c], anti_aliasing=True, preserve_range=True)
		image = image.astype(np.uint8)
		print(i, j, files[i][j])
		print(i, j, os.path.join(os.path.dirname(files[i][j]), '{}.png'.format(j)))
		print(image.shape)				
		a = Image.fromarray(image)
		a.save(os.path.join(os.path.dirname(files[i][j]), '{}.png'.format(j)))
		#skimage.io.imsave(fname=os.path.join(os.path.dirname(files[i][j]), '{}.png'.format(j)), arr=image, plugin='pil')

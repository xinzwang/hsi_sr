"""
Make Images Even
"""
import glob
import numpy as np
from tqdm import tqdm



# data_path = '/data2/wangxinzhe/codes/datasets/ICVL/train/'
save_path = '/data2/wangxinzhe/codes/datasets/ICVL/test/'
# paths = glob.glob(data_path + '*.npy')

paths = [
	'/data2/wangxinzhe/codes/datasets/ICVL/test/mor_0328-1209-2.npy',
	'/data2/wangxinzhe/codes/datasets/ICVL/test/Master5000K.npy',
	'/data2/wangxinzhe/codes/datasets/ICVL/test/Master2900k.npy',
	'/data2/wangxinzhe/codes/datasets/ICVL/test/IDS_COLORCHECK_1020-1223.npy',
	'/data2/wangxinzhe/codes/datasets/ICVL/test/hill_0325-1219.npy',
	'/data2/wangxinzhe/codes/datasets/ICVL/test/CC_40D_2_1103-0917.npy',
	'/data2/wangxinzhe/codes/datasets/ICVL/test/bulb_0822-0903.npy'
]

# imgs = np.zeros(31)
for p in tqdm(paths):
	name = p.split('/')[-1]
	data = np.load(p).astype(np.float32)
	H,W,C = data.shape
	# if imgs is None:
	# 	imgs = np.mean(data, axis=(0,1))
	# else:
	# 	imgs += np.mean(data, axis=(0,1))

	hsi = data[0:H//2*2, 0:W//2*2, :]

	print(data.shape, hsi.shape)

	np.save(save_path + name, hsi)

# band_mean = imgs / len(paths)

# print(band_mean)


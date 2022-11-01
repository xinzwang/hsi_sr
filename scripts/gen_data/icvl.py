"""
Split train, test data of ICVL

We subtract the minimum value in the original data, because this value is likely to be the black level.

Band average:

-- ICVL train
[0.01832518 0.03254801 0.04365423 0.04891146 0.06562264 0.08431768
 0.09772718 0.10706355 0.11712462 0.11732849 0.12029859 0.122063
 0.11772784 0.12460217 0.12786401 0.12982089 0.13137352 0.1326661
 0.13456418 0.13321937 0.13643164 0.14330673 0.14481425 0.14151649
 0.14189706 0.13287783 0.12861345 0.12850192 0.12490193 0.10994423
 0.12022068]

-- ICVL test
[0.01729363 0.03049092 0.04092131 0.04601089 0.06195875 0.07968732
 0.0923645  0.10128357 0.11090829 0.11125512 0.11434484 0.11621738
 0.11185981 0.11808831 0.12127427 0.12320871 0.12480043 0.12683405
 0.12952175 0.12929031 0.13247538 0.1391692  0.14079932 0.13812297
 0.13853336 0.13039579 0.12614289 0.12616725 0.12223995 0.10811897
 0.11722904]
"""


import numpy as np
import glob
import h5py
from tqdm import tqdm
from utils import gen_panel_from_area, norm

data_path = '/data2/wangxinzhe/codes/datasets/ICVL/raw/'
train_path = '/data2/wangxinzhe/codes/datasets/ICVL/train/'
test_path = '/data2/wangxinzhe/codes/datasets/ICVL/test/'

panel_size=64
train_num = 100


paths = glob.glob(data_path+'*.mat')
np.random.shuffle(paths)

# train
max_=4095.0
min_=177.5

avg_channels = np.zeros(31)
for p in tqdm(paths[0:train_num]):
	name = p.split('/')[-1].split('.')[0]
	data = h5py.File(p, 'r')
	hsi = data['rad']
	hsi = np.array(hsi).transpose(1,2,0)	# CHW->HWC

	
	hsi = (hsi - min_) / (max_ - min_)


	panels = gen_panel_from_area(hsi, panel_size=panel_size, stride=panel_size, ratio=1)
	print('subs:%d %s' %(len(panels), name))
	cnt=0
	for img in panels:
		np.save(train_path + '%s_%d.npy'%(name, cnt), img)
		cnt += 1

	avg = np.mean(hsi, axis=(0,1))
	avg_channels += avg

print(avg_channels / len(paths[0:train_num]))


# test
avg_channels = np.zeros(31)
for p in tqdm(paths[train_num:]):
	name = p.split('/')[-1].split('.')[0]
	data = h5py.File(p, 'r')
	hsi = data['rad']
	hsi = np.array(hsi).transpose(1,2,0)	# CHW->HWC
	hsi = (hsi - min_) / (max_ - min_)

	np.save(test_path + '%s.npy'%(name), hsi)

	avg = np.mean(hsi, axis=(0,1))
	avg_channels += avg

print(avg_channels / len(paths[train_num:]))







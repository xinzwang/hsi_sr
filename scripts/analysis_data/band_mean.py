import glob
import numpy as np
from tqdm import tqdm


# data_path = '/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256.npy'
data_path = '/data2/wangxinzhe/codes/datasets/ICVL/train/'


def SingleBandMean(path):
	data = np.load(data_path)
	band_mean = np.mean(data, axis=(0,1,2))
	return band_mean

def MultiBandMean(path):
	paths = glob.glob(path + '*.npy')
	imgs = None
	for p in tqdm(paths):
		data = np.load(p)
		if imgs is None:
			imgs = data
		else:
			imgs += data
	band_mean = np.mean(imgs / len(paths), axis=(0,1))
	return band_mean

# print(SingleBandMean(data_path))
print(MultiBandMean(data_path))

# ICVL train
# [0.01957541, 0.03548769, 0.04804641, 0.0540534 , 0.07289123,
#  0.09405381, 0.1092032 , 0.11983354, 0.13126786, 0.13164204,
#  0.13511308, 0.13725129, 0.13227056, 0.14001403, 0.14365877,
#  0.14589214, 0.14789122, 0.14955386, 0.15195944, 0.15082051,
#  0.15452292, 0.16227495, 0.16411468, 0.16046677, 0.16094302,
#  0.15096227, 0.1462362 , 0.14621027, 0.14183682, 0.12467857,
#  0.13591873]

# [0.01957541 0.03548769 0.04804641 0.0540534  0.07289123 0.09405381
#  0.1092032  0.11983354 0.13126786 0.13164204 0.13511308 0.13725129
#  0.13227056 0.14001403 0.14365877 0.14589214 0.14789122 0.14955386
#  0.15195944 0.15082051 0.15452292 0.16227495 0.16411468 0.16046677
#  0.16094302 0.15096227 0.1462362  0.14621027 0.14183682 0.12467857
#  0.13591873]
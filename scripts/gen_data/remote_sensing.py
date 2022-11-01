import cv2
import numpy as np
from utils import gen_panel_from_area
from scipy.io import loadmat


# data_path = '/data2/wangxinzhe/codes/datasets/Pavia/cls/hsi.npy'
# save_train_path = '/data2/wangxinzhe/codes/datasets/Pavia/sr/train_x420_y230_N256.npy'
# save_test_path = '/data2/wangxinzhe/codes/datasets/Pavia/sr/test_x420_y230_N256.npy'
data_path = '/data2/wangxinzhe/codes/datasets/Salinas/Salinas_corrected.mat'
save_train_path = '/data2/wangxinzhe/codes/datasets/Salinas/train_x192_45_N128.npy'
save_test_path = '/data2/wangxinzhe/codes/datasets/Salinas/test_x192_45_N128.npy'

# Coordinates of upper left corner and lower right corner of the test image
# test_area = [(420, 230), (676, 486)]
test_area = [(192, 45), (320, 173)]

# hsi_raw = np.load(data_path)
hsi_raw = loadmat(data_path)['salinas_corrected']

# norm
max_ = hsi_raw.max()
min_ = hsi_raw.min()
hsi_raw = (hsi_raw - min_) / (max_ - min_)
hsi_raw = hsi_raw.astype(np.float32)

# test data
hsi_test = hsi_raw[test_area[0][0]:test_area[1][0], test_area[0][1]:test_area[1][1], ...]
print(hsi_test.shape)

# Cut full image to 4 area
area1 = hsi_raw[:test_area[0][0], ...]
area2 = hsi_raw[test_area[0][0]:test_area[1][0], :test_area[0][1], ...]
area3 = hsi_raw[test_area[0][0]:test_area[1][0], test_area[1][1]:, ...]
area4 = hsi_raw[test_area[1][0]:, ...]

res = gen_panel_from_area(area1, panel_size=64, stride=64, ratio=1)
res = res + gen_panel_from_area(area2, panel_size=64, stride=64, ratio=1)
res = res + gen_panel_from_area(area3, panel_size=64, stride=64, ratio=1)
res = res + gen_panel_from_area(area4, panel_size=64, stride=64, ratio=1)

# enhanced with down sample 0.7
res = res + gen_panel_from_area(area1, panel_size=64, stride=64, ratio=0.7)
res = res + gen_panel_from_area(area2, panel_size=64, stride=64, ratio=0.7)
res = res + gen_panel_from_area(area3, panel_size=64, stride=64, ratio=0.7)
res = res + gen_panel_from_area(area4, panel_size=64, stride=64, ratio=0.7)

res = np.array(res)

print('panel num:', res.shape[0])
print(res.shape)

np.save(save_train_path, res)
np.save(save_test_path, np.array([hsi_test]))
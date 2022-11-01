import cv2


# Gen train dataset
def gen_panel_from_area(area, panel_size=64, stride=64, ratio=1):
	# down sample
	if ratio != 1:
		assert ratio < 1, Exception('Ratio should be less than 1. Upsampling is not recommended')
		area = cv2.resize(area, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

	H, W, C = area.shape

	if not(panel_size<H and panel_size<W):
		return []
		# Exception('Panelsize is too large')

	P, Q = H // panel_size, W // panel_size
	res = []
	for i in range(P):
		for j in range(Q):
			x, y = i*panel_size, j * panel_size
			t = area[x:x+panel_size, y:y+panel_size, ...]
			res.append(t)
	return res

# Norm
def norm(img):
	max_ = img.max()
	min_ = img.min()
	return (img - min_) / (max_-min_)
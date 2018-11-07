from scipy.misc import imread
import numpy as np
# import matplotlib.pyplot as plt

img = imread('1/input.ppm')
height, width, _ = img.shape
print(img.shape)

gray_img = np.zeros(shape=img.shape[0:2], dtype=int)
out_img = np.zeros(shape=img.shape, dtype=int)
histogram = [0 for _ in range(256)]
cdf = [0 for _ in range(256)]

for i in range(height):
	for j in range(width):
		r = img[i][j][0]
		g = img[i][j][1]
		b = img[i][j][2]
		gray_img[i][j] = 0.21*r + 0.71*g + 0.07*b
		print('gray[%d]: %d' % 
			(i*width + j, gray_img[i][j]))
		histogram[gray_img[i][j]] += 1



cdf[0] = histogram[0]
for i in range(1, 256):
	cdf[i] = cdf[i - 1] + histogram[i]
	print('hist[%d]: %d' % (i, histogram[i]))
	print('cdf[%d]: %d' % (i, cdf[i]))
cdfmin = cdf[0]

def correct_color(val):
	t = 255 * (cdf[val] - cdfmin) / (width * height - cdfmin)
	return min(max(t, 0), 255)

for i in range(height):
	for j in range(width):
		for channel in range(3):
			out_img[i][j][channel] = correct_color(img[i][j][channel])
			# print('output[%d][%d][%d] color: %d; %.4f' % (i, j, channel, out_img[i][j][channel], out_img[i][j][channel]/255))


# original_out = imread('0/output.ppm')
# for i in range(height):
# 	for j in range(width):
# 		for channel in range(3):
# 			if out_img[i][j][channel] != original_out[i][j][channel]:
# 				print('diff: output[%d][%d][%d] %.4f; original_out[%d][%d][%d] %.4f' % 
# 					(i, j, channel, out_img[i][j][channel]/255,
# 						i, j, channel, original_out[i][j][channel]/255))

# print(out_img)

# plot = plt.imshow(out_img)
# plt.show()

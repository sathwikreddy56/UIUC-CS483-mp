from scipy.misc import imread
import numpy as np
import math
# import matplotlib.pyplot as plt

img = imread('0/input.ppm')
width, height, _ = img.shape
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
		# print('gray[%d]: %d' % 
		# 	(i*width + j, gray_img[i][j]))
		histogram[gray_img[i][j]] += 1


def p(x):
	return x/(width * height)


cdf[0] = p(histogram[0])
for i in range(1, 256):
	cdf[i] = cdf[i - 1] + p(histogram[i])
	print('cdf[%d]: %d' % (i, cdf[i] * (width * height)))
cdfmin = min(cdf)

def correct_color(val):
	t = 255 * (cdf[val] - cdfmin) / (1.0 - cdfmin)
	return min(max(t, 0), 255)

for i in range(height):
	for j in range(width):
		for c in range(3):
			out_img[i][j][c] = correct_color(img[i][j][c])

original_out = imread('0/output.ppm')
for i in range(height):
	for j in range(width):
		for channel in range(3):
			if abs(out_img[i][j][channel] - original_out[i][j][channel]) > 2:
				print('diff: output[%d][%d][%d] %d; original_out[%d][%d][%d] %d' % 
					(i, j, channel, out_img[i][j][channel],
						i, j, channel, original_out[i][j][channel]))

# plot = plt.imshow(out_img, cmap='gray')
# plt.show()

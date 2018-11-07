from scipy.misc import imread
import numpy as np
# import matplotlib.pyplot as plt

img = imread('1/output.ppm')
width, height, channel = img.shape

for i in range(height):
	for j in range(width):
		for c in range(channel):
			print('img[%d][%d][%d] = %d; %.4f' % 
				(i, j, c, img[i][j][c], img[i][j][c]/255))
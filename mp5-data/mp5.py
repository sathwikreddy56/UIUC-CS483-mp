import numpy as np

data = np.loadtxt('3/input.raw')[1:]
scan = [sum(data[:i + 1]) for i in range(len(data))]
print(data, 'len=%d' % len(data))
for i in range(len(scan)):
	print('scan[%d] = %.1f' % (i, scan[i]))
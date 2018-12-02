import numpy as np
from scipy import sparse

FOLDER_DATASET = '0/'

col = np.loadtxt(FOLDER_DATASET + 'col.raw', dtype=int, skiprows=1)
row = np.loadtxt(FOLDER_DATASET + 'row.raw', dtype=int, skiprows=1)
data = np.loadtxt(FOLDER_DATASET + 'data.raw', skiprows=1)
vec = np.loadtxt(FOLDER_DATASET + 'vec.raw', skiprows=1)
output = np.loadtxt(FOLDER_DATASET + 'output.raw', skiprows=1)

mat = sparse.coo_matrix((data, (row, col)))
print(mat.shape, len(vec))

print(mat.todense())

print(np.dot(mat.todense(), vec))
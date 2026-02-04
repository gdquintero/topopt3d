import pandas as pd
import numpy as np

nelemx = 5
nelemy = 4
nnodesx = 6
nnodesy = 5
nelem = 20
nodes = 30

groups = 4 + (nelemy - 1) * 2

nel1 = np.zeros(groups,dtype=int)
nel2 = np.zeros(groups,dtype=int)
inic = np.zeros(nodes * 2,dtype=int)

nel1[:2] = 8
nel1[-2:] = 8
nel1[2:-2] = 12

nel2[:2] = 12
nel2[-2:] = 12
nel2[2:-2] = 18

for i in range(1,groups):
    inic[i] = inic[i - 1] + nel1[i - 1]

inic[groups] = inic[groups-1] + nel1[-1]

for j in range(1,groups):
    inic[j + groups] = inic[j + groups - 1] + nel2[j - 1]

for i in range(1,nelemx - 1):
    inic[(i + 1) * groups:groups * (i + 2)] = sum(nel2) + inic[i * groups:groups * (i + 1)]

inic[groups * nelemx] = inic[groups * nelemx - 1] + nel2[-1]

for i in range(1,groups):
    inic[i + groups * nelemx] = inic[i + groups * nelemx - 1] + nel1[i - 1]

print(inic)
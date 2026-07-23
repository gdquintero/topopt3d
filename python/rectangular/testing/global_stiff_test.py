import pandas as pd
from get_rows_cols_na import get_rows_cols
from get_pos import get_pos
from global_stiff import global_stiff
import numpy as np
from elem_stiff import elem_stiff
import time as tm
import warnings
import matplotlib.pyplot as plt
from global_stiff_trad_test import global_stiff_trad

warnings.filterwarnings("ignore")

np.set_printoptions(threshold=np.inf)

# start = tm.time()
struct = pd.read_json("./python/rectangular/struct5x4.json")
kel = elem_stiff(struct)[:, :, 0]
print(kel)
kelc = np.ravel(kel)

print(kelc)

rows, cols, inic = get_rows_cols(struct)
rows = rows - 1
cols = cols - 1
pos = get_pos(struct, inic)

pos = pos - 1

K = global_stiff(struct, np.ones(int(struct.nelem)), 1, rows, cols, inic, kelc, pos)
Ktrad = global_stiff_trad(struct, kel)
# end = tm.time()

# print(end - start)
# plt.spy(K.toarray())
# plt.show()

print(K.toarray() == Ktrad)

# plt.spy(K.toarray())
# plt.show()
# print(K)

import pandas as pd
from get_rows_cols_na import get_rows_cols
from get_pos import get_pos
import matplotlib.pyplot as plt
from global_stiff import global_stiff
import numpy as np
from elem_stiff import elem_stiff
np.set_printoptions(threshold=np.inf)
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
K = global_stiff(struct, np.ones(int(struct.nelem)), 0.3, rows, cols, inic, np.ones(64), pos)

plt.spy(K.toarray())
plt.show()
# print(K)

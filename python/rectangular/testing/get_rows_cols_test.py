import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from get_rows_cols_na import get_rows_cols
import numpy as np

struct = pd.read_json("./python/rectangular/struct5x4.json")

row, col, inic = get_rows_cols(struct)
row = row - 1
col = col - 1

data = np.ones(inic[-1])

matrix = csr_matrix((data, (row, col)), shape=(60, 60))

plt.spy(matrix.toarray())

plt.show()

# print(struct)

import pandas as pd
from get_rows_cols_na import get_rows_cols
from get_pos import get_pos
import numpy as np
from linel import linel
np.set_printoptions(threshold=np.inf)

struct = pd.read_json("./python/rectangular/struct5x4.json")
realMatrix = pd.read_excel("get_pos_global_stiff_2D.xlsx", sheet_name="Hoja1")
lin = linel(struct)
lin = lin - 1
print(realMatrix)
row, cols, inic = get_rows_cols(struct)

matrixRealArray = np.array([], dtype=int)
for i in range(int(struct.nelem)):
    linEl = lin[:, i]
    matrixElArray = np.array([], dtype=int)
    for j in linEl:
        for k in linEl:
            matrixElArray = np.array([*matrixElArray, realMatrix.iloc[k, j]]) 

    matrixRealArray = np.array([*matrixRealArray, matrixElArray])


pos = get_pos(struct, inic)

print(pos)
print(" ")
print(matrixRealArray)
print(pos == matrixRealArray)
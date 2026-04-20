import pandas as pd
import numpy as np
from linel import linel
from get_rows_cols_na import get_rows_cols

struct  =  pd.read_json("./python/rectangular/struct5x4.json")
lin = linel(struct)
row, cols, inic = get_rows_cols(struct)

matrix = pd.read_excel("get_pos_global_stiff_2D.xlsx", sheet_name="Hoja1")
indexes = []
for i in range(int(struct["nelem"])):
    linElem = lin[:, i] - 1
    pasos=[]
    print(i)
    for j in linElem:
        indexesObtained = []
        for k in linElem:
            
            indexesObtained.append(matrix.iloc[int(k), int(j)] - inic[int(j)])
            # print(j, k)
        
        
        if(indexesObtained not in pasos):
            print(indexesObtained)

        pasos.append(indexesObtained)
        indexes.append(indexesObtained)
        

df = pd.DataFrame(data = indexes, columns = np.ones(8))
print(df.drop_duplicates())
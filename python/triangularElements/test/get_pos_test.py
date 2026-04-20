import pandas as pd
from get_rows_cols import get_rows_cols
from linel import linel
import numpy as np
from get_pos import get_pos

np.set_printoptions(threshold=np.inf)

struct = pd.read_json("./python/triangularElements/struct3x3.json")
matrixeven = pd.read_excel("./Ktrian5x5.xlsx", sheet_name="matrizpar")
matrixOdd = pd.read_excel("./Ktrian5x5.xlsx", sheet_name="matrizImpar")

matrixeven = matrixeven.astype(int)
matrixOdd = matrixOdd.astype(int)

matrix = matrixeven
lin = linel(struct).astype(int)
indexes = []
(row, col, inic) = get_rows_cols(struct)

position = np.array([])
# print(lin)
for i in range(int(struct["nelem"]/2)):
    linElemE = lin[:, 2*i + 1]  - 1
    linElemO = lin[:, 2*i] - 1

    arrayE = np.array([], dtype=int)
    arrayO = np.array([], dtype=int)
    for k in linElemO:
        for j in linElemO:
            arrayO = np.array([*arrayO, matrixOdd.iloc[j, k]], dtype=int)
    
    for k in linElemE:
        for j in linElemE:
            arrayE = np.array([*arrayE, matrixeven.iloc[j, k]], dtype=int)

    
    position = np.array([*position, arrayO, arrayE])
    


# print(position)           

pos = get_pos(struct, inic)

# print(pos)

print(position == pos)

# for i in range(int(struct["nelem"]/2)):
#     linElem = lin[:, 2*i + 1] - 1
#     pasos=[]
#     print(i)
#     for j in linElem:
#         indexesObtained = []
#         for k in linElem:
            
#             indexesObtained.append(matrix.iloc[int(k), int(j)] - inic[int(j)])
#             # print(j, k)
        
        
#         if(indexesObtained not in pasos):
#             print(indexesObtained)

#         pasos.append(indexesObtained)
#         indexes.append(indexesObtained)
        


# print(lin)


# indexes = []

# # print(matrix)
# for i in range(len(matrix)):
#     indexesObtained = []
#     for j in range(len(matrix)):
#         if  matrix.iloc[j,i] != 0:
            
#             indexesObtained.append(matrix.iloc[j, i] - inic[i])
#     print(indexesObtained)
#     indexes.append(indexesObtained)

# df = pd.DataFrame(data = indexes, columns = ["yo", "soy", "ñapi", "la", "piña", "parlante"])
# print(df.drop_duplicates())

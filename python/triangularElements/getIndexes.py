import pandas as pd
from get_rows_cols import get_rows_cols
from linel import linel
from collections import OrderedDict

struct = pd.read_json("./python/triangularElements/struct3x3.json")
matrix = pd.read_excel("./Ktrian5x5.xlsx", sheet_name="matrizImpar")
lin = linel(struct)
indexes = []
(row, col, inic) = get_rows_cols(struct)

print(matrix)

for i in range(int(struct["nelem"]/2)):
    linElem = lin[:, 2*i] - 1
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

df = pd.DataFrame(data = indexes, columns = ["yo", "soy", "ñapi", "la", "piña", "parlante"])
print(df.drop_duplicates())

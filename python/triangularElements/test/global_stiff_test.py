from elem_stiff import elem_stiff
import pandas as pd
from global_stiff import global_stiff
from get_rows_cols import get_rows_cols
from get_pos import get_pos
from numpy import ravel
from numpy import ones
from numpy import set_printoptions
from numpy import inf
# from matplotlib.pyplot import spy
# from matplotlib.pyplot import show
# from global_stiff_trad_test import global_stiff_trad
import warnings as wn
import time as tm
wn.filterwarnings("ignore")

set_printoptions(threshold=inf)
struct = pd.read_json("./python/triangularElements/struct3x3.json")


start = tm.time()
row, col, inic = get_rows_cols(struct)
# print(inic)

pos = get_pos(struct, inic)

# print(pos.shape)
# print(pos)

row = row - 1
col = col - 1
pos = pos - 1

kOdd, kEven = elem_stiff(struct)

kOddc = ravel(kOdd)
kEvenc = ravel(kEven)


K = global_stiff(struct, ones(int(struct.nelem)), 1, row, col, inic, kEvenc, kOddc, pos )

end = tm.time()

print(end - start)

# Ktrad = global_stiff_trad(struct, kOdd, kEven)


# Kar = K.toarray()
# print("Valor mio", "valor tradicional", "fila", "columna")
# for i in range(len(Kar)):
#     for j in range(len(Kar)):

#         if Kar[i, j] != Ktrad[i, j] : 
#             print(Kar[i, j], Ktrad[i, j], i, j)

# # KCoo = K.tocoo()

# # for r, c, v in zip(KCoo.row, KCoo.col, KCoo.data):
# #     print(f"({r}, {c})\t{v}")



# # # print(K)
# print(K == Ktrad)
# print(K.toarray())
# print(K.indices)
# spy(Ktrad)
# show()
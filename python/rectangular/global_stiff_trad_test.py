from linel import linel
import pandas as pd
from elem_stiff import elem_stiff
import time as tm
from numpy import eye
from numpy import zeros
from numpy import array
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# start = tm.time()

# struct = pd.read_json("./python/rectangular/struct5x4.json")
# kel = elem_stiff(struct)[:, :, 0]

def global_stiff_trad(struct, kel):
    Id = eye(2*int(struct.nodesNumber))
    lin = linel(struct)
    lin = lin - 1
    K = zeros((2*int(struct.nodesNumber), 2*int(struct.nodesNumber)))

    # P = zeros((2*int(struct.nodesNumber), 8))
    for i in range(int(struct.nelem)):
        P = array([Id[:, lin[0, i]],  Id[:, lin[1, i]], Id[:, lin[2, i]], Id[:, lin[3, i]], 
            Id[:, lin[4, i]], Id[:, lin[5, i]], Id[:, lin[6, i]], Id[:, lin[7, i]]])
        
        # print(P.transpose())
        K += P.transpose() @ kel @ P    
    return K

# end = tm.time()

# print(end - start)
# plt.spy(K)
# plt.show()
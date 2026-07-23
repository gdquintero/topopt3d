import pandas as pd
from linel import linel
from elem_stiff import elem_stiff
from numpy import eye
from numpy import zeros
from numpy import array
from matplotlib.pyplot import spy
from matplotlib.pyplot import show
import time as tm

def global_stiff_trad(struct, kodd, keven):
    Id = eye(2*int(struct.nodesNumber))
    lin = linel(struct)
    lin = lin - 1
    print(lin)
    K = zeros((2*int(struct.nodesNumber), 2*int(struct.nodesNumber)))

    # P = zeros((2*int(struct.nodesNumber), 8))
    for i in range(int(struct.nelem/2)):

        POdd = array([Id[lin[0, 2*i], :],  Id[lin[1, 2*i], :], Id[lin[2, 2*i], :], Id[lin[3, 2*i], :], Id[lin[4, 2*i], : ], Id[lin[5, 2*i], : ]])
        PEven = array([Id[lin[0, 2*i + 1], : ],  Id[lin[1, 2*i + 1], : ], Id[lin[2, 2*i + 1], : ], Id[lin[3, 2*i + 1], : ], Id[lin[4, 2*i + 1], : ], Id[lin[5, 2*i + 1], :]])
        
        # print(P.transpose())
        K += PEven.transpose() @ keven @ PEven +  POdd.transpose() @ kodd @ POdd
    return K



struct = pd.read_json("./python/triangularElements/struct3x3.json")

start = tm.time()
kodd, keven = elem_stiff(struct)

# print(linel(struct))
K = global_stiff_trad(struct, kodd, keven)

end = tm.time()

print(end - start)
# spy(K)
# show()
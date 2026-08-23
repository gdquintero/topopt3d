from linel import linel
from numpy import zeros
def global_stiff_tradv2(struct, kel, x, p):
    lin = linel(struct)
    lin = lin - 1
    K = zeros((2*int(struct["nodesNumber"]), 2*int(struct["nodesNumber"])))
    for k in range(int(struct["nelem"])):
        for i in range(8):
            for j in range(8):
                K[lin[i, k], lin[j, k]] += (x[k]**p)*kel[i, j]
    return K
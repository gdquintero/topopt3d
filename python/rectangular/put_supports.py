from numpy import zeros
from numpy import double
def put_supports(struct):
    structSupp = struct["supp"][0]
    linap = zeros(2*len(structSupp["node"]), dtype = double)
    counter = 0
    suppix = structSupp["ix"]
    suppiy = structSupp["iy"]
    for i in range(len(structSupp["node"])):
        if suppix[i] == 1:
            linap[counter] = 2*structSupp["node"][i] - 1
            counter += 1
        if suppiy[i] == 1:
            linap[counter] = 2*structSupp["node"][i] 
            counter += 1
    return linap

